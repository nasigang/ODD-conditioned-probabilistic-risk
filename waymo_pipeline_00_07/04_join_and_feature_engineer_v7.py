#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""04_join_and_feature_engineer_v7_patched_v8.py

v8 변경점(핵심):
- Perception object 중 TYPE_SIGN(예: stop_sign)을 TTC/overlap/hit_future 계산 대상에서 제외
  -> stop_sign은 ODD 메타데이터(has_stop_sign 등)로만 사용하고, risk interaction 후보에서 빠져야 정상임.
  -> 결과적으로 09 렌더에서 'TARGET(HIT)/(CONTACT)'로 표지판이 잡히는 현상이 사라짐.

권장:
- 이 스크립트로 04를 --overwrite 재생성 후 05→06→07을 다시 수행할 것.
"""

from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# -------------------------
# Canonicalization (Waymo-ish)
# -------------------------

def _canon_time(s: Any) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "unknown"
    t = str(s).strip().lower()
    if "dawn" in t or "dusk" in t:
        return "Dawn/Dusk"
    if "night" in t:
        return "Night"
    if "day" in t:
        return "Day"
    return "unknown"


def _canon_weather(s: Any) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "unknown"
    w = str(s).strip().lower()
    if "sun" in w or "clear" in w:
        return "Sunny"
    if "rain" in w:
        return "Rain"
    return "unknown"


def _canon_category(s: Any) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "unknown"
    c = str(s).strip().lower()
    if c in {"straight", "merge", "split", "intersection"}:
        return c
    return "unknown"


# -------------------------
# Condition schema (fixed dim)
# -------------------------

def build_odd_input_schema_v7(meta_df: pd.DataFrame) -> Dict[str, Any]:
    categorical = {
        "odd_time": ["Day", "Dawn/Dusk", "Night", "unknown"],
        "odd_weather": ["Sunny", "Rain", "unknown"],
        "odd_category": ["straight", "merge", "split", "intersection", "unknown"],
        "traffic_density_bin": ["low", "mid", "high", "unknown"],
        "infra_density_bin": ["low", "mid", "high", "unknown"],
        "occlusion_bin": ["low", "mid", "high", "unknown"],
        "speed_bin": ["low", "mid", "high", "unknown"],
    }
    binary = [
        "has_crosswalk",
        "has_stop_sign",
        "has_speed_bump",
        "has_driveway",
    ]
    cont = [
        # ego dynamics (segment stats)
        "ego_speed_mps_mean",
        "ego_speed_mps_std",
        "ego_speed_mps_p95",
        "ego_accel_mps2_mean",
        "ego_accel_mps2_std",
        "ego_accel_mps2_p95",
        "ego_yaw_rate_rps_mean",
        "ego_yaw_rate_rps_p95",

        # traffic / interaction complexity (segment stats)
        "interaction_object_count_mean",
        "interaction_vehicle_count_mean",
        "interaction_pedestrian_count_mean",
        "interaction_cyclist_count_mean",
        "near_object_count_30m_mean",
        "interaction_type_entropy",

        # perception quality / visibility (segment stats)
        "occlusion_low_points_ratio_mean",
        "occlusion_low_points_ratio_p95",
        "mean_lidar_points_in_box_mean",

        # infrastructure density / speed limit (segment stats)
        "infra_density_mean",
        "infra_density_p95",
        "speed_limit_mph_mean",
        "intersection_active_ratio",

        # static map counts (segment-level)
        "map_lane_count",
        "map_stop_sign_count",
        "map_crosswalk_count",
        "map_speed_bump_count",
        "map_driveway_count",

        # topology probabilities (segment-level)
        "topo_p_intersection",
        "topo_p_merge",
        "topo_p_split",
        "topo_p_straight",
    ]
    z = {}
    for c in cont:
        x = pd.to_numeric(meta_df.get(c, pd.Series([], dtype=float)), errors="coerce")
        mu = float(np.nanmean(x.values)) if len(x) else 0.0
        sd = float(np.nanstd(x.values)) if len(x) else 1.0
        if not np.isfinite(sd) or sd < 1e-6:
            sd = 1.0
        if not np.isfinite(mu):
            mu = 0.0
        z[c] = {"mean": mu, "std": sd}

    vector_order: List[str] = []
    for k, vals in categorical.items():
        vector_order.extend([f"{k}={v}" for v in vals])
    vector_order.extend(binary)
    vector_order.extend(cont)

    return {
        "version": "v7.1",
        "categorical": categorical,
        "binary": binary,
        "continuous": {"names": cont, "zscore": z},
        "vector_order": vector_order,
        "cond_dim": len(vector_order),
    }


def encode_condition_vector(row: pd.Series, schema: Dict[str, Any]) -> np.ndarray:
    cat = schema["categorical"]
    bin_keys = schema["binary"]
    cont = schema["continuous"]["names"]
    z = schema["continuous"]["zscore"]

    parts: List[float] = []

    for k, vals in cat.items():
        v = row.get(k, "unknown")
        if k == "odd_time":
            v = _canon_time(v)
        elif k == "odd_weather":
            v = _canon_weather(v)
        elif k == "odd_category":
            v = _canon_category(v)
        v = v if v in vals else "unknown"
        parts.extend([1.0 if v == vv else 0.0 for vv in vals])

    for b in bin_keys:
        x = row.get(b, 0)
        try:
            xf = float(x)
            parts.append(1.0 if np.isfinite(xf) and xf > 0.0 else 0.0)
        except Exception:
            parts.append(0.0)

    for c in cont:
        x = row.get(c, np.nan)
        try:
            xf = float(x)
        except Exception:
            xf = np.nan
        if not np.isfinite(xf):
            parts.append(0.0)
        else:
            parts.append((xf - z[c]["mean"]) / z[c]["std"])

    return np.asarray(parts, dtype=np.float32)


# -------------------------
# SAT helper (single pair OBB overlap)
# -------------------------

def sat_obb_overlap_2d(
    c1: np.ndarray, yaw1: float, hl1: float, hw1: float,
    c2: np.ndarray, yaw2: float, hl2: float, hw2: float,
) -> bool:
    c1 = np.asarray(c1, dtype=np.float32)
    c2 = np.asarray(c2, dtype=np.float32)
    t = c2 - c1

    c1c = float(np.cos(yaw1)); s1s = float(np.sin(yaw1))
    c2c = float(np.cos(yaw2)); s2s = float(np.sin(yaw2))

    a1 = np.array([c1c, s1s], dtype=np.float32)
    a2 = np.array([-s1s, c1c], dtype=np.float32)
    b1 = np.array([c2c, s2s], dtype=np.float32)
    b2 = np.array([-s2s, c2c], dtype=np.float32)

    def r_proj(axis: np.ndarray, ax1: np.ndarray, ax2: np.ndarray, hl: float, hw: float) -> float:
        return float(abs(np.dot(axis, ax1)) * hl + abs(np.dot(axis, ax2)) * hw)

    axes = (a1, a2, b1, b2)
    for axis in axes:
        dist = float(abs(np.dot(t, axis)))
        ra = r_proj(axis, a1, a2, hl1, hw1)
        rb = r_proj(axis, b1, b2, hl2, hw2)
        if dist > (ra + rb):
            return False
    return True


# -------------------------
# Pairwise risk geometry
# -------------------------

def compute_all_pairs_ttc(
    pos: np.ndarray,
    vel: np.ndarray,
    length_m: np.ndarray,
    width_m: np.ndarray,
    heading_rad: np.ndarray,
    *,
    eps_speed: float = 1e-6,
    eps_range: float = 1e-3,
    bbox_margin_m: float = 0.3,
    overlap_margin_m: float = 0.15,
    closing_speed_thr_mps: float = 0.0,
    ttc_max_s: float = float("inf"),
    ego_future_hit_sat: bool = True,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    pos = np.asarray(pos, dtype=np.float32)
    vel = np.asarray(vel, dtype=np.float32)
    length_m = np.asarray(length_m, dtype=np.float32)
    width_m = np.asarray(width_m, dtype=np.float32)
    heading_rad = np.asarray(heading_rad, dtype=np.float32)

    N = int(pos.shape[0])
    if N == 0:
        z = np.zeros((0, 0), dtype=np.float32)
        z2 = np.zeros((0, 0, 2), dtype=np.float32)
        z8 = np.zeros((0, 0), dtype=np.int8)
        return z, z2, z2, z, z, z8, z8, z8, z8, z, z, z, z

    r = pos[None, :, :] - pos[:, None, :]
    dv = vel[None, :, :] - vel[:, None, :]

    dist = np.sqrt(r[..., 0] ** 2 + r[..., 1] ** 2).astype(np.float32)
    vv = (dv[..., 0] ** 2) + (dv[..., 1] ** 2)
    rel_speed = np.sqrt(vv).astype(np.float32)

    dot_rv = (r[..., 0] * dv[..., 0]) + (r[..., 1] * dv[..., 1])
    closing_speed = (-dot_rv) / (dist + eps_range)

    approaching = ((rel_speed > eps_speed) & (closing_speed > closing_speed_thr_mps)).astype(np.int8)

    # u along dv
    u = np.zeros_like(dv, dtype=np.float32)
    m = rel_speed > eps_speed
    u[m, 0] = dv[m, 0] / rel_speed[m]
    u[m, 1] = dv[m, 1] / rel_speed[m]
    u_perp = np.stack([-u[..., 1], u[..., 0]], axis=-1).astype(np.float32)

    # object local axes
    ch = np.cos(heading_rad).astype(np.float32)
    sh = np.sin(heading_rad).astype(np.float32)
    e_long = np.stack([ch, sh], axis=-1)   # (N,2)
    e_lat = np.stack([-sh, ch], axis=-1)   # (N,2)

    hx = 0.5 * length_m
    hy = 0.5 * width_m

    def _absdot_axis_i(v_ij: np.ndarray, axis_i: np.ndarray) -> np.ndarray:
        return np.abs(v_ij[..., 0] * axis_i[:, None, 0] + v_ij[..., 1] * axis_i[:, None, 1])

    def _absdot_axis_j(v_ij: np.ndarray, axis_j: np.ndarray) -> np.ndarray:
        return np.abs(v_ij[..., 0] * axis_j[None, :, 0] + v_ij[..., 1] * axis_j[None, :, 1])

    au_el_i = _absdot_axis_i(u, e_long)
    au_et_i = _absdot_axis_i(u, e_lat)
    au_el_j = _absdot_axis_j(u, e_long)
    au_et_j = _absdot_axis_j(u, e_lat)

    ap_el_i = _absdot_axis_i(u_perp, e_long)
    ap_et_i = _absdot_axis_i(u_perp, e_lat)
    ap_el_j = _absdot_axis_j(u_perp, e_long)
    ap_et_j = _absdot_axis_j(u_perp, e_lat)

    hx_i = hx[:, None]
    hy_i = hy[:, None]
    hx_j = hx[None, :]
    hy_j = hy[None, :]

    ext_i_u = hx_i * au_el_i + hy_i * au_et_i
    ext_j_u = hx_j * au_el_j + hy_j * au_et_j
    ext_i_p = hx_i * ap_el_i + hy_i * ap_et_i
    ext_j_p = hx_j * ap_el_j + hy_j * ap_et_j

    L_eff = (ext_i_u + ext_j_u).astype(np.float32)
    W_eff = (ext_i_p + ext_j_p).astype(np.float32)

    s = (r[..., 0] * u[..., 0] + r[..., 1] * u[..., 1]).astype(np.float32)
    l = (r[..., 0] * u_perp[..., 0] + r[..., 1] * u_perp[..., 1]).astype(np.float32)
    d_perp = np.abs(l).astype(np.float32)

    L_future = (L_eff + bbox_margin_m).astype(np.float32)
    W_future = (W_eff + bbox_margin_m).astype(np.float32)

    corridor = (rel_speed > eps_speed) & (d_perp <= W_future)

    # ---- SAT overlap test (all-pairs current physical overlap) ----
    shrink = max(float(overlap_margin_m), 0.0) * 0.5
    hx_s = np.maximum(hx - shrink, 0.0).astype(np.float32)
    hy_s = np.maximum(hy - shrink, 0.0).astype(np.float32)

    hx_si = hx_s[:, None]
    hy_si = hy_s[:, None]
    hx_sj = hx_s[None, :]
    hy_sj = hy_s[None, :]

    abs_elielj = np.abs(e_long[:, None, 0] * e_long[None, :, 0] + e_long[:, None, 1] * e_long[None, :, 1]).astype(np.float32)
    abs_elieltj = np.abs(e_long[:, None, 0] * e_lat[None, :, 0] + e_long[:, None, 1] * e_lat[None, :, 1]).astype(np.float32)
    abs_etielj = np.abs(e_lat[:, None, 0] * e_long[None, :, 0] + e_lat[:, None, 1] * e_long[None, :, 1]).astype(np.float32)
    abs_etietj = np.abs(e_lat[:, None, 0] * e_lat[None, :, 0] + e_lat[:, None, 1] * e_lat[None, :, 1]).astype(np.float32)

    t_A1 = np.abs(r[..., 0] * e_long[:, None, 0] + r[..., 1] * e_long[:, None, 1]).astype(np.float32)
    r_i_A1 = hx_si
    r_j_A1 = hx_sj * abs_elielj + hy_sj * abs_elieltj
    sep_A1 = t_A1 > (r_i_A1 + r_j_A1)

    t_A2 = np.abs(r[..., 0] * e_lat[:, None, 0] + r[..., 1] * e_lat[:, None, 1]).astype(np.float32)
    r_i_A2 = hy_si
    r_j_A2 = hx_sj * abs_etielj + hy_sj * abs_etietj
    sep_A2 = t_A2 > (r_i_A2 + r_j_A2)

    t_B1 = np.abs(r[..., 0] * e_long[None, :, 0] + r[..., 1] * e_long[None, :, 1]).astype(np.float32)
    r_j_B1 = hx_sj
    r_i_B1 = hx_si * abs_elielj + hy_si * abs_etielj
    sep_B1 = t_B1 > (r_i_B1 + r_j_B1)

    t_B2 = np.abs(r[..., 0] * e_lat[None, :, 0] + r[..., 1] * e_lat[None, :, 1]).astype(np.float32)
    r_j_B2 = hy_sj
    r_i_B2 = hx_si * abs_elieltj + hy_si * abs_etietj
    sep_B2 = t_B2 > (r_i_B2 + r_j_B2)

    overlap_now = (~(sep_A1 | sep_A2 | sep_B1 | sep_B2)).astype(np.int8)

    # ---- HIT_FUTURE (base corridor approximation, all-pairs) ----
    hit_future = (
        (overlap_now == 0)
        & corridor
        & (s < -L_future)
        & (closing_speed > closing_speed_thr_mps)
    ).astype(np.int8)

    dtc = np.full((N, N), np.inf, dtype=np.float32)
    ttc = np.full((N, N), np.inf, dtype=np.float32)

    m_ov = overlap_now.astype(bool)
    dtc[m_ov] = 0.0
    ttc[m_ov] = 0.0

    m_hit = hit_future.astype(bool)
    if np.any(m_hit):
        dtc[m_hit] = (-L_future[m_hit] - s[m_hit]).astype(np.float32)
        ttc[m_hit] = (dtc[m_hit] / rel_speed[m_hit]).astype(np.float32)

    # ---- ego-pair HIT_FUTURE override: t* + SAT at future time ----
    if ego_future_hit_sat and N >= 2:
        ego = 0

        for j in range(1, N):
            if overlap_now[ego, j] == 0:
                hit_future[ego, j] = 0
                hit_future[j, ego] = 0
                ttc[ego, j] = np.inf
                ttc[j, ego] = np.inf
                dtc[ego, j] = np.inf
                dtc[j, ego] = np.inf

        expand = max(float(bbox_margin_m), 0.0) * 0.5
        ego_hl_f = float(max(0.5 * length_m[ego] + expand, 0.0))
        ego_hw_f = float(max(0.5 * width_m[ego] + expand, 0.0))

        for j in range(1, N):
            if overlap_now[ego, j] == 1:
                continue

            v_rel = vel[j] - vel[ego]
            r_rel = pos[j] - pos[ego]
            vv_s = float(np.dot(v_rel, v_rel))
            if vv_s <= float(eps_speed) ** 2:
                continue

            t_star = float(-np.dot(r_rel, v_rel) / (vv_s + 1e-12))
            if not np.isfinite(t_star) or t_star <= 0.0:
                continue
            if np.isfinite(ttc_max_s) and t_star > float(ttc_max_s):
                continue

            cs = float(closing_speed[ego, j])
            if cs <= float(closing_speed_thr_mps):
                continue

            ego_c_t = pos[ego] + vel[ego] * t_star
            obj_c_t = pos[j] + vel[j] * t_star

            obj_hl_f = float(max(0.5 * length_m[j] + expand, 0.0))
            obj_hw_f = float(max(0.5 * width_m[j] + expand, 0.0))

            if sat_obb_overlap_2d(
                ego_c_t, float(heading_rad[ego]), ego_hl_f, ego_hw_f,
                obj_c_t, float(heading_rad[j]), obj_hl_f, obj_hw_f,
            ):
                hit_future[ego, j] = 1
                hit_future[j, ego] = 1
                ttc[ego, j] = np.float32(t_star)
                ttc[j, ego] = np.float32(t_star)
                dtc_val = float(np.linalg.norm(v_rel) * t_star)
                dtc[ego, j] = np.float32(dtc_val)
                dtc[j, ego] = np.float32(dtc_val)

    if np.isfinite(ttc_max_s):
        too_big = ttc > float(ttc_max_s)
        if np.any(too_big):
            ttc[too_big] = np.inf
            dtc[too_big] = np.inf
            hit_future = (hit_future & (~too_big).astype(np.int8)).astype(np.int8)

    collision_pred = ((overlap_now == 1) | (hit_future == 1)).astype(np.int8)

    np.fill_diagonal(ttc, np.inf)
    np.fill_diagonal(dtc, np.inf)
    np.fill_diagonal(dist, 0.0)
    np.fill_diagonal(rel_speed, 0.0)
    np.fill_diagonal(overlap_now, 0)
    np.fill_diagonal(hit_future, 0)
    np.fill_diagonal(collision_pred, 0)
    np.fill_diagonal(approaching, 0)

    return (
        ttc.astype(np.float32),
        r.astype(np.float32),
        dv.astype(np.float32),
        dist.astype(np.float32),
        rel_speed.astype(np.float32),
        approaching.astype(np.int8),
        collision_pred.astype(np.int8),
        overlap_now.astype(np.int8),
        hit_future.astype(np.int8),
        dtc.astype(np.float32),
        s.astype(np.float32),
        l.astype(np.float32),
        d_perp.astype(np.float32),
    )


# -------------------------
# IO helpers
# -------------------------

def _read_partition(root: str, seg_id: str) -> pd.DataFrame:
    path = os.path.join(root, f"segment_id={seg_id}")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pq.read_table(path).to_pandas()


def _parse_int_set(csv_str: str) -> Set[int]:
    out: Set[int] = set()
    s = (csv_str or "").strip()
    if not s:
        return out
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.add(int(tok))
        except Exception:
            pass
    return out


def _filter_out_excluded_objects(
    g: pd.DataFrame,
    exclude_type_ids: Set[int],
    exclude_name_keywords: List[str],
) -> pd.DataFrame:
    if g.empty:
        return g

    g2 = g

    # 1) exclude by numeric type id
    if exclude_type_ids and "type" in g2.columns:
        t = pd.to_numeric(g2["type"], errors="coerce").fillna(-1).astype(int)
        g2 = g2[~t.isin(list(exclude_type_ids))]

    # 2) exclude by type name keywords if such column exists (robustness)
    name_cols = [c for c in ["type_name", "type_str", "label_type", "category"] if c in g2.columns]
    if exclude_name_keywords and name_cols:
        pat = "|".join([str(k).strip() for k in exclude_name_keywords if str(k).strip()])
        if pat:
            mask_any = False
            for c in name_cols:
                mask_any = mask_any | g2[c].astype(str).str.upper().str.contains(pat.upper(), na=False)
            g2 = g2[~mask_any]

    return g2


# -------------------------
# main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--only_valid_for_gssm", action="store_true")
    ap.add_argument("--eps", type=float, default=1e-6)

    ap.add_argument("--ego_length_m", type=float, default=4.8)
    ap.add_argument("--ego_width_m", type=float, default=2.0)
    ap.add_argument("--default_obj_length_m", type=float, default=4.5)
    ap.add_argument("--default_obj_width_m", type=float, default=1.8)

    ap.add_argument("--bbox_margin_m", type=float, default=0.3)
    ap.add_argument("--overlap_margin_m", type=float, default=0.15)
    ap.add_argument("--closing_speed_thr_mps", type=float, default=0.0)
    ap.add_argument("--ttc_max_s", type=float, default=float("inf"))
    ap.add_argument("--ego_future_hit_sat", action="store_true")
    ap.add_argument("--no_ego_future_hit_sat", action="store_true")

    ap.add_argument("--max_nodes", type=int, default=0)

    # ✅ NEW: exclude infra objects (e.g., stop_sign) from risk interaction candidates
    ap.add_argument(
        "--exclude_object_types",
        type=str,
        default="3",
        help="Comma-separated object type ids to EXCLUDE from TTC/overlap/hit_future computation. "
             "Default '3' (commonly TYPE_SIGN in Waymo Label.Type). "
             "If your type id differs, set accordingly, e.g. --exclude_object_types 4",
    )
    ap.add_argument(
        "--exclude_object_type_keywords",
        type=str,
        default="SIGN,STOP_SIGN",
        help="If base_objects has a string type column, exclude rows whose type name contains any of these keywords.",
    )

    args = ap.parse_args()

    use_ego_future_sat = bool(args.ego_future_hit_sat) and (not bool(args.no_ego_future_hit_sat))
    if not use_ego_future_sat:
        use_ego_future_sat = not bool(args.no_ego_future_hit_sat)

    exclude_type_ids = _parse_int_set(args.exclude_object_types)
    exclude_keywords = [k.strip() for k in (args.exclude_object_type_keywords or "").split(",") if k.strip()]

    staging = os.path.join(args.out_root, "staging")
    frm_root = os.path.join(staging, "base_frames")
    obj_root = os.path.join(staging, "base_objects")
    meta_path = os.path.join(staging, "segment_metadata.parquet")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing: {meta_path} (run 03_extract_metadata_v7.py first)")

    out_pairs = os.path.join(staging, "interaction_pairs_v7")
    out_cond = os.path.join(staging, "segment_condition_vectors_v7.parquet")
    out_schema = os.path.join(staging, "odd_input_schema_v7.json")

    if args.overwrite:
        if os.path.exists(out_pairs):
            import shutil
            shutil.rmtree(out_pairs)
        if os.path.exists(out_cond):
            os.remove(out_cond)
        if os.path.exists(out_schema):
            os.remove(out_schema)
    os.makedirs(out_pairs, exist_ok=True)

    meta_df = pd.read_parquet(meta_path)
    if meta_df.empty:
        raise RuntimeError("segment_metadata.parquet is empty")

    schema = build_odd_input_schema_v7(meta_df)
    with open(out_schema, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    if not os.path.exists(frm_root):
        raise RuntimeError(f"Missing frames root: {frm_root}")

    seg_ids = sorted([d.split("segment_id=")[-1] for d in os.listdir(frm_root) if d.startswith("segment_id=")])
    if not seg_ids:
        raise RuntimeError(f"No segments found under: {frm_root}")

    meta_df_idx = meta_df.set_index("segment_id", drop=False)

    cond_rows: List[Dict[str, Any]] = []
    for sid in seg_ids:
        if sid not in meta_df_idx.index:
            continue
        mrow = meta_df_idx.loc[sid]
        if isinstance(mrow, pd.DataFrame):
            mrow = mrow.iloc[0]
        if args.only_valid_for_gssm and int(mrow.get("is_valid_for_gssm", 0)) != 1:
            continue
        vec = encode_condition_vector(mrow, schema)
        cond_rows.append({"segment_id": sid, "cond_dim": int(schema["cond_dim"]), "cond_vec": vec.tolist()})

    cond_df = pd.DataFrame(cond_rows)
    pq.write_table(pa.Table.from_pandas(cond_df, preserve_index=False), out_cond)

    print(f"[04_v7_patched_v8] segments={len(seg_ids)} (only_valid_for_gssm={args.only_valid_for_gssm})")
    print(f" - ego_future_hit_sat: {use_ego_future_sat}")
    print(f" - bbox_margin_m={args.bbox_margin_m}, overlap_margin_m={args.overlap_margin_m}, closing_speed_thr_mps={args.closing_speed_thr_mps}, ttc_max_s={args.ttc_max_s}")
    print(f" - EXCLUDE object types: {sorted(list(exclude_type_ids))}  keywords={exclude_keywords}")

    for sid in seg_ids:
        if sid not in meta_df_idx.index:
            continue
        mrow = meta_df_idx.loc[sid]
        if isinstance(mrow, pd.DataFrame):
            mrow = mrow.iloc[0]
        if args.only_valid_for_gssm and int(mrow.get("is_valid_for_gssm", 0)) != 1:
            continue

        df_f = _read_partition(frm_root, sid)
        df_o = _read_partition(obj_root, sid)
        if df_f.empty or df_o.empty:
            continue

        req_f = ["frame_label", "timestamp_micros", "ego_x", "ego_y", "ego_vx", "ego_vy", "ego_yaw"]
        if any(c not in df_f.columns for c in req_f):
            continue
        req_o = ["frame_label", "obj_id", "type", "x", "y", "speed_x", "speed_y", "length", "width", "heading"]
        if any(c not in df_o.columns for c in req_o):
            continue

        df_f = df_f.sort_values("frame_label")
        ego_map = df_f.set_index("frame_label")[["timestamp_micros", "ego_x", "ego_y", "ego_vx", "ego_vy", "ego_yaw"]].to_dict("index")

        out_rows: List[pd.DataFrame] = []

        for fl, g in df_o.groupby("frame_label", sort=False):
            eg = ego_map.get(int(fl))
            if eg is None:
                continue

            ego_pos = np.array([eg["ego_x"], eg["ego_y"]], dtype=np.float32)
            ego_vel = np.array([eg["ego_vx"], eg["ego_vy"]], dtype=np.float32)
            ego_yaw = float(eg["ego_yaw"])

            # finite filter
            g = g[np.isfinite(g["x"]) & np.isfinite(g["y"]) & np.isfinite(g["speed_x"]) & np.isfinite(g["speed_y"])]

            # ✅ 핵심: stop_sign(TYPE_SIGN) 등 제외
            g = _filter_out_excluded_objects(g, exclude_type_ids, exclude_keywords)

            if g.empty:
                continue

            if args.max_nodes and args.max_nodes > 0:
                dx = (g["x"].values.astype(np.float32) - ego_pos[0])
                dy = (g["y"].values.astype(np.float32) - ego_pos[1])
                rr = np.sqrt(dx * dx + dy * dy)
                k = max(int(args.max_nodes) - 1, 0)
                if k > 0 and len(g) > k:
                    idx = np.argsort(rr)[:k]
                    g = g.iloc[idx]

            if g.empty:
                continue

            obj_len = pd.to_numeric(g["length"], errors="coerce").fillna(args.default_obj_length_m).clip(lower=0.5).values.astype(np.float32)
            obj_wid = pd.to_numeric(g["width"], errors="coerce").fillna(args.default_obj_width_m).clip(lower=0.5).values.astype(np.float32)

            pos = np.vstack([ego_pos[None, :], g[["x", "y"]].values.astype(np.float32)])
            vel = np.vstack([ego_vel[None, :], g[["speed_x", "speed_y"]].values.astype(np.float32)])
            length = np.concatenate([np.array([args.ego_length_m], dtype=np.float32), obj_len])
            width = np.concatenate([np.array([args.ego_width_m], dtype=np.float32), obj_wid])
            heading = np.concatenate([np.array([ego_yaw], dtype=np.float32), g["heading"].fillna(0.0).values.astype(np.float32)])

            ttc, rel_pos, rel_vel, rng, rel_spd, appr, coll, ov, hit, dtc, s_m, l_m, d_perp = compute_all_pairs_ttc(
                pos, vel, length, width, heading,
                eps_speed=float(args.eps),
                bbox_margin_m=float(args.bbox_margin_m),
                overlap_margin_m=float(args.overlap_margin_m),
                closing_speed_thr_mps=float(args.closing_speed_thr_mps),
                ttc_max_s=float(args.ttc_max_s),
                ego_future_hit_sat=use_ego_future_sat,
            )

            N = pos.shape[0]
            if N < 2:
                continue

            ii, jj = np.where(~np.eye(N, dtype=bool))

            src_ids = ["ego"] + g["obj_id"].astype(str).tolist()
            dst_ids = src_ids
            src_types = [1] + g["type"].fillna(0).astype(int).tolist()
            dst_types = src_types

            src_is_ego = (ii == 0).astype(np.int8)
            dst_is_ego = (jj == 0).astype(np.int8)

            src_speed = np.linalg.norm(vel[ii], axis=1).astype(np.float32)

            df_e = pd.DataFrame({
                "segment_id": sid,
                "frame_label": int(fl),
                "timestamp_micros": int(eg["timestamp_micros"]),
                "src_id": [src_ids[k] for k in ii],
                "dst_id": [dst_ids[k] for k in jj],
                "src_type": [src_types[k] for k in ii],
                "dst_type": [dst_types[k] for k in jj],
                "src_is_ego": src_is_ego,
                "dst_is_ego": dst_is_ego,
                "src_speed_mps": src_speed,
                "rel_pos_x": rel_pos[ii, jj, 0],
                "rel_pos_y": rel_pos[ii, jj, 1],
                "rel_vel_x": rel_vel[ii, jj, 0],
                "rel_vel_y": rel_vel[ii, jj, 1],
                "range_m": rng[ii, jj],
                "rel_speed_mps": rel_spd[ii, jj],
                "approaching": appr[ii, jj],
                "collision_pred": coll[ii, jj],
                "overlap_now": ov[ii, jj],
                "hit_future": hit[ii, jj],
                "ttc_2d": ttc[ii, jj],
                "dtc_m": dtc[ii, jj],
                "s_m": s_m[ii, jj],
                "l_m": l_m[ii, jj],
                "d_perp_m": d_perp[ii, jj],
            })
            out_rows.append(df_e)

        if out_rows:
            df_out = pd.concat(out_rows, ignore_index=True)
            pq.write_to_dataset(
                pa.Table.from_pandas(df_out, preserve_index=False),
                root_path=out_pairs,
                partition_cols=["segment_id"],
            )
            print(f" - wrote {sid}: frames={len(out_rows)} rows={len(df_out)}")

    print("[OK] 04_join_and_feature_engineer_v7_patched_v8 done.")


if __name__ == "__main__":
    main()

