#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_enhance_and_preprocess_v7_patched_v4_directionality.py

Builds GSSM-ready per-frame samples from interaction_pairs_v7, *plus* frame-wise ego/ODD/perception features
for explainable 2-stage training (Stage-1: risk occurrence; Stage-2: risk distribution).

Inputs (under <out_root>/staging):
  - interaction_pairs_v7/segment_id=.../*.parquet
  - segment_condition_vectors_v7.parquet          (from 04)
  - (optional but recommended) base_frames/segment_id=.../*.parquet
  - (optional but recommended) base_odd_features/segment_id=.../*.parquet

Outputs:
  <out_root>/<out_subdir>/<split>/shard_###.pt + index.json

Each sample item:
  {
    "segment_id": str,
    "frame_label": int,
    "x": List[float],     # [risk_ttci, density, ego..., odd..., pq...]
    "c": List[float],     # condition vector (segment-level)
    "y_soft": float,      # soft label (risk_ttci)
    "label": int,         # optional pseudo label
    "n_edges": int,
    "n_flagged": int,
  }

Notes / Design choices (pragmatic, stable):
  - Any NaN in frame-wise features is imputed to 0, except distances which are imputed to DIST_FAR_M.
  - has_traffic_light_visible(t): boolean presence only (no state), derived from camera labels.
  - signalized intersection: is_near_intersection(t) AND has_traffic_light_visible(t).
"""

from __future__ import annotations

import os
import json
import glob
import argparse
import hashlib
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

TTC_CLIP_MAX = 10.0
DIST_FAR_M = 1000.0  # for missing / far distances


def _hash01(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / float(0xFFFFFFFF)


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _segment_in_split(sid: str, split_name: str, train_ratio: float, val_ratio: float) -> bool:
    if split_name == "all":
        return True
    r = _hash01(sid)
    if r < train_ratio:
        return split_name == "train"
    if r < train_ratio + val_ratio:
        return split_name == "val"
    return split_name == "test"


def _iter_segment_ids(pairs_root: str) -> List[str]:
    seg_dirs = sorted(glob.glob(os.path.join(pairs_root, "segment_id=*")))
    return [os.path.basename(d).split("=", 1)[-1] for d in seg_dirs]


def _load_cond_map(cond_path: str) -> Dict[str, np.ndarray]:
    df = pd.read_parquet(cond_path)
    m: Dict[str, np.ndarray] = {}
    for _, r in df.iterrows():
        sid = str(r["segment_id"])
        vec = np.asarray(r["cond_vec"], dtype=np.float32)
        m[sid] = vec
    return m


def _ttci_from_ttc(ttc: float, eps: float, ttc_floor_s: float) -> float:
    if not np.isfinite(ttc):
        return 0.0
    if ttc <= 0:
        ttc = float(ttc_floor_s)
    v = 1.0 / (float(ttc) + float(eps))
    if v < 0:
        v = 0.0
    if v > TTC_CLIP_MAX:
        v = TTC_CLIP_MAX
    return float(v)


def _read_partition_df(root: str, sid: str, cols: Optional[List[str]] = None) -> pd.DataFrame:
    part_dir = os.path.join(root, f"segment_id={sid}")
    files = sorted(glob.glob(os.path.join(part_dir, "*.parquet")))
    if not files:
        return pd.DataFrame()

    dfs = []
    for fp in files:
        try:
            if cols is None:
                df = pd.read_parquet(fp)
            else:
                df = pd.read_parquet(fp, columns=[c for c in cols if c is not None])
        except Exception:
            # fallback: read full then subset
            df = pd.read_parquet(fp)
            if cols is not None:
                use = [c for c in cols if c in df.columns]
                df = df[use]
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return out


def _as_int_frame_label(df: pd.DataFrame) -> pd.Series:
    if "frame_label" not in df.columns:
        return pd.Series([], dtype="int64")
    fl = pd.to_numeric(df["frame_label"], errors="coerce")
    fl = fl.fillna(-1).astype(int)
    return fl


def _load_frame_features(base_frames_root: str, sid: str) -> pd.DataFrame:
    cols = [
        "frame_label",
        "ego_speed_mps",
        "ego_accel_mps2",
        "ego_yaw_rate_rps",
        "ego_jerk_mps3",
        "ego_yaw",
        "ego_speed_lag1",
        "ego_accel_lag1",
        "ego_speed_lag2",
        "ego_accel_lag2",
    ]
    df = _read_partition_df(base_frames_root, sid, cols=cols)
    if df.empty:
        return df
    df["frame_label"] = _as_int_frame_label(df)
    df = df[df["frame_label"] >= 0].copy()
    # de-dup: keep last occurrence per frame_label (safe for parquet sharding)
    df = df.sort_values("frame_label").drop_duplicates("frame_label", keep="last")
    df = df.set_index("frame_label", drop=True)
    return df


def _load_odd_features(base_odd_root: str, sid: str) -> pd.DataFrame:
    cols = [
        "frame_label",
        "is_near_intersection",
        "dist_to_intersection_lane_m",
        "is_near_stop_sign",
        "is_near_crosswalk",
        "is_near_speed_bump",
        "is_near_driveway",
        "has_traffic_light_visible",
        "is_near_signalized_intersection",
        "dist_to_signalized_intersection_lane_m",
        "dyn_label_count_30m",
        "mean_lidar_points_in_box_dyn_30m",
        "occlusion_low_points_ratio_dyn_30m",
    ]
    df = _read_partition_df(base_odd_root, sid, cols=cols)
    if df.empty:
        return df
    df["frame_label"] = _as_int_frame_label(df)
    df = df[df["frame_label"] >= 0].copy()
    df = df.sort_values("frame_label").drop_duplicates("frame_label", keep="last")
    df = df.set_index("frame_label", drop=True)
    return df


def _pick_collision_flag(df: pd.DataFrame) -> Optional[str]:
    if "collision_course" in df.columns:
        return "collision_course"
    if "collision_pred" in df.columns:
        return "collision_pred"
    return None


def _get_rel_speed(df: pd.DataFrame) -> Tuple[np.ndarray, bool]:
    if "rel_speed_mps" in df.columns:
        rs = pd.to_numeric(df["rel_speed_mps"], errors="coerce").values
        return rs, True
    if "rel_vel_x" in df.columns and "rel_vel_y" in df.columns:
        rvx = pd.to_numeric(df["rel_vel_x"], errors="coerce").fillna(0.0).values
        rvy = pd.to_numeric(df["rel_vel_y"], errors="coerce").fillna(0.0).values
        rs = np.sqrt(rvx * rvx + rvy * rvy)
        return rs, True
    return np.full((len(df),), np.nan, dtype=np.float32), False


class FrameAgg:
    """Per-frame aggregates computed from interaction_pairs.

    Notes:
      - min_ttc is computed from ttc_2d among candidate (approaching/collision) pairs.
      - Directional aggregates are computed from rel_pos/rel_vel in ego frame, *without using TTC values*.
        This avoids re-injecting TTC-derived leakage into the conditioning features.
    """

    __slots__ = (
        "min_ttc",
        "flag_cnt",
        "total_cnt",
        # directional interaction field (ego->others, approaching-only)
        "appr_cnt_any",
        "appr_cnt_front",
        "appr_cnt_left",
        "appr_cnt_right",
        "appr_cnt_rear",
        "min_range_any",
        "min_range_front",
        "min_range_left",
        "min_range_right",
        "min_range_rear",
        "max_close_any",
        "max_close_front",
        "max_close_left",
        "max_close_right",
        "max_close_rear",
    )

    def __init__(self):
        self.min_ttc = np.inf
        self.flag_cnt = 0
        self.total_cnt = 0

        self.appr_cnt_any = 0
        self.appr_cnt_front = 0
        self.appr_cnt_left = 0
        self.appr_cnt_right = 0
        self.appr_cnt_rear = 0

        self.min_range_any = np.inf
        self.min_range_front = np.inf
        self.min_range_left = np.inf
        self.min_range_right = np.inf
        self.min_range_rear = np.inf

        self.max_close_any = 0.0
        self.max_close_front = 0.0
        self.max_close_left = 0.0
        self.max_close_right = 0.0
        self.max_close_rear = 0.0


class ShardWriter:

    def __init__(self, out_dir: str, split_name: str, shard_size: int):
        self.out_dir = out_dir
        self.split_name = split_name
        self.shard_size = shard_size
        self.buffer: List[Dict[str, Any]] = []
        self.shard_idx = 0
        self.total_written = 0
        self.shards: List[Dict[str, Any]] = []
        _ensure_dir(out_dir)

    def add(self, item: Dict[str, Any]):
        self.buffer.append(item)
        if self.shard_size > 0 and len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        shard_path = os.path.join(self.out_dir, f"shard_{self.shard_idx:03d}.pt")
        torch.save(self.buffer, shard_path)
        n = len(self.buffer)
        self.shards.append({"path": shard_path, "n": n})
        self.total_written += n
        self.buffer = []
        self.shard_idx += 1

    def close(self, *, x_schema: List[str], x_groups: Dict[str, List[int]]) -> str:
        self.flush()
        index_path = os.path.join(self.out_dir, "index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "split": self.split_name,
                    "format": "sharded_list_v2",
                    "total_samples": self.total_written,
                    "num_shards": len(self.shards),
                    "shards": self.shards,
                    "x_schema": x_schema,
                    "x_groups": x_groups,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return index_path


def _update_aggs_from_part(
    df: pd.DataFrame,
    aggs: Dict[int, FrameAgg],
    *,
    eps: float,
    distance_max: float,
    ego_only: bool,
    use_all_pairs_density: bool,
    density_mode: str,
    min_ego_speed_mps: float,
    min_rel_speed_mps: float,
    exclude_static_close: bool,
    static_close_range_m: float,
    static_close_rel_speed_mps: float,
    ego_speed_map: Optional[Dict[int, float]],
    ego_yaw_map: Optional[Dict[int, float]],
) -> None:
    if "range_m" not in df.columns and "dist_m" in df.columns:
        df = df.rename(columns={"dist_m": "range_m"})

    for c in ["frame_label", "range_m", "ttc_2d"]:
        if c not in df.columns:
            raise RuntimeError(f"pairs missing required column: {c}")

    if ego_only and "src_is_ego" not in df.columns:
        raise RuntimeError("pairs missing src_is_ego (use --ego_only requires it)")

    df = df.copy()
    df["frame_label"] = pd.to_numeric(df["frame_label"], errors="coerce").astype("Int64")
    df["range_m"] = pd.to_numeric(df["range_m"], errors="coerce")
    df["ttc_2d"] = pd.to_numeric(df["ttc_2d"], errors="coerce")

    df = df[df["frame_label"].notna()]
    if df.empty:
        return

    df = df[np.isfinite(df["range_m"].values) & (df["range_m"].values <= float(distance_max))]
    if df.empty:
        return

    coll_flag = _pick_collision_flag(df)
    has_approach = "approaching" in df.columns
    if has_approach:
        df["approaching"] = pd.to_numeric(df["approaching"], errors="coerce").fillna(0).astype(int)
    if coll_flag is not None:
        df[coll_flag] = pd.to_numeric(df[coll_flag], errors="coerce").fillna(0).astype(int)

    # density
    if use_all_pairs_density:
        df_den = df
    else:
        df_den = df[df["src_is_ego"] == 1] if ego_only else df

    den_flag = None
    if density_mode == "collision" and coll_flag is not None:
        den_flag = coll_flag
    elif has_approach:
        den_flag = "approaching"
    elif coll_flag is not None:
        den_flag = coll_flag

    if den_flag is not None and not df_den.empty:
        for fl, g in df_den.groupby("frame_label", sort=False):
            fl_i = int(fl)
            a = aggs.get(fl_i)
            if a is None:
                a = FrameAgg()
                aggs[fl_i] = a
            a.total_cnt += int(len(g))
            a.flag_cnt += int((g[den_flag].values == 1).sum())


    # --- Directional interaction field (ego->others, approaching-only) ---
    # Goal: preserve *interaction directionality* in the conditioning features without feeding TTC-derived
    # values back into the model. We use rel_pos/rel_vel (already present in interaction_pairs) to derive:
    #   - sector counts (front/left/right/rear)
    #   - min range per sector
    #   - max line-of-sight closing speed per sector
    # These are computed in the ego frame using ego_yaw at the same frame.
    if ego_yaw_map is not None and len(ego_yaw_map) > 0:
        need_cols = {"rel_pos_x", "rel_pos_y", "rel_vel_x", "rel_vel_y", "range_m"}
        if need_cols.issubset(set(df.columns)) and ("src_is_ego" in df.columns):
            df_dir = df[df["src_is_ego"] == 1]
            if not df_dir.empty:
                fl_s = df_dir["frame_label"].astype(int)
                yaw = fl_s.map(ego_yaw_map).astype(float).to_numpy()
                yaw = np.where(np.isfinite(yaw), yaw, 0.0)
                cy = np.cos(yaw)
                sy = np.sin(yaw)

                rx = pd.to_numeric(df_dir["rel_pos_x"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                ry = pd.to_numeric(df_dir["rel_pos_y"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

                # rotate rel_pos into ego frame
                x_ego = cy * rx + sy * ry
                y_ego = -sy * rx + cy * ry

                abs_x = np.abs(x_ego)
                abs_y = np.abs(y_ego)
                lr = (abs_y > abs_x)
                sector = np.empty(len(df_dir), dtype=np.int8)
                # 0=front, 3=rear, 1=left, 2=right
                sector[~lr] = np.where(x_ego[~lr] >= 0.0, 0, 3)
                sector[lr] = np.where(y_ego[lr] > 0.0, 1, 2)

                rng = pd.to_numeric(df_dir["range_m"], errors="coerce").to_numpy(dtype=np.float32)
                rvx = pd.to_numeric(df_dir["rel_vel_x"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                rvy = pd.to_numeric(df_dir["rel_vel_y"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

                # radial closing speed along line-of-sight (positive => approaching)
                dot = rx * rvx + ry * rvy
                close = np.maximum(0.0, -dot / (rng + 1e-6))

                if "approaching" in df_dir.columns:
                    appr = pd.to_numeric(df_dir["approaching"], errors="coerce").fillna(0).astype(int).to_numpy()
                    appr_mask = (appr == 1)
                else:
                    appr_mask = (close > 0.0)

                if appr_mask.any():
                    tmp = pd.DataFrame({
                        "frame_label": fl_s.to_numpy()[appr_mask],
                        "sector": sector[appr_mask],
                        "range_m": rng[appr_mask],
                        "close_mps": close[appr_mask],
                    })
                    gg = tmp.groupby(["frame_label", "sector"], sort=False).agg(
                        cnt=("sector", "size"),
                        min_rng=("range_m", "min"),
                        max_close=("close_mps", "max"),
                    )
                    for (fl, sec), r in gg.iterrows():
                        fl_i = int(fl)
                        sec_i = int(sec)
                        a = aggs.get(fl_i)
                        if a is None:
                            a = FrameAgg()
                            aggs[fl_i] = a

                        cnt = int(r["cnt"])
                        min_rng = float(r["min_rng"]) if np.isfinite(r["min_rng"]) else np.inf
                        max_close = float(r["max_close"]) if np.isfinite(r["max_close"]) else 0.0

                        a.appr_cnt_any += cnt
                        a.min_range_any = min(a.min_range_any, min_rng)
                        a.max_close_any = max(a.max_close_any, max_close)

                        if sec_i == 0:
                            a.appr_cnt_front += cnt
                            a.min_range_front = min(a.min_range_front, min_rng)
                            a.max_close_front = max(a.max_close_front, max_close)
                        elif sec_i == 1:
                            a.appr_cnt_left += cnt
                            a.min_range_left = min(a.min_range_left, min_rng)
                            a.max_close_left = max(a.max_close_left, max_close)
                        elif sec_i == 2:
                            a.appr_cnt_right += cnt
                            a.min_range_right = min(a.min_range_right, min_rng)
                            a.max_close_right = max(a.max_close_right, max_close)
                        else:
                            a.appr_cnt_rear += cnt
                            a.min_range_rear = min(a.min_range_rear, min_rng)
                            a.max_close_rear = max(a.max_close_rear, max_close)
    # risk candidates
    df_risk = df[df["src_is_ego"] == 1] if ego_only else df

    if coll_flag is not None:
        df_cand = df_risk[df_risk[coll_flag] == 1]
    elif has_approach:
        df_cand = df_risk[df_risk["approaching"] == 1]
    else:
        df_cand = df_risk

    if df_cand.empty:
        return

    # ego speed filter
    if min_ego_speed_mps > 0:
        if "src_speed_mps" in df_cand.columns:
            sspd = pd.to_numeric(df_cand["src_speed_mps"], errors="coerce")
            df_cand = df_cand[sspd.isna() | (sspd.values >= float(min_ego_speed_mps))]
        elif ego_speed_map:
            spd_series = df_cand["frame_label"].astype(int).map(ego_speed_map)
            df_cand = df_cand[spd_series.isna() | (spd_series.values >= float(min_ego_speed_mps))]
        if df_cand.empty:
            return

    # rel speed filter + static-close exclusion
    rel_speed, ok_rel = _get_rel_speed(df_cand)
    if (min_rel_speed_mps > 0 or exclude_static_close) and ok_rel:
        if min_rel_speed_mps > 0:
            mask = np.isfinite(rel_speed) & (rel_speed >= float(min_rel_speed_mps))
            df_cand = df_cand[mask]
            rel_speed = rel_speed[mask]
            if df_cand.empty:
                return

        if exclude_static_close:
            rng = df_cand["range_m"].values
            mask_sc = (rng < float(static_close_range_m)) & (rel_speed < float(static_close_rel_speed_mps))
            df_cand = df_cand[~mask_sc]
            if df_cand.empty:
                return

    # finite ttc only
    df_cand = df_cand[np.isfinite(df_cand["ttc_2d"].values)]
    if df_cand.empty:
        return

    for fl, min_ttc_part in df_cand.groupby("frame_label", sort=False)["ttc_2d"].min().items():
        fl_i = int(fl)
        a = aggs.get(fl_i)
        if a is None:
            a = FrameAgg()
            aggs[fl_i] = a
        v = float(min_ttc_part) if np.isfinite(min_ttc_part) else np.inf
        if np.isfinite(v) and v < a.min_ttc:
            a.min_ttc = v


def _build_x(
    frame_label: int,
    *,
    best_ttci: float,
    density: float,
    frm_df: pd.DataFrame,
    odd_df: pd.DataFrame,
) -> List[float]:
    # defaults
    def _get(df: pd.DataFrame, key: str, default: float) -> float:
        if df is None or df.empty:
            return float(default)
        if key not in df.columns:
            return float(default)
        v = df.at[frame_label, key] if frame_label in df.index else default
        try:
            v = float(v)
        except Exception:
            v = default
        if not np.isfinite(v):
            return float(default)
        return float(v)

    # ego
    ego_speed = _get(frm_df, "ego_speed_mps", 0.0)
    ego_accel = _get(frm_df, "ego_accel_mps2", 0.0)
    ego_yaw_rate = _get(frm_df, "ego_yaw_rate_rps", 0.0)
    ego_jerk = _get(frm_df, "ego_jerk_mps3", 0.0)
    ego_speed_lag1 = _get(frm_df, "ego_speed_lag1", 0.0)
    ego_accel_lag1 = _get(frm_df, "ego_accel_lag1", 0.0)
    ego_speed_lag2 = _get(frm_df, "ego_speed_lag2", 0.0)
    ego_accel_lag2 = _get(frm_df, "ego_accel_lag2", 0.0)

    # near/static ODD
    is_near_intersection = 1.0 if _get(odd_df, "is_near_intersection", 0.0) >= 1.0 else 0.0
    dist_to_intersection = _get(odd_df, "dist_to_intersection_lane_m", DIST_FAR_M)
    is_near_stop_sign = 1.0 if _get(odd_df, "is_near_stop_sign", 0.0) >= 1.0 else 0.0
    is_near_crosswalk = 1.0 if _get(odd_df, "is_near_crosswalk", 0.0) >= 1.0 else 0.0
    is_near_speed_bump = 1.0 if _get(odd_df, "is_near_speed_bump", 0.0) >= 1.0 else 0.0
    is_near_driveway = 1.0 if _get(odd_df, "is_near_driveway", 0.0) >= 1.0 else 0.0

    # traffic light presence only (no state)
    has_tl_vis = 1.0 if _get(odd_df, "has_traffic_light_visible", 0.0) > 0.0 else 0.0

    # signalized intersection (preferred: precomputed; fallback: near_intersection & tl_visible)
    sig_flag = _get(odd_df, "is_near_signalized_intersection", np.nan)
    if np.isfinite(sig_flag):
        is_near_signalized = 1.0 if sig_flag >= 1.0 else 0.0
    else:
        is_near_signalized = 1.0 if (is_near_intersection >= 1.0 and has_tl_vis >= 1.0) else 0.0

    dist_to_signalized = _get(odd_df, "dist_to_signalized_intersection_lane_m", np.nan)
    if not np.isfinite(dist_to_signalized):
        # if we are signalized and have intersection dist, reuse it; else far
        dist_to_signalized = dist_to_intersection if is_near_signalized >= 1.0 else DIST_FAR_M

    # dynamic perception quality (near-field, dynamic labels)
    dyn_cnt_30m = _get(odd_df, "dyn_label_count_30m", 0.0)
    dyn_lidar_30m = _get(odd_df, "mean_lidar_points_in_box_dyn_30m", 0.0)
    dyn_occ_30m = _get(odd_df, "occlusion_low_points_ratio_dyn_30m", 0.0)

    x = [
        float(best_ttci),
        float(density),

        # ego kinematics (vehicle-frame, frame-wise)
        ego_speed, ego_accel, ego_yaw_rate, ego_jerk,
        ego_speed_lag1, ego_accel_lag1,
        ego_speed_lag2, ego_accel_lag2,

        # ODD proximity flags + distances
        is_near_intersection, dist_to_intersection,
        is_near_stop_sign, is_near_crosswalk, is_near_speed_bump, is_near_driveway,
        has_tl_vis, is_near_signalized, dist_to_signalized,

        # perception quality (dynamic, near-field)
        dyn_cnt_30m, dyn_lidar_30m, dyn_occ_30m,
    ]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)

    ap.add_argument("--split_name", type=str, default="train", choices=["train", "val", "test", "all"])
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    ap.add_argument("--distance_max", type=float, default=70.0)
    ap.add_argument("--eps", type=float, default=1e-6)

    ap.add_argument("--ego_only", action="store_true")
    ap.add_argument("--use_all_pairs_density", action="store_true")
    ap.add_argument("--density_mode", type=str, default="approaching", choices=["approaching", "collision"])

    ap.add_argument("--pseudo_label", action="store_true")
    ap.add_argument("--label_thr_ttci", type=float, default=2.0)

    ap.add_argument("--strict_cond", action="store_true")
    ap.add_argument("--allow_fallback_cond", action="store_true")

    ap.add_argument("--shard_size", type=int, default=200000)
    ap.add_argument("--out_subdir", type=str, default="processed_v7")

    ap.add_argument("--min_ego_speed_mps", type=float, default=0.0)
    ap.add_argument("--min_rel_speed_mps", type=float, default=0.0)
    ap.add_argument("--exclude_static_close", action="store_true")
    ap.add_argument("--static_close_range_m", type=float, default=3.0)
    ap.add_argument("--static_close_rel_speed_mps", type=float, default=0.5)
    ap.add_argument("--ttc_floor_s", type=float, default=0.05)

    args = ap.parse_args()

    out_root = os.path.abspath(args.out_root)
    staging = os.path.join(out_root, "staging")

    pairs_root = os.path.join(staging, "interaction_pairs_v7")
    cond_path = os.path.join(staging, "segment_condition_vectors_v7.parquet")
    base_frames_root = os.path.join(staging, "base_frames")
    base_odd_root = os.path.join(staging, "base_odd_features")

    if not os.path.exists(pairs_root):
        raise SystemExit(f"[ERROR] not found: {pairs_root}")
    if not os.path.exists(cond_path):
        raise SystemExit(f"[ERROR] not found: {cond_path} (run 04 first)")

    cond_map = _load_cond_map(cond_path)

    out_dir = os.path.join(out_root, args.out_subdir, args.split_name)
    writer = ShardWriter(out_dir, args.split_name, args.shard_size)

    seg_ids = _iter_segment_ids(pairs_root)
    if not seg_ids:
        raise SystemExit("[ERROR] no segment partitions found under interaction_pairs_v7")

    # x schema (fixed order)
    x_schema = [
        "best_ttci",
        "density",
        "ego_speed_mps",
        "ego_accel_mps2",
        "ego_yaw_rate_rps",
        "ego_jerk_mps3",
        "ego_speed_lag1",
        "ego_accel_lag1",
        "ego_speed_lag2",
        "ego_accel_lag2",
        "is_near_intersection",
        "dist_to_intersection_lane_m",
        "is_near_stop_sign",
        "is_near_crosswalk",
        "is_near_speed_bump",
        "is_near_driveway",
        "has_traffic_light_visible",
        "is_near_signalized_intersection",
        "dist_to_signalized_intersection_lane_m",
        "dyn_label_count_30m",
        "mean_lidar_points_in_box_dyn_30m",
        "occlusion_low_points_ratio_dyn_30m",
        "appr_cnt_front",
        "appr_cnt_left",
        "appr_cnt_right",
        "appr_cnt_rear",
        "min_range_front_m",
        "min_range_left_m",
        "min_range_right_m",
        "min_range_rear_m",
        "max_closing_speed_front_mps",
        "max_closing_speed_left_mps",
        "max_closing_speed_right_mps",
        "max_closing_speed_rear_mps",
        "appr_cnt_any",
        "min_range_any_m",
        "max_closing_speed_any_mps",
    ]
    x_groups = {
        "risk": [0, 1],
        "ego": [2, 3, 4, 5, 6, 7, 8, 9],
        "odd": [10, 11, 12, 13, 14, 15, 16, 17, 18],
        "perception_quality": [19, 20, 21],
        "interaction_dir": [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
    }

    kept_segments = 0
    skipped_no_cond = 0
    total_samples = 0

    for sid in seg_ids:
        if not _segment_in_split(sid, args.split_name, args.train_ratio, args.val_ratio):
            continue

        cvec = cond_map.get(sid, None)
        if cvec is None:
            if args.strict_cond and not args.allow_fallback_cond:
                skipped_no_cond += 1
                continue
            if args.allow_fallback_cond:
                cvec = np.zeros((1,), dtype=np.float32)
            else:
                skipped_no_cond += 1
                continue

        # load per-frame aux features (recommended)
        frm_df = pd.DataFrame()
        if os.path.exists(base_frames_root):
            frm_df = _load_frame_features(base_frames_root, sid)

        odd_df = pd.DataFrame()
        if os.path.exists(base_odd_root):
            odd_df = _load_odd_features(base_odd_root, sid)

        ego_speed_map: Optional[Dict[int, float]] = None
        if args.min_ego_speed_mps > 0:
            if not frm_df.empty and "ego_speed_mps" in frm_df.columns:
                ego_speed_map = {int(k): float(v) for k, v in frm_df["ego_speed_mps"].items() if np.isfinite(v)}
            else:
                ego_speed_map = None


        ego_yaw_map: Optional[Dict[int, float]] = None
        if not frm_df.empty and "ego_yaw" in frm_df.columns:
            ego_yaw_map = {int(k): float(v) for k, v in frm_df["ego_yaw"].items() if np.isfinite(v)}
        aggs: Dict[int, FrameAgg] = {}

        part_files = sorted(glob.glob(os.path.join(pairs_root, f"segment_id={sid}", "*.parquet")))
        if not part_files:
            continue

        for fp in part_files:
            df = pd.read_parquet(fp)
            if "segment_id" not in df.columns:
                df["segment_id"] = sid

            _update_aggs_from_part(
                df,
                aggs,
                eps=args.eps,
                distance_max=args.distance_max,
                ego_only=args.ego_only,
                use_all_pairs_density=args.use_all_pairs_density,
                density_mode=args.density_mode,
                min_ego_speed_mps=args.min_ego_speed_mps,
                min_rel_speed_mps=args.min_rel_speed_mps,
                exclude_static_close=args.exclude_static_close,
                static_close_range_m=args.static_close_range_m,
                static_close_rel_speed_mps=args.static_close_rel_speed_mps,
                ego_speed_map=ego_speed_map,
                ego_yaw_map=ego_yaw_map,
            )

        if not aggs:
            continue

        for fl in sorted(aggs.keys()):
            a = aggs[fl]
            density = float(a.flag_cnt / max(1, a.total_cnt))
            best_ttci = _ttci_from_ttc(a.min_ttc, args.eps, args.ttc_floor_s)

            x = _build_x(
                int(fl),
                best_ttci=best_ttci,
                density=density,
                frm_df=frm_df,
                odd_df=odd_df,
            )

            # directionality summary (approaching ego->others), defaults: far/0 when absent
            def _far(v: float) -> float:
                if v is None or (not np.isfinite(v)) or (v == np.inf):
                    return float(DIST_FAR_M)
                return float(v)

            x.extend([
                float(a.appr_cnt_front), float(a.appr_cnt_left), float(a.appr_cnt_right), float(a.appr_cnt_rear),
                _far(a.min_range_front), _far(a.min_range_left), _far(a.min_range_right), _far(a.min_range_rear),
                float(a.max_close_front), float(a.max_close_left), float(a.max_close_right), float(a.max_close_rear),
                float(a.appr_cnt_any), _far(a.min_range_any), float(a.max_close_any),
            ])

            item: Dict[str, Any] = {
                "segment_id": sid,
                "frame_label": int(fl),
                "x": [float(v) for v in x],
                "c": cvec.astype(np.float32).tolist(),
                "y_soft": float(best_ttci),
                "n_edges": int(a.total_cnt),
                "n_flagged": int(a.flag_cnt),
            }
            item["label"] = int(best_ttci > float(args.label_thr_ttci)) if args.pseudo_label else -1

            writer.add(item)
            total_samples += 1

        kept_segments += 1

    index_path = writer.close(x_schema=x_schema, x_groups=x_groups)
    print(f"[OK] wrote shards under: {os.path.dirname(index_path)}")
    print(f" - index: {index_path}")
    print(f" - samples: {total_samples:,}  segments: {kept_segments}  skipped_no_cond: {skipped_no_cond}")


if __name__ == "__main__":
    main()
