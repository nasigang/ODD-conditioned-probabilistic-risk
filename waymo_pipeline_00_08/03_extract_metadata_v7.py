#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_extract_metadata_v7.py
- Segment-level metadata summarizer + ODD topology classifier
- Designed for GSSM-variant research: conditional sampling P(risk feature | ODD)
- Input: staging/base_frames, staging/base_odd_features, (optional) staging/segment_static_map
- Output: staging/segment_metadata.parquet   (keeps 04_join_and_feature_engineer_new.py compatible)
"""

from __future__ import annotations

import os
import argparse
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# -------------------------
# Robust helpers
# -------------------------
def _col(df: pd.DataFrame, *cands: str) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None


def _mode_str(s: pd.Series, default: str = "unknown") -> str:
    if s is None or s.empty:
        return default
    try:
        m = s.dropna().mode()
        return str(m.iloc[0]) if len(m) else default
    except Exception:
        return default


def _wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a[np.isfinite(a)]


def _pct(a: np.ndarray, q: float) -> float:
    a = _finite(a)
    return float(np.percentile(a, q)) if a.size else np.nan


def _stats(a: np.ndarray, prefix: str) -> Dict[str, Any]:
    a = _finite(a)
    if a.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_p95": np.nan,
        }
    return {
        f"{prefix}_n": int(a.size),
        f"{prefix}_mean": float(a.mean()),
        f"{prefix}_std": float(a.std(ddof=0)),
        f"{prefix}_max": float(a.max()),
        f"{prefix}_p95": _pct(a, 95),
    }


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _bin_by_threshold(x: float, thr_lo: float, thr_hi: float, labels=("low", "mid", "high")) -> str:
    if not np.isfinite(x):
        return "unknown"
    if x <= thr_lo:
        return labels[0]
    if x <= thr_hi:
        return labels[1]
    return labels[2]


# -------------------------
# Topology classification
# -------------------------
def classify_topology_frame(odd_df: pd.DataFrame) -> pd.Series:
    """
    Frame-wise topology label.
    Uses the strongest available signals in base_odd_features:
      - intersection_complexity (preferred)
      - lane_interpolating
      - lane_entry_count / lane_exit_count
    Returns a Series aligned with odd_df rows.
    """
    if odd_df.empty:
        return pd.Series(dtype="object")

    c_near_inter = _col(odd_df, "is_near_intersection")
    c_inter = _col(odd_df, "intersection_complexity")
    c_interp = _col(odd_df, "lane_interpolating", "interpolating")
    c_entry = _col(odd_df, "lane_entry_count", "entry_count")
    c_exit = _col(odd_df, "lane_exit_count", "exit_count")

    near_inter = odd_df[c_near_inter].fillna(0).astype(float) if c_near_inter else pd.Series(0.0, index=odd_df.index)
    inter = odd_df[c_inter].fillna(0).astype(float) if c_inter else pd.Series(0.0, index=odd_df.index)
    interp = odd_df[c_interp].fillna(0).astype(float) if c_interp else pd.Series(0.0, index=odd_df.index)
    entry = odd_df[c_entry].fillna(0).astype(float) if c_entry else pd.Series(0.0, index=odd_df.index)
    exit_ = odd_df[c_exit].fillna(0).astype(float) if c_exit else pd.Series(0.0, index=odd_df.index)

    # Heuristic thresholds (tunable). Keep deterministic defaults.
    # Prefer distance-based 'near intersection' flag when available (more faithful than complexity-only heuristics).
    is_intersection = (near_inter >= 1.0) | (interp >= 1.0) | (inter >= 3.0) | ((entry >= 2.0) & (exit_ >= 2.0))
    is_merge = (entry - exit_) >= 1.0
    is_split = (exit_ - entry) >= 1.0

    lab = np.full(len(odd_df), "straight", dtype=object)
    lab[is_split.values] = "split"
    lab[is_merge.values] = "merge"
    lab[is_intersection.values] = "intersection"  # strongest wins

    return pd.Series(lab, index=odd_df.index)


def summarize_segment_v7(seg_id: str, odd_df: pd.DataFrame, frm_df: pd.DataFrame, smap_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "segment_id": seg_id,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "version": "v7",
    }

    # -------------------------
    # Segment duration / frames
    # -------------------------
    if not frm_df.empty and "timestamp_micros" in frm_df.columns:
        ts = frm_df["timestamp_micros"].astype(np.int64).values
        row["n_frames"] = int(len(ts))
        row["duration_s"] = float((ts.max() - ts.min()) * 1e-6) if len(ts) else np.nan
    else:
        row["n_frames"] = 0
        row["duration_s"] = np.nan

    # -------------------------
    # ODD context summary (time/weather/location)
    # -------------------------
    if not odd_df.empty:
        row["odd_time"] = _mode_str(odd_df.get("time_of_day", pd.Series(dtype="object")))
        row["odd_weather"] = _mode_str(odd_df.get("weather", pd.Series(dtype="object")))
        row["odd_location"] = _mode_str(odd_df.get("location", pd.Series(dtype="object")))
    else:
        row["odd_time"] = "unknown"
        row["odd_weather"] = "unknown"
        row["odd_location"] = "unknown"

    # -------------------------
    # ODD instantiation: infrastructure & lane geometry/topology
    # -------------------------
    if not odd_df.empty:
        # infra density
        c_infra_d = _col(odd_df, "infra_density_r")
        if c_infra_d:
            row.update(_stats(odd_df[c_infra_d].values, "infra_density"))
        else:
            row.update(_stats(np.array([], dtype=np.float64), "infra_density"))

        # infra presence (keep 04_new compatibility cols)
        for name, cand in [
            ("infra_stop_sign", ("infra_stop_sign_count", "stop_sign_count_r")),
            ("infra_crosswalk", ("infra_crosswalk_count", "crosswalk_count_r")),
            ("infra_speed_bump", ("infra_speed_bump_count", "speed_bump_count_r")),
            ("infra_driveway", ("infra_driveway_count", "driveway_count_r")),
        ]:
            c = _col(odd_df, *cand)
            mx = float(odd_df[c].max()) if c else np.nan
            row[f"{name}_max"] = mx
            row[f"has_{name.replace('infra_', '')}"] = int(np.isfinite(mx) and mx > 0)

        # traffic-light visibility (from camera_labels; state not included)
        c_tl = _col(odd_df, "has_traffic_light_visible", "traffic_light_visible_count")
        if c_tl:
            mx_tl = float(odd_df[c_tl].max())
            row["has_traffic_light"] = int(np.isfinite(mx_tl) and mx_tl > 0)
            row["traffic_light_visible_ratio"] = float(np.mean(odd_df[c_tl].fillna(0).astype(float).values > 0))
        else:
            row["has_traffic_light"] = 0
            row["traffic_light_visible_ratio"] = np.nan


        # signalized intersection activity (frame-wise): near_intersection AND traffic_light_visible
        # (preferred) use precomputed is_near_signalized_intersection from 02.
        c_sig = _col(odd_df, "is_near_signalized_intersection")
        if c_sig:
            sgi = odd_df[c_sig].fillna(0).astype(float).values
            row["signalized_intersection_ratio"] = float(np.mean(sgi >= 1.0))
        else:
            c_near2 = _col(odd_df, "is_near_intersection")
            if c_near2 and c_tl:
                sgi = (odd_df[c_near2].fillna(0).astype(float).values >= 1.0) & (odd_df[c_tl].fillna(0).astype(float).values >= 1.0)
                row["signalized_intersection_ratio"] = float(np.mean(sgi))
            else:
                row["signalized_intersection_ratio"] = np.nan

        # distances to (signalized) intersection lane (optional but highly informative for braking risk)
        c_dint = _col(odd_df, "dist_to_intersection_lane_m")
        if c_dint:
            row.update(_stats(pd.to_numeric(odd_df[c_dint], errors="coerce").values, "dist_to_intersection_lane_m"))
        else:
            row.update(_stats(np.array([], dtype=np.float64), "dist_to_intersection_lane_m"))

        c_dsig = _col(odd_df, "dist_to_signalized_intersection_lane_m")
        if c_dsig:
            row.update(_stats(pd.to_numeric(odd_df[c_dsig], errors="coerce").values, "dist_to_signalized_intersection_lane_m"))
        else:
            row.update(_stats(np.array([], dtype=np.float64), "dist_to_signalized_intersection_lane_m"))


        # lane type / speed limit summaries
        c_lane_type = _col(odd_df, "lane_type")
        row["lane_type_mode"] = _mode_str(odd_df[c_lane_type].astype("Int64")) if c_lane_type else "unknown"

        c_speed = _col(odd_df, "speed_limit_mph")
        row.update(_stats(odd_df[c_speed].values, "speed_limit_mph") if c_speed else _stats(np.array([], dtype=np.float64), "speed_limit_mph"))


        # intersection activity (ratio + peak)
        c_near = _col(odd_df, "is_near_intersection")
        c_inter = _col(odd_df, "intersection_complexity")
        if c_near:
            ni = odd_df[c_near].fillna(0).astype(float).values
            row["intersection_active_ratio"] = float(np.mean(ni >= 1.0))
        elif c_inter:
            ic = odd_df[c_inter].fillna(0).astype(float).values
            row["intersection_active_ratio"] = float(np.mean(ic >= 3.0))
        else:
            row["intersection_active_ratio"] = np.nan

        if c_inter:
            ic = odd_df[c_inter].fillna(0).astype(float).values
            row["peak_intersection_complexity"] = float(np.max(ic)) if ic.size else np.nan
        else:
            row["peak_intersection_complexity"] = np.nan

        # lane curvature stats if present
        for base in ["current_lane_curv_mean", "current_lane_curv_p90", "current_lane_curv_max",
                     "current_lane_length_m", "current_lane_heading_change_abs_sum"]:
            row.update(_stats(odd_df[base].values, base) if base in odd_df.columns else _stats(np.array([], dtype=np.float64), base))

        # topology classification (frame-wise -> segment-wise)
        topo = classify_topology_frame(odd_df)
        if len(topo):
            vc = topo.value_counts(normalize=True)
            row["odd_category"] = str(vc.idxmax())  # 04_new expects odd_category
            row["topo_p_intersection"] = float(vc.get("intersection", 0.0))
            row["topo_p_merge"] = float(vc.get("merge", 0.0))
            row["topo_p_split"] = float(vc.get("split", 0.0))
            row["topo_p_straight"] = float(vc.get("straight", 0.0))
        else:
            row["odd_category"] = "unknown"
            row["topo_p_intersection"] = row["topo_p_merge"] = row["topo_p_split"] = row["topo_p_straight"] = np.nan
    else:
        row.update(_stats(np.array([], dtype=np.float64), "infra_density"))
        row["infra_stop_sign_max"] = row["infra_crosswalk_max"] = row["infra_speed_bump_max"] = row["infra_driveway_max"] = np.nan
        row["has_stop_sign"] = row["has_crosswalk"] = row["has_speed_bump"] = row["has_driveway"] = 0
        row["has_traffic_light"] = 0
        row["traffic_light_visible_ratio"] = np.nan
        row["lane_type_mode"] = "unknown"
        row.update(_stats(np.array([], dtype=np.float64), "speed_limit_mph"))
        row["intersection_active_ratio"] = np.nan
        row["peak_intersection_complexity"] = np.nan
        for base in ["current_lane_curv_mean", "current_lane_curv_p90", "current_lane_curv_max",
                     "current_lane_length_m", "current_lane_heading_change_abs_sum"]:
            row.update(_stats(np.array([], dtype=np.float64), base))
        row["odd_category"] = "unknown"
        row["topo_p_intersection"] = row["topo_p_merge"] = row["topo_p_split"] = row["topo_p_straight"] = np.nan

    # -------------------------
    # Interaction complexity / occlusion (MUST come from 02 frame features)
    # -------------------------
    if not odd_df.empty:
        c_int = _col(odd_df, "interaction_object_count", "interaction_count")
        if c_int:
            row["avg_obj_density"] = float(odd_df[c_int].mean())  # 04_new expects avg_obj_density
            row["max_obj_density"] = float(odd_df[c_int].max())
            row.update(_stats(odd_df[c_int].values, "interaction_object_count"))
        else:
            row["avg_obj_density"] = np.nan
            row["max_obj_density"] = np.nan
            row.update(_stats(np.array([], dtype=np.float64), "interaction_object_count"))

        for k in ["interaction_vehicle_count", "interaction_pedestrian_count", "interaction_cyclist_count", "interaction_sign_count",
                  "near_object_count_30m",
                  "mean_lidar_points_in_box", "occlusion_low_points_ratio",
                  # NEW: dynamic-only perception quality aggregates (from 02 v3+)
                  "dyn_label_count", "dyn_label_count_30m",
                  "mean_lidar_points_in_box_dyn", "mean_lidar_points_in_box_dyn_30m",
                  "occlusion_low_points_ratio_dyn", "occlusion_low_points_ratio_dyn_30m"]:

            row.update(_stats(odd_df[k].values, k) if k in odd_df.columns else _stats(np.array([], dtype=np.float64), k))

        counts_mean = np.array([
            row.get("interaction_vehicle_count_mean", 0.0) or 0.0,
            row.get("interaction_pedestrian_count_mean", 0.0) or 0.0,
            row.get("interaction_cyclist_count_mean", 0.0) or 0.0,
            row.get("interaction_sign_count_mean", 0.0) or 0.0,
        ], dtype=np.float64)
        row["interaction_type_entropy"] = _entropy_from_counts(counts_mean)

        # bins for conditional sampling (deterministic thresholds; tune later)
        row["traffic_density_bin"] = _bin_by_threshold(row.get("interaction_object_count_mean", np.nan), 10.0, 30.0)
        row["occlusion_bin"] = _bin_by_threshold(row.get("occlusion_low_points_ratio_mean", np.nan), 0.10, 0.30)
        row["infra_density_bin"] = _bin_by_threshold(row.get("infra_density_mean", np.nan), 1e-4, 5e-4)
    else:
        row["avg_obj_density"] = np.nan
        row["max_obj_density"] = np.nan
        row.update(_stats(np.array([], dtype=np.float64), "interaction_object_count"))
        for k in ["interaction_vehicle_count", "interaction_pedestrian_count", "interaction_cyclist_count", "interaction_sign_count",
                  "near_object_count_30m",
                  "mean_lidar_points_in_box", "occlusion_low_points_ratio",
                  # NEW: dynamic-only perception quality aggregates (from 02 v3+)
                  "dyn_label_count", "dyn_label_count_30m",
                  "mean_lidar_points_in_box_dyn", "mean_lidar_points_in_box_dyn_30m",
                  "occlusion_low_points_ratio_dyn", "occlusion_low_points_ratio_dyn_30m"]:

            row.update(_stats(np.array([], dtype=np.float64), k))
        row["interaction_type_entropy"] = 0.0
        row["traffic_density_bin"] = "unknown"
        row["occlusion_bin"] = "unknown"
        row["infra_density_bin"] = "unknown"

    # -------------------------
    # Ego kinematics (speed/accel/jerk/yaw-rate)
    # -------------------------
    if not frm_df.empty:
        frm_df = frm_df.sort_values("timestamp_micros")

        # Prefer precomputed (vehicle-frame) signals from 02_v3+
        c_speed = _col(frm_df, "ego_speed_mps")
        c_accel = _col(frm_df, "ego_accel_mps2")
        c_jerk = _col(frm_df, "ego_jerk_mps3")
        c_yaw_rate = _col(frm_df, "ego_yaw_rate_rps")

        c_vlong = _col(frm_df, "ego_v_long_mps")
        c_vlat = _col(frm_df, "ego_v_lat_mps")
        c_alat = _col(frm_df, "ego_a_lat_mps2")

        # Lags (optional)
        lag_cols = [
            "ego_speed_lag1", "ego_accel_lag1",
            "ego_speed_lag2", "ego_accel_lag2",
        ]

        # speed
        speed = None
        if c_speed:
            speed = pd.to_numeric(frm_df[c_speed], errors="coerce").values
        elif c_vlong and c_vlat:
            vlong = pd.to_numeric(frm_df[c_vlong], errors="coerce").values
            vlat = pd.to_numeric(frm_df[c_vlat], errors="coerce").values
            speed = np.sqrt(vlong * vlong + vlat * vlat)
        else:
            # fallback: compute from ego_vx/vy if present
            c_vx = _col(frm_df, "ego_vx")
            c_vy = _col(frm_df, "ego_vy")
            if c_vx and c_vy:
                vx = pd.to_numeric(frm_df[c_vx], errors="coerce").values
                vy = pd.to_numeric(frm_df[c_vy], errors="coerce").values
                speed = np.sqrt(vx * vx + vy * vy)

        if speed is not None:
            row.update(_stats(speed, "ego_speed_mps"))
            row["ego_stop_ratio"] = float(np.mean(_finite(speed) < 0.1)) if np.isfinite(speed).any() else np.nan
            row["speed_bin"] = _bin_by_threshold(row.get("ego_speed_mps_mean", np.nan), 5.0, 15.0)
        else:
            row.update(_stats(np.array([], dtype=np.float64), "ego_speed_mps"))
            row["ego_stop_ratio"] = np.nan
            row["speed_bin"] = "unknown"

        # accel / jerk / yaw-rate (use precomputed if available)
        if c_accel:
            accel = pd.to_numeric(frm_df[c_accel], errors="coerce").values
            row.update(_stats(accel, "ego_accel_mps2"))
        else:
            row.update(_stats(np.array([], dtype=np.float64), "ego_accel_mps2"))

        if c_jerk:
            jerk = pd.to_numeric(frm_df[c_jerk], errors="coerce").values
            row.update(_stats(jerk, "ego_jerk_mps3"))
        else:
            row.update(_stats(np.array([], dtype=np.float64), "ego_jerk_mps3"))

        if c_yaw_rate:
            yr = pd.to_numeric(frm_df[c_yaw_rate], errors="coerce").values
            row.update(_stats(yr, "ego_yaw_rate_rps"))
        else:
            row.update(_stats(np.array([], dtype=np.float64), "ego_yaw_rate_rps"))

        # Vehicle-frame components (optional)
        if c_vlong:
            row.update(_stats(pd.to_numeric(frm_df[c_vlong], errors="coerce").values, "ego_v_long_mps"))
        else:
            row.update(_stats(np.array([], dtype=np.float64), "ego_v_long_mps"))
        if c_vlat:
            row.update(_stats(pd.to_numeric(frm_df[c_vlat], errors="coerce").values, "ego_v_lat_mps"))
        else:
            row.update(_stats(np.array([], dtype=np.float64), "ego_v_lat_mps"))
        if c_alat:
            row.update(_stats(pd.to_numeric(frm_df[c_alat], errors="coerce").values, "ego_a_lat_mps2"))
        else:
            row.update(_stats(np.array([], dtype=np.float64), "ego_a_lat_mps2"))

        # lag stats
        for lc in lag_cols:
            if lc in frm_df.columns:
                row.update(_stats(pd.to_numeric(frm_df[lc], errors="coerce").values, lc))
            else:
                row.update(_stats(np.array([], dtype=np.float64), lc))
    else:
        row.update(_stats(np.array([], dtype=np.float64), "ego_speed_mps"))
        row.update(_stats(np.array([], dtype=np.float64), "ego_accel_mps2"))
        row.update(_stats(np.array([], dtype=np.float64), "ego_jerk_mps3"))
        row.update(_stats(np.array([], dtype=np.float64), "ego_yaw_rate_rps"))
        row.update(_stats(np.array([], dtype=np.float64), "ego_v_long_mps"))
        row.update(_stats(np.array([], dtype=np.float64), "ego_v_lat_mps"))
        row.update(_stats(np.array([], dtype=np.float64), "ego_a_lat_mps2"))
        row.update(_stats(np.array([], dtype=np.float64), "ego_speed_lag1"))
        row.update(_stats(np.array([], dtype=np.float64), "ego_accel_lag1"))
        row.update(_stats(np.array([], dtype=np.float64), "ego_speed_lag2"))
        row.update(_stats(np.array([], dtype=np.float64), "ego_accel_lag2"))
        row["ego_stop_ratio"] = np.nan
        row["speed_bin"] = "unknown"

# -------------------------
    # Static map summary (optional)
    # -------------------------
    if smap_df is not None and not smap_df.empty:
        for c in ["map_lane_count", "map_stop_sign_count", "map_crosswalk_count", "map_speed_bump_count", "map_driveway_count"]:
            row[c] = int(pd.to_numeric(smap_df[c], errors="coerce").fillna(0).iloc[0]) if c in smap_df.columns else 0
    else:
        for c in ["map_lane_count", "map_stop_sign_count", "map_crosswalk_count", "map_speed_bump_count", "map_driveway_count"]:
            row[c] = 0

    # -------------------------
    # Quality flag
    # -------------------------
    row["is_valid_for_gssm"] = int(
        (row.get("ego_speed_mps_mean", 0.0) or 0.0) > 0.1 and
        (row.get("interaction_object_count_mean", 0.0) or 0.0) > 0.0 and
        (row.get("duration_s", 0.0) or 0.0) > 1.0
    )

    return row


def _read_partition(root: str, seg_id: str) -> pd.DataFrame:
    path = os.path.join(root, f"segment_id={seg_id}")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pq.read_table(path).to_pandas()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--out_name",
        type=str,
        default="segment_metadata.parquet",
        help="Output filename under staging/ (default keeps 04_new compatibility)",
    )
    args = ap.parse_args()

    staging = os.path.join(args.out_root, "staging")
    odd_root = os.path.join(staging, "base_odd_features")
    frm_root = os.path.join(staging, "base_frames")
    smap_root = os.path.join(staging, "segment_static_map")

    out_path = os.path.join(staging, args.out_name)
    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_path} (use --overwrite)")

    seg_ids = sorted([d.split("segment_id=")[-1] for d in os.listdir(frm_root) if d.startswith("segment_id=")])
    if not seg_ids:
        raise RuntimeError(f"No segments found under: {frm_root}")

    rows = []
    print(f"[03_v7] Processing segments: {len(seg_ids)}")

    for sid in seg_ids:
        frm = _read_partition(frm_root, sid)
        odd = _read_partition(odd_root, sid)
        smap = _read_partition(smap_root, sid)

        try:
            rows.append(summarize_segment_v7(sid, odd, frm, smap))
        except Exception as e:
            print(f"  - Error in {sid}: {e}")
            rows.append({"segment_id": sid, "processed_at": datetime.now().isoformat(timespec="seconds"), "version": "v7", "error": str(e)})

    df = pd.DataFrame(rows)

    front = [
        "segment_id", "processed_at", "version",
        "odd_time", "odd_weather", "odd_location",
        "odd_category", "traffic_density_bin", "infra_density_bin", "occlusion_bin", "speed_bin",
        "n_frames", "duration_s", "is_valid_for_gssm",
        "avg_obj_density", "max_obj_density",
        "has_crosswalk", "has_stop_sign", "has_speed_bump", "has_driveway",
    ]
    cols = front + [c for c in df.columns if c not in front]
    df = df[cols]

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path)
    print(f"[Done] Wrote: {out_path}  (rows={len(df)})")


if __name__ == "__main__":
    main()

