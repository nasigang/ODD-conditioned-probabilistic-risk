#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_build_base_tables_v7.py
(v7 patched: map_features oneof fix)

핵심 변경사항
1) frame_label을 enumerate(i)가 아니라, "세그먼트 단위"로 증가하는 seg_frame_idx 사용 (중복 방지)
2) segment_static_map은 map_features 최초 캐시 시점에 즉시 1회 저장(중단/중간 flush에도 안전)
3) TFRecord compression은 확장자 추정 + 실패 시 자동 재시도(NONE -> GZIP)
4) use_map_pose_offset 선택 결과 및 lane 매칭 품질(lane_match_dist)을 프레임/ODD에 기록(사후 QC)

출력
- staging/base_frames (partitioned by segment_id)
- staging/base_objects (partitioned by segment_id)
- staging/base_odd_features (partitioned by segment_id)
- staging/segment_static_map (partitioned by segment_id, 1 row per segment)

권장 실행
python 02_build_base_tables_v7.py --wod_root <tfrecord_root> --out_root <out_root> --overwrite
"""

# IMPORTANT:
# - Waymo LaserLabel.box.center_{x,y,z} are in VEHICLE frame. (Must transform for global TTC)
# - This script outputs object x/y/speed_x/speed_y in GLOBAL/MAP frame (primary),
#   and also keeps vehicle-frame values in local_* / *_vehicle columns for debugging.

import argparse
import glob
import json
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    raise RuntimeError("pyarrow is required to write parquet datasets") from e


# -------------------------
# Small utilities
# -------------------------
def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    try:
        return getattr(obj, attr)
    except Exception:
        return default


def _ensure_empty_dir(path: str, overwrite: bool):
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f"Output dir exists: {path} (use --overwrite)")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _write_df_to_dataset(df: pd.DataFrame, out_dir: str, partition_cols: List[str]):
    if df.empty:
        return
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(table, root_path=out_dir, partition_cols=partition_cols)


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


# -------------------------
# Pose / Ego helpers
# -------------------------
def _pose_matrix(frame) -> np.ndarray:
    # frame.pose.transform is a 4x4 row-major float array
    T = np.array(frame.pose.transform, dtype=np.float64).reshape(4, 4)
    return T


def _yaw_from_pose(T: np.ndarray) -> float:
    # yaw from rotation matrix (assuming z-up)
    return float(np.arctan2(T[1, 0], T[0, 0]))


def _ego_xy_modes(frame) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Returns:
      (x_no_offset, y_no_offset, x_with_offset, y_with_offset, T)
    """
    T = _pose_matrix(frame)
    x0, y0 = float(T[0, 3]), float(T[1, 3])
    ox = float(frame.map_pose_offset.x) if frame.HasField("map_pose_offset") else 0.0
    oy = float(frame.map_pose_offset.y) if frame.HasField("map_pose_offset") else 0.0
    x1, y1 = x0 + ox, y0 + oy
    return x0, y0, x1, y1, T


# -------------------------
# Nearest neighbor index (lane matching)
# -------------------------
class _NNIndex:
    def __init__(self, pts_xy: np.ndarray):
        self.pts = np.asarray(pts_xy, dtype=np.float64)
        self.mode = "bruteforce"
        self.tree = None
        try:
            from scipy.spatial import cKDTree  # optional
            self.tree = cKDTree(self.pts)
            self.mode = "ckdtree"
        except Exception:
            self.tree = None
            self.mode = "bruteforce"

    def query(self, xy: np.ndarray) -> Tuple[float, int]:
        if self.pts.size == 0:
            return np.nan, -1
        q = np.asarray(xy, dtype=np.float64).reshape(1, 2)
        if self.tree is not None:
            dist, idx = self.tree.query(q, k=1)
            return float(dist[0]), int(idx[0])
        d = np.linalg.norm(self.pts - q, axis=1)
        idx = int(np.argmin(d))
        return float(d[idx]), idx


# -------------------------
# Map feature parsing
# -------------------------
def _polyline_xy(mp_list) -> np.ndarray:
    if not mp_list:
        return np.zeros((0, 2), dtype=np.float64)
    xs = [float(p.x) for p in mp_list]
    ys = [float(p.y) for p in mp_list]
    return np.stack([xs, ys], axis=1).astype(np.float64)


def _mean_xy_from_poly(poly_xy: np.ndarray) -> np.ndarray:
    if poly_xy.size == 0:
        return np.array([np.nan, np.nan], dtype=np.float64)
    return np.mean(poly_xy, axis=0)


def _lane_geom_stats(poly_xy: np.ndarray) -> Dict[str, float]:
    if poly_xy.shape[0] < 2:
        return {
            "lane_length_m": np.nan,
            "lane_heading_change_abs_sum": np.nan,
            "lane_curv_mean": np.nan,
            "lane_curv_p90": np.nan,
            "lane_curv_max": np.nan,
        }

    dxy = np.diff(poly_xy, axis=0)
    seg_len = np.linalg.norm(dxy, axis=1)
    lane_length = float(np.sum(seg_len))

    headings = np.arctan2(dxy[:, 1], dxy[:, 0])
    dhead = _wrap_pi(np.diff(headings))
    heading_change_abs_sum = float(np.sum(np.abs(dhead))) if dhead.size else 0.0

    # curvature proxy: |dheading| / ds
    ds = seg_len[1:]  # aligns with dhead
    valid = ds > 1e-6
    curv = np.abs(dhead[valid] / ds[valid]) if np.any(valid) else np.array([], dtype=np.float64)

    if curv.size == 0:
        return {
            "lane_length_m": lane_length,
            "lane_heading_change_abs_sum": heading_change_abs_sum,
            "lane_curv_mean": np.nan,
            "lane_curv_p90": np.nan,
            "lane_curv_max": np.nan,
        }

    return {
        "lane_length_m": lane_length,
        "lane_heading_change_abs_sum": heading_change_abs_sum,
        "lane_curv_mean": float(np.mean(curv)),
        "lane_curv_p90": float(np.percentile(curv, 90)),
        "lane_curv_max": float(np.max(curv)),
    }


def _map_feature_kind(mf: Any) -> Optional[str]:
    """Return MapFeature oneof kind robustly.

    Waymo MapFeature uses a protobuf oneof (commonly named 'feature_data').
    Accessing mf.stop_sign / mf.lane directly can yield default submessages,
    which breaks type checks like `is not None`.
    """
    for oneof_name in ("feature_data", "feature", "data"):
        try:
            k = mf.WhichOneof(oneof_name)
        except Exception:
            k = None
        if k:
            return str(k)
    return None

def build_map_cache_and_static_row(segment_id: str, map_features) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build:
      - stop_sign_xy, crosswalk_xy, speed_bump_xy, driveway_xy : Nx2 points (centroid approximation via mean)
      - lane_points_xy : all lane polyline points
      - lane_meta_by_id : lane_id -> meta
      - lane_nn : NN index on lane_points_xy
    Also returns a segment_static_map row (segment-level).
    """
    stop_sign_xy = []
    crosswalk_xy = []
    speed_bump_xy = []
    driveway_xy = []

    # lane points for generic lane matching (all lanes)
    lane_points = []
    lane_point_lane_ids = []

    # lane points restricted to interpolating lanes (typically intersection connectors)
    inter_lane_points = []
    inter_lane_point_lane_ids = []

    lane_meta_by_id: Dict[int, Dict[str, Any]] = {}

    counts = {
        "map_features_count": 0,
        "map_lane_count": 0,
        "map_stop_sign_count": 0,
        "map_crosswalk_count": 0,
        "map_speed_bump_count": 0,
        "map_driveway_count": 0,
        "map_road_line_count": 0,
        "map_road_edge_count": 0,
    }
    for mf in map_features:
        counts["map_features_count"] += 1
        mf_id = int(_safe_get(mf, "id", 0))

        fd = _map_feature_kind(mf)
        if not fd:
            continue

        # IMPORTANT:
        # - Always branch by protobuf oneof kind (fd).
        # - Do NOT rely on `mf.stop_sign is not None` etc. (default submessages can appear).
        if fd == "stop_sign":
            counts["map_stop_sign_count"] += 1
            ss = _safe_get(mf, "stop_sign", None)
            p = _safe_get(ss, "position", None) if ss is not None else None
            if p is not None:
                try:
                    x, y = float(p.x), float(p.y)
                    if np.isfinite(x) and np.isfinite(y):
                        stop_sign_xy.append([x, y])
                except Exception:
                    pass
            continue

        if fd == "crosswalk":
            counts["map_crosswalk_count"] += 1
            cw = _safe_get(mf, "crosswalk", None)
            poly = _polyline_xy(_safe_get(cw, "polygon", None))
            xy = _mean_xy_from_poly(poly)
            if np.all(np.isfinite(xy)):
                crosswalk_xy.append([float(xy[0]), float(xy[1])])
            continue

        if fd == "speed_bump":
            counts["map_speed_bump_count"] += 1
            sb = _safe_get(mf, "speed_bump", None)
            poly = _polyline_xy(_safe_get(sb, "polygon", None))
            xy = _mean_xy_from_poly(poly)
            if np.all(np.isfinite(xy)):
                speed_bump_xy.append([float(xy[0]), float(xy[1])])
            continue

        if fd == "driveway":
            counts["map_driveway_count"] += 1
            dw = _safe_get(mf, "driveway", None)
            poly = _polyline_xy(_safe_get(dw, "polygon", None))
            xy = _mean_xy_from_poly(poly)
            if np.all(np.isfinite(xy)):
                driveway_xy.append([float(xy[0]), float(xy[1])])
            continue

        if fd == "road_line":
            counts["map_road_line_count"] += 1
            continue

        if fd == "road_edge":
            counts["map_road_edge_count"] += 1
            continue

        if fd == "lane":
            counts["map_lane_count"] += 1
            ln = _safe_get(mf, "lane", None)

            interpolating = int(bool(_safe_get(ln, "interpolating", False)))
            poly = _polyline_xy(_safe_get(ln, "polyline", None))
            if poly.size:
                for pt in poly:
                    xpt, ypt = float(pt[0]), float(pt[1])
                    lane_points.append([xpt, ypt])
                    lane_point_lane_ids.append(mf_id)
                    if interpolating == 1:
                        inter_lane_points.append([xpt, ypt])
                        inter_lane_point_lane_ids.append(mf_id)

            entry_count = len(getattr(ln, "entry_lanes", [])) if ln is not None else 0
            exit_count = len(getattr(ln, "exit_lanes", [])) if ln is not None else 0
            left_n = len(getattr(ln, "left_neighbors", [])) if ln is not None else 0
            right_n = len(getattr(ln, "right_neighbors", [])) if ln is not None else 0

            lane_type = int(_safe_get(ln, "type", 0))
            speed_limit = float(_safe_get(ln, "speed_limit_mph", np.nan))

            geom = _lane_geom_stats(poly)

            lane_meta_by_id[mf_id] = {
                "lane_id": mf_id,
                "lane_type": lane_type,
                "speed_limit_mph": speed_limit,
                "entry_count": entry_count,
                "exit_count": exit_count,
                "left_neighbor_count": left_n,
                "right_neighbor_count": right_n,
                "interpolating": interpolating,
                **geom,
            }
            continue

        continue

    stop_sign_xy = np.asarray(stop_sign_xy, dtype=np.float64).reshape(-1, 2) if stop_sign_xy else np.zeros((0, 2), dtype=np.float64)
    crosswalk_xy = np.asarray(crosswalk_xy, dtype=np.float64).reshape(-1, 2) if crosswalk_xy else np.zeros((0, 2), dtype=np.float64)
    speed_bump_xy = np.asarray(speed_bump_xy, dtype=np.float64).reshape(-1, 2) if speed_bump_xy else np.zeros((0, 2), dtype=np.float64)
    driveway_xy = np.asarray(driveway_xy, dtype=np.float64).reshape(-1, 2) if driveway_xy else np.zeros((0, 2), dtype=np.float64)

    lane_points_xy = np.asarray(lane_points, dtype=np.float64).reshape(-1, 2) if lane_points else np.zeros((0, 2), dtype=np.float64)
    lane_point_lane_ids = np.asarray(lane_point_lane_ids, dtype=np.int64) if lane_point_lane_ids else np.zeros((0,), dtype=np.int64)

    lane_nn = _NNIndex(lane_points_xy)

    inter_lane_points_xy = np.asarray(inter_lane_points, dtype=np.float64).reshape(-1, 2) if inter_lane_points else np.zeros((0, 2), dtype=np.float64)
    inter_lane_point_lane_ids = np.asarray(inter_lane_point_lane_ids, dtype=np.int64) if inter_lane_point_lane_ids else np.zeros((0,), dtype=np.int64)
    inter_lane_nn = _NNIndex(inter_lane_points_xy)

    cache = {
        "stop_sign_xy": stop_sign_xy,
        "crosswalk_xy": crosswalk_xy,
        "speed_bump_xy": speed_bump_xy,
        "driveway_xy": driveway_xy,
        "lane_points_xy": lane_points_xy,
        "lane_point_lane_ids": lane_point_lane_ids,
        "inter_lane_points_xy": inter_lane_points_xy,
        "inter_lane_point_lane_ids": inter_lane_point_lane_ids,
        "inter_lane_nn": inter_lane_nn,
        "lane_meta_by_id": lane_meta_by_id,
        "lane_nn": lane_nn,
    }

    # segment_static_map row
    static_row = {
        "segment_id": str(segment_id),
        "static_map_found": 1,
        **counts,
        "lane_nn_mode": lane_nn.mode,
    }
    return cache, static_row


def _count_within_radius(xy: np.ndarray, x: float, y: float, r: float) -> int:
    if xy is None or xy.size == 0:
        return 0
    d = np.linalg.norm(xy - np.asarray([x, y], dtype=np.float64), axis=1)
    return int(np.sum(d <= r))

def _min_dist(xy: np.ndarray, x: float, y: float) -> float:
    """Min Euclidean distance from (x,y) to a point set. Returns NaN if empty."""
    if xy is None or xy.size == 0:
        return float("nan")
    d = np.linalg.norm(xy - np.asarray([x, y], dtype=np.float64), axis=1)
    return float(np.min(d)) if d.size else float("nan")

def _ema(prev: Optional[float], cur: float, alpha: float) -> float:
    """Causal exponential moving average (EMA).

    - Prevents future leakage (uses only past state).
    - If `cur` is NaN/inf, keeps previous value.
    - If `prev` is None/NaN, initializes with `cur`.
    """
    try:
        cur_f = float(cur)
    except Exception:
        cur_f = float("nan")
    if not np.isfinite(cur_f):
        return float(prev) if (prev is not None and np.isfinite(prev)) else float("nan")

    if prev is None or (isinstance(prev, float) and (not np.isfinite(prev))):
        return cur_f

    a = float(alpha)
    if a < 0.0:
        a = 0.0
    if a > 1.0:
        a = 1.0
    return float(a * cur_f + (1.0 - a) * float(prev))




def _match_lane(map_cache: Dict[str, Any], x: float, y: float, lane_match_max_dist: float) -> Tuple[int, float, Dict[str, Any]]:
    """
    Returns:
      lane_id, lane_dist_m, lane_meta (dict) if matched else (-1, nan, {})
    """
    lane_nn: _NNIndex = map_cache["lane_nn"]
    dist, idx = lane_nn.query(np.asarray([x, y], dtype=np.float64))
    if not np.isfinite(dist) or dist > lane_match_max_dist or idx < 0:
        return -1, np.nan, {}
    lane_id = int(map_cache["lane_point_lane_ids"][idx]) if map_cache["lane_point_lane_ids"].size else -1
    meta = map_cache["lane_meta_by_id"].get(lane_id, {})
    return lane_id, float(dist), meta


def _choose_offset_mode_once(frame, map_cache: Dict[str, Any]) -> bool:
    """
    Decide once per segment: use map_pose_offset or not.
    Policy: pick the pose (offset vs no-offset) that is closer to the nearest lane point.
    """
    x0, y0, x1, y1, _T = _ego_xy_modes(frame)

    lane_nn: _NNIndex = map_cache["lane_nn"]
    d0, _ = lane_nn.query(np.asarray([x0, y0], dtype=np.float64))
    d1, _ = lane_nn.query(np.asarray([x1, y1], dtype=np.float64))

    if not np.isfinite(d0) and np.isfinite(d1):
        return True
    if np.isfinite(d0) and not np.isfinite(d1):
        return False
    if not np.isfinite(d0) and not np.isfinite(d1):
        return False
    return bool(d1 < d0)


# -------------------------
# LaserLabel extraction (GLOBAL/MAP primary)
# -------------------------
def _extract_laser_label_row(label, T_vehicle_to_global: np.ndarray, ego_yaw: float) -> Dict[str, Any]:
    """
    IMPORTANT:
    - label.box.center_{x,y,z} are in the *vehicle frame* (ego frame).
    - Downstream (04) typically assumes object x/y are in the same global frame as ego_x/ego_y.
      So here we export global/map-frame coordinates and velocities as primary.
    """
    box = _safe_get(label, "box", None)
    meta = _safe_get(label, "metadata", None)

    # vehicle-frame quantities (Waymo)
    lx = float(_safe_get(box, "center_x", np.nan))
    ly = float(_safe_get(box, "center_y", np.nan))
    lz = float(_safe_get(box, "center_z", np.nan))
    lvx = float(_safe_get(meta, "speed_x", np.nan))
    lvy = float(_safe_get(meta, "speed_y", np.nan))
    lvz = float(_safe_get(meta, "speed_z", np.nan))
    lax = float(_safe_get(meta, "accel_x", np.nan))
    lay = float(_safe_get(meta, "accel_y", np.nan))
    laz = float(_safe_get(meta, "accel_z", np.nan))
    lheading = float(_safe_get(box, "heading", np.nan))

    # global/map-frame transform
    T = np.asarray(T_vehicle_to_global, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]

    gx = gy = gz = np.nan
    gvx = gvy = gvz = np.nan
    gax = gay = gaz = np.nan

    if np.isfinite(lx) and np.isfinite(ly) and np.isfinite(lz):
        p = R @ np.asarray([lx, ly, lz], dtype=np.float64) + t
        gx, gy, gz = float(p[0]), float(p[1]), float(p[2])

    if np.isfinite(lvx) and np.isfinite(lvy) and np.isfinite(lvz):
        v = R @ np.asarray([lvx, lvy, lvz], dtype=np.float64)
        gvx, gvy, gvz = float(v[0]), float(v[1]), float(v[2])

    if np.isfinite(lax) and np.isfinite(lay) and np.isfinite(laz):
        a = R @ np.asarray([lax, lay, laz], dtype=np.float64)
        gax, gay, gaz = float(a[0]), float(a[1]), float(a[2])

    heading_global = np.nan
    if np.isfinite(lheading) and np.isfinite(ego_yaw):
        heading_global = float(_wrap_pi(ego_yaw + lheading))

    obj_t = int(_safe_get(label, "type", 0))

    return {
        "obj_id": str(_safe_get(label, "id", "")),
        "obj_type": obj_t,
        "type": obj_t,  # alias for 04_new

        # global/map-frame (primary, used by 04)
        "x": gx,
        "y": gy,
        "z": gz,
        "speed_x": gvx,
        "speed_y": gvy,
        "speed_z": gvz,
        "accel_x": gax,
        "accel_y": gay,
        "accel_z": gaz,
        "heading": heading_global,

        # vehicle-frame (kept for debugging / alternative ego-frame TTC)
        "local_x": lx,
        "local_y": ly,
        "local_z": lz,
        "speed_x_vehicle": lvx,
        "speed_y_vehicle": lvy,
        "speed_z_vehicle": lvz,
        "accel_x_vehicle": lax,
        "accel_y_vehicle": lay,
        "accel_z_vehicle": laz,
        "heading_vehicle": lheading,

        "length": float(_safe_get(box, "length", np.nan)),
        "width": float(_safe_get(box, "width", np.nan)),
        "height": float(_safe_get(box, "height", np.nan)),
        "num_lidar_points_in_box": int(_safe_get(label, "num_lidar_points_in_box", 0)),
        "is_ego": 0,
    }


# -------------------------
# TFRecord opener (robust)
# -------------------------
def _open_tfrecord_dataset(fp: str):
    import tensorflow as tf

    # try uncompressed, then gzip
    try:
        ds = tf.data.TFRecordDataset(fp)
        return ds, "none"
    except Exception:
        try:
            ds = tf.data.TFRecordDataset(fp, compression_type="GZIP")
            return ds, "gzip"
        except Exception as e:
            raise RuntimeError(f"Cannot open TFRecord: {fp}") from e


# -------------------------
# Main builder
# -------------------------
def build_from_wod_v7(
    tfrecords: List[str],
    out_root: str,
    overwrite: bool,
    radius: float = 50.0,
    lane_match_max_dist: float = 5.0,
    near_radius: float = 20.0,
    intersection_near_dist: float = 15.0,
    max_files: Optional[int] = None,
    flush_every_rows: int = 300_000,
) -> Dict[str, Any]:
    from waymo_open_dataset import dataset_pb2 as open_dataset

    if max_files is not None:
        tfrecords = tfrecords[:max_files]

    base_dirs = {
        "frm": os.path.join(out_root, "staging", "base_frames"),
        "obj": os.path.join(out_root, "staging", "base_objects"),
        "odd": os.path.join(out_root, "staging", "base_odd_features"),
        "smap": os.path.join(out_root, "staging", "segment_static_map"),
    }
    for d in base_dirs.values():
        _ensure_empty_dir(d, overwrite)

    report = {
        "status": "running",
        "processed_files": 0,
        "processed_segments": 0,
        "tfrecords": tfrecords,
        "warnings": [],
        "lane_nn_mode": None,
        "version": "v7",
        "notes": [
            "Object x/y/speed_x/speed_y are exported in GLOBAL/MAP frame (primary).",
            "Vehicle-frame values are kept in local_* and *_vehicle columns.",
        ],
    }

    # per-segment state
    seg_id = None
    seg_frame_idx = 0  # segment-level frame counter
    map_cache = None
    use_map_pose_offset = None
    static_map_written = False

    # ego finite difference (segment-local)
    prev_ts = None
    prev_ego_xy = None
    prev_ego_speed = None
    prev2_ego_speed = None
    prev_ego_accel = None
    prev2_ego_accel = None
    prev_ego_yaw = None
    ema_v_long = None
    ema_yaw_rate = None
    ema_a_long = None

    obj_rows: List[Dict[str, Any]] = []
    frm_rows: List[Dict[str, Any]] = []
    odd_rows: List[Dict[str, Any]] = []

    def flush_rows(current_seg_id: str):
        nonlocal obj_rows, frm_rows, odd_rows
        if current_seg_id is None:
            return
        if obj_rows:
            _write_df_to_dataset(pd.DataFrame(obj_rows), base_dirs["obj"], ["segment_id"])
            obj_rows = []
        if frm_rows:
            _write_df_to_dataset(pd.DataFrame(frm_rows), base_dirs["frm"], ["segment_id"])
            frm_rows = []
        if odd_rows:
            _write_df_to_dataset(pd.DataFrame(odd_rows), base_dirs["odd"], ["segment_id"])
            odd_rows = []

    def flush_if_too_big(current_seg_id: str):
        if len(obj_rows) >= flush_every_rows or len(frm_rows) >= flush_every_rows or len(odd_rows) >= flush_every_rows:
            flush_rows(current_seg_id)

    def write_missing_static_map_row(current_seg_id: str):
        nonlocal static_map_written
        if static_map_written:
            return
        row = {
            "segment_id": str(current_seg_id),
            "static_map_found": 0,
            "map_features_count": 0,
            "map_lane_count": 0,
            "map_stop_sign_count": 0,
            "map_crosswalk_count": 0,
            "map_speed_bump_count": 0,
            "map_driveway_count": 0,
            "map_road_line_count": 0,
            "map_road_edge_count": 0,
            "lane_nn_mode": "none",
        }
        _write_df_to_dataset(pd.DataFrame([row]), base_dirs["smap"], ["segment_id"])
        static_map_written = True

    for fp in tfrecords:
        ds, comp_used = _open_tfrecord_dataset(fp)
        report["processed_files"] += 1

        for file_i, raw in enumerate(ds):
            frame = open_dataset.Frame()
            frame.ParseFromString(raw.numpy())
            frame_seg_id = str(frame.context.name)

            # segment boundary detection
            if seg_id is None:
                seg_id = frame_seg_id
                seg_frame_idx = 0
                map_cache = None
                use_map_pose_offset = None
                static_map_written = False
                prev_ts = None
                prev_ego_xy = None
                prev_ego_speed = None
                prev2_ego_speed = None
                prev_ego_accel = None
                prev2_ego_accel = None
                prev_ego_yaw = None
                ema_v_long = None
                ema_yaw_rate = None
                ema_a_long = None

            if frame_seg_id != seg_id:
                if not static_map_written:
                    write_missing_static_map_row(seg_id)
                flush_rows(seg_id)
                report["processed_segments"] += 1

                # reset for new segment
                seg_id = frame_seg_id
                seg_frame_idx = 0
                map_cache = None
                use_map_pose_offset = None
                static_map_written = False
                prev_ts = None
                prev_ego_xy = None
                prev_ego_speed = None
                prev2_ego_speed = None
                prev_ego_accel = None
                prev2_ego_accel = None
                prev_ego_yaw = None
                ema_v_long = None
                ema_yaw_rate = None
                ema_a_long = None

            # cache map_features when first appears in this segment
            if (map_cache is None) and frame.map_features:
                map_cache, static_row = build_map_cache_and_static_row(seg_id, frame.map_features)
                if report["lane_nn_mode"] is None:
                    report["lane_nn_mode"] = map_cache["lane_nn"].mode
                if not static_map_written:
                    _write_df_to_dataset(pd.DataFrame([static_row]), base_dirs["smap"], ["segment_id"])
                    static_map_written = True

            # choose offset mode once we have lane index
            if use_map_pose_offset is None and map_cache is not None:
                use_map_pose_offset = _choose_offset_mode_once(frame, map_cache)

            # ego pose in global (two modes)
            x0, y0, x1, y1, T = _ego_xy_modes(frame)
            ego_x, ego_y = (x1, y1) if (use_map_pose_offset is True) else (x0, y0)
            # Build a vehicle->global/map transform consistent with the chosen ego_x/ego_y
            # (translation is updated by map_pose_offset selection; rotation stays from frame.pose.transform)
            T_map = np.asarray(T, dtype=np.float64).reshape(4, 4).copy()
            if use_map_pose_offset is True:
                # also apply z offset if available
                oz = float(frame.map_pose_offset.z) if frame.HasField("map_pose_offset") else 0.0
                T_map[0, 3] = float(x1)
                T_map[1, 3] = float(y1)
                T_map[2, 3] = float(T_map[2, 3] + oz)
            else:
                T_map[0, 3] = float(x0)
                T_map[1, 3] = float(y0)

            ego_yaw = _yaw_from_pose(T)
            ts = int(frame.timestamp_micros)
            # ego finite-difference kinematics (global/map -> vehicle frame) + causal smoothing
            dt_s = np.nan
            ego_vx, ego_vy = np.nan, np.nan
            if prev_ts is not None and prev_ego_xy is not None:
                dt_s = (ts - prev_ts) * 1e-6
                if dt_s > 1e-6:
                    ego_vx = (ego_x - prev_ego_xy[0]) / dt_s
                    ego_vy = (ego_y - prev_ego_xy[1]) / dt_s

            speed_valid = bool(np.isfinite(ego_vx) and np.isfinite(ego_vy) and np.isfinite(dt_s) and dt_s > 1e-6)

            # yaw rate (signed; wrap to [-pi, pi])
            yaw_valid = bool(prev_ego_yaw is not None and np.isfinite(prev_ego_yaw) and np.isfinite(dt_s) and dt_s > 1e-6)
            if yaw_valid:
                dy = float(_wrap_pi(np.array([ego_yaw - float(prev_ego_yaw)], dtype=np.float64))[0])
                yaw_rate_raw = float(dy / dt_s)
            else:
                yaw_rate_raw = np.nan

            # vehicle-frame velocity decomposition
            if speed_valid:
                c = float(np.cos(ego_yaw))
                s = float(np.sin(ego_yaw))
                v_long_raw = c * float(ego_vx) + s * float(ego_vy)
                v_lat_raw = -s * float(ego_vx) + c * float(ego_vy)
            else:
                v_long_raw = np.nan
                v_lat_raw = np.nan

            # causal EMA (no future leakage)
            # NOTE: alpha는 baseline에서는 고정값으로 두고, 필요 시 CLI 옵션으로 분리하세요.
            ALPHA_V_LONG = 0.35
            ALPHA_YAW_RATE = 0.35
            ALPHA_A_LONG = 0.35

            v_long_sm = _ema(ema_v_long, v_long_raw, ALPHA_V_LONG) if speed_valid else float("nan")
            yaw_rate_sm = _ema(ema_yaw_rate, yaw_rate_raw, ALPHA_YAW_RATE) if yaw_valid else float("nan")

            # longitudinal accel/jerk on *smoothed* v_long (more stable than raw differencing)
            accel_valid = bool(speed_valid and (ema_v_long is not None) and np.isfinite(v_long_sm) and np.isfinite(dt_s) and dt_s > 1e-6)
            if accel_valid:
                a_long_raw = float((float(v_long_sm) - float(ema_v_long)) / dt_s)
            else:
                a_long_raw = np.nan

            a_long_sm = _ema(ema_a_long, a_long_raw, ALPHA_A_LONG) if np.isfinite(a_long_raw) else float("nan")

            jerk_valid = bool((ema_a_long is not None) and np.isfinite(a_long_sm) and np.isfinite(dt_s) and dt_s > 1e-6)
            if jerk_valid:
                jerk_long = float((float(a_long_sm) - float(ema_a_long)) / dt_s)
            else:
                jerk_long = float("nan")

            # lateral accel proxy (signed): a_lat ≈ v_long * yaw_rate
            if np.isfinite(v_long_sm) and np.isfinite(yaw_rate_sm):
                a_lat = float(float(v_long_sm) * float(yaw_rate_sm))
            else:
                a_lat = float("nan")

            # final outputs (fill first-frame with 0.0 for stability)
            ego_v_long_mps = float(v_long_sm) if np.isfinite(v_long_sm) else 0.0
            ego_v_lat_mps = float(v_lat_raw) if np.isfinite(v_lat_raw) else 0.0
            ego_speed_mps = float(np.hypot(ego_v_long_mps, ego_v_lat_mps))

            ego_yaw_rate_rps = float(yaw_rate_sm) if np.isfinite(yaw_rate_sm) else 0.0
            ego_accel_mps2 = float(a_long_sm) if np.isfinite(a_long_sm) else 0.0
            ego_jerk_mps3 = float(jerk_long) if np.isfinite(jerk_long) else 0.0
            ego_a_lat_mps2 = float(a_lat) if np.isfinite(a_lat) else 0.0

            # lag features (option-1: if missing, use current frame)
            ego_speed_lag1 = float(prev_ego_speed) if (prev_ego_speed is not None and np.isfinite(prev_ego_speed)) else float(ego_speed_mps)
            ego_speed_lag2 = float(prev2_ego_speed) if (prev2_ego_speed is not None and np.isfinite(prev2_ego_speed)) else float(ego_speed_lag1)
            ego_accel_lag1 = float(prev_ego_accel) if (prev_ego_accel is not None and np.isfinite(prev_ego_accel)) else float(ego_accel_mps2)
            ego_accel_lag2 = float(prev2_ego_accel) if (prev2_ego_accel is not None and np.isfinite(prev2_ego_accel)) else float(ego_accel_lag1)

            # frames (segment-level frame_label)
            frm_rows.append({
                "segment_id": seg_id,
                "frame_label": int(seg_frame_idx),
                "timestamp_micros": ts,
                "dt_s": float(dt_s) if np.isfinite(dt_s) else np.nan,
                "ego_x": float(ego_x),
                "ego_y": float(ego_y),
                "ego_gx": float(ego_x),  # legacy
                "ego_gy": float(ego_y),  # legacy
                "ego_yaw": float(ego_yaw),
                "ego_vx": float(ego_vx) if np.isfinite(ego_vx) else np.nan,
                "ego_vy": float(ego_vy) if np.isfinite(ego_vy) else np.nan,
                "ego_v_long_mps": float(ego_v_long_mps),
                "ego_v_lat_mps": float(ego_v_lat_mps),
                "ego_a_lat_mps2": float(ego_a_lat_mps2),
                "ego_speed_mps": float(ego_speed_mps),
                "ego_accel_mps2": float(ego_accel_mps2),
                "ego_yaw_rate_rps": float(ego_yaw_rate_rps),
                "ego_jerk_mps3": float(ego_jerk_mps3),
                "ego_speed_lag1": float(ego_speed_lag1),
                "ego_accel_lag1": float(ego_accel_lag1),
                "ego_speed_lag2": float(ego_speed_lag2),
                "ego_accel_lag2": float(ego_accel_lag2),
                "use_map_pose_offset": int(bool(use_map_pose_offset)) if use_map_pose_offset is not None else 0,
                "tfrecord_file": os.path.basename(fp),
                "file_frame_idx": int(file_i),
                "tfrecord_compression": str(comp_used),
            })
            # update prev states (for next frame)
            prev_ts = ts
            prev_ego_xy = (ego_x, ego_y)

            # EMA states update only when raw inputs are valid (prevents first-frame artifacts)
            if speed_valid and np.isfinite(v_long_raw):
                ema_v_long = float(v_long_sm) if np.isfinite(v_long_sm) else ema_v_long
            if yaw_valid and np.isfinite(yaw_rate_raw):
                ema_yaw_rate = float(yaw_rate_sm) if np.isfinite(yaw_rate_sm) else ema_yaw_rate
            if accel_valid and np.isfinite(a_long_raw):
                ema_a_long = float(a_long_sm) if np.isfinite(a_long_sm) else ema_a_long

            prev2_ego_speed = prev_ego_speed
            if speed_valid:
                prev_ego_speed = float(ego_speed_mps)

            prev2_ego_accel = prev_ego_accel
            if accel_valid:
                prev_ego_accel = float(ego_accel_mps2)

            prev_ego_yaw = ego_yaw

            # ODD (frame-level)
            stats = frame.context.stats
            odd_feat = {
                "segment_id": seg_id,
                "frame_label": int(seg_frame_idx),
                "timestamp_micros": ts,
                "time_of_day": str(_safe_get(stats, "time_of_day", "")),
                "location": str(_safe_get(stats, "location", "")),
                "weather": str(_safe_get(stats, "weather", "")),
                "radius_m": float(radius),

                "near_radius_m": float(near_radius),
                "intersection_near_dist_m": float(intersection_near_dist),

                # dynamic near-flags (frame-level)
                "is_near_intersection": 0,
                "is_near_stop_sign": 0,
                "is_near_crosswalk": 0,
                "is_near_speed_bump": 0,
                "is_near_driveway": 0,

                # optional debug distances (meters)
                "dist_to_intersection_lane_m": np.nan,
                "dist_to_stop_sign_m": np.nan,
                "dist_to_crosswalk_m": np.nan,
                "dist_to_speed_bump_m": np.nan,
                "dist_to_driveway_m": np.nan,

                # camera-label based traffic-light visibility (frame-level)
                "traffic_light_visible_count": 0,
                "has_traffic_light_visible": 0,

                # derived (intersection + traffic light)
                "is_near_signalized_intersection": 0,
                "dist_to_signalized_intersection_lane_m": np.nan,

                # aliases for downstream (03_new/04_new)
                "infra_stop_sign_count": 0,
                "infra_crosswalk_count": 0,
                "infra_speed_bump_count": 0,
                "infra_driveway_count": 0,

                # interaction / perception complexity (frame-level)
                "interaction_object_count": 0,
                "interaction_vehicle_count": 0,
                "interaction_pedestrian_count": 0,
                "interaction_cyclist_count": 0,
                "interaction_sign_count": 0,
                "near_object_count_30m": 0,
                "mean_lidar_points_in_box": np.nan,
                "occlusion_low_points_ratio": np.nan,

                # improved perception-quality (dynamic-only + 30m)
                "dyn_label_count": 0,
                "dyn_label_count_30m": 0,
                "mean_lidar_points_in_box_dyn": np.nan,
                "mean_lidar_points_in_box_dyn_30m": np.nan,
                "occlusion_low_points_ratio_dyn": np.nan,
                "occlusion_low_points_ratio_dyn_30m": np.nan,

                "stop_sign_count_r": 0,
                "crosswalk_count_r": 0,
                "speed_bump_count_r": 0,
                "driveway_count_r": 0,
                "infra_density_r": np.nan,

                "lane_id": -1,
                "lane_match_dist_m": np.nan,
                "lane_type": np.nan,
                "speed_limit_mph": np.nan,
                "left_neighbor_count": np.nan,
                "right_neighbor_count": np.nan,
                "entry_count": np.nan,
                "exit_count": np.nan,
                "interpolating": np.nan,
                "lane_entry_count": np.nan,
                "lane_exit_count": np.nan,
                "lane_interpolating": np.nan,
                "intersection_complexity": np.nan,

                "current_lane_length_m": np.nan,
                "current_lane_heading_change_abs_sum": np.nan,
                "current_lane_curv_mean": np.nan,
                "current_lane_curv_p90": np.nan,
                "current_lane_curv_max": np.nan,

                "use_map_pose_offset": int(bool(use_map_pose_offset)) if use_map_pose_offset is not None else 0,
            }

            if map_cache is not None:
                ss = _count_within_radius(map_cache["stop_sign_xy"], ego_x, ego_y, radius)
                cw = _count_within_radius(map_cache["crosswalk_xy"], ego_x, ego_y, radius)
                sb = _count_within_radius(map_cache["speed_bump_xy"], ego_x, ego_y, radius)
                dw = _count_within_radius(map_cache["driveway_xy"], ego_x, ego_y, radius)

                odd_feat["stop_sign_count_r"] = ss
                odd_feat["crosswalk_count_r"] = cw
                odd_feat["speed_bump_count_r"] = sb
                odd_feat["driveway_count_r"] = dw

                # alias names (03_new expects infra_*_count)
                odd_feat["infra_stop_sign_count"] = ss
                odd_feat["infra_crosswalk_count"] = cw
                odd_feat["infra_speed_bump_count"] = sb
                odd_feat["infra_driveway_count"] = dw

                area = np.pi * (radius ** 2)
                odd_feat["infra_density_r"] = float((ss + cw + sb + dw) / area) if area > 0 else np.nan

                # --- dynamic near flags (frame-level) ---
                d_ss = _min_dist(map_cache["stop_sign_xy"], ego_x, ego_y)
                d_cw = _min_dist(map_cache["crosswalk_xy"], ego_x, ego_y)
                d_sb = _min_dist(map_cache["speed_bump_xy"], ego_x, ego_y)
                d_dw = _min_dist(map_cache["driveway_xy"], ego_x, ego_y)

                odd_feat["dist_to_stop_sign_m"] = d_ss
                odd_feat["dist_to_crosswalk_m"] = d_cw
                odd_feat["dist_to_speed_bump_m"] = d_sb
                odd_feat["dist_to_driveway_m"] = d_dw

                odd_feat["is_near_stop_sign"] = int(np.isfinite(d_ss) and (d_ss <= near_radius))
                odd_feat["is_near_crosswalk"] = int(np.isfinite(d_cw) and (d_cw <= near_radius))
                odd_feat["is_near_speed_bump"] = int(np.isfinite(d_sb) and (d_sb <= near_radius))
                odd_feat["is_near_driveway"] = int(np.isfinite(d_dw) and (d_dw <= near_radius))

                inter_nn = map_cache.get("inter_lane_nn", None)
                if inter_nn is not None:
                    d_int, _idx_int = inter_nn.query(np.asarray([ego_x, ego_y], dtype=np.float64))
                else:
                    d_int = float("nan")
                odd_feat["dist_to_intersection_lane_m"] = float(d_int) if np.isfinite(d_int) else float("nan")
                odd_feat["is_near_intersection"] = int(np.isfinite(d_int) and (d_int <= intersection_near_dist))


                lane_id, lane_dist, meta = _match_lane(map_cache, ego_x, ego_y, lane_match_max_dist)
                odd_feat["lane_id"] = int(lane_id)
                odd_feat["lane_match_dist_m"] = float(lane_dist) if np.isfinite(lane_dist) else np.nan

                if meta:
                    odd_feat["lane_type"] = float(meta.get("lane_type", np.nan))
                    odd_feat["speed_limit_mph"] = float(meta.get("speed_limit_mph", np.nan))
                    odd_feat["left_neighbor_count"] = float(meta.get("left_neighbor_count", np.nan))
                    odd_feat["right_neighbor_count"] = float(meta.get("right_neighbor_count", np.nan))
                    odd_feat["entry_count"] = float(meta.get("entry_count", np.nan))
                    odd_feat["exit_count"] = float(meta.get("exit_count", np.nan))
                    odd_feat["interpolating"] = float(meta.get("interpolating", np.nan))

                    # alias names (03_new expects lane_* )
                    odd_feat["lane_entry_count"] = float(meta.get("entry_count", np.nan))
                    odd_feat["lane_exit_count"] = float(meta.get("exit_count", np.nan))
                    odd_feat["lane_interpolating"] = float(meta.get("interpolating", np.nan))


                    # If current lane itself is interpolating, treat as intersection-near (strong signal)
                    if int(meta.get("interpolating", 0)) == 1:
                        odd_feat["is_near_intersection"] = 1
                        odd_feat["dist_to_intersection_lane_m"] = 0.0

                    deg = meta.get("entry_count", 0) + meta.get("exit_count", 0)
                    nei = meta.get("left_neighbor_count", 0) + meta.get("right_neighbor_count", 0)
                    itp = meta.get("interpolating", 0)
                    odd_feat["intersection_complexity"] = float(deg + nei + itp)

                    odd_feat["current_lane_length_m"] = float(meta.get("lane_length_m", np.nan))
                    odd_feat["current_lane_heading_change_abs_sum"] = float(meta.get("lane_heading_change_abs_sum", np.nan))
                    odd_feat["current_lane_curv_mean"] = float(meta.get("lane_curv_mean", np.nan))
                    odd_feat["current_lane_curv_p90"] = float(meta.get("lane_curv_p90", np.nan))
                    odd_feat["current_lane_curv_max"] = float(meta.get("lane_curv_max", np.nan))

            # -----------------
            # Interaction / perception complexity (frame-level)
            # -----------------
            labs = list(frame.laser_labels)
            odd_feat["interaction_object_count"] = int(len(labs))
            if labs:
                # label.type: 1 vehicle, 2 pedestrian, 3 sign, 4 cyclist (Waymo conventions)
                types = np.array([int(_safe_get(l, "type", 0)) for l in labs], dtype=np.int32)
                odd_feat["interaction_vehicle_count"] = int(np.sum(types == 1))
                odd_feat["interaction_pedestrian_count"] = int(np.sum(types == 2))
                odd_feat["interaction_sign_count"] = int(np.sum(types == 3))
                odd_feat["interaction_cyclist_count"] = int(np.sum(types == 4))

                # objects within 30 m around ego (use vehicle-frame position; ego is at origin there)
                loc_xy = np.array([[float(_safe_get(l.box, "center_x", np.nan)),
                                    float(_safe_get(l.box, "center_y", np.nan))] for l in labs], dtype=np.float64)
                d = np.linalg.norm(loc_xy, axis=1)
                near30 = d <= 30.0
                odd_feat["near_object_count_30m"] = int(np.sum(near30))

                pts = np.array([int(_safe_get(l, "num_lidar_points_in_box", 0)) for l in labs], dtype=np.int32)
                odd_feat["mean_lidar_points_in_box"] = float(np.mean(pts)) if pts.size else np.nan
                # simple occlusion proxy: low lidar points ratio
                low = np.sum(pts <= 5)
                odd_feat["occlusion_low_points_ratio"] = float(low / pts.size) if pts.size else np.nan

                # improved: dynamic-only(veh/ped/cyc) + 30m to reduce distance/type bias
                dyn = (types == 1) | (types == 2) | (types == 4)
                odd_feat["dyn_label_count"] = int(np.sum(dyn))
                if np.any(dyn):
                    pts_dyn = pts[dyn]
                    odd_feat["mean_lidar_points_in_box_dyn"] = float(np.mean(pts_dyn)) if pts_dyn.size else np.nan
                    odd_feat["occlusion_low_points_ratio_dyn"] = float(np.sum(pts_dyn <= 5) / pts_dyn.size) if pts_dyn.size else np.nan
                else:
                    odd_feat["mean_lidar_points_in_box_dyn"] = np.nan
                    odd_feat["occlusion_low_points_ratio_dyn"] = np.nan

                dyn30 = dyn & near30
                odd_feat["dyn_label_count_30m"] = int(np.sum(dyn30))
                if np.any(dyn30):
                    pts_dyn30 = pts[dyn30]
                    odd_feat["mean_lidar_points_in_box_dyn_30m"] = float(np.mean(pts_dyn30)) if pts_dyn30.size else np.nan
                    odd_feat["occlusion_low_points_ratio_dyn_30m"] = float(np.sum(pts_dyn30 <= 5) / pts_dyn30.size) if pts_dyn30.size else np.nan
                else:
                    odd_feat["mean_lidar_points_in_box_dyn_30m"] = np.nan
                    odd_feat["occlusion_low_points_ratio_dyn_30m"] = np.nan
            else:
                odd_feat["mean_lidar_points_in_box"] = np.nan
                odd_feat["occlusion_low_points_ratio"] = np.nan

                odd_feat["dyn_label_count"] = 0
                odd_feat["dyn_label_count_30m"] = 0
                odd_feat["mean_lidar_points_in_box_dyn"] = np.nan
                odd_feat["mean_lidar_points_in_box_dyn_30m"] = np.nan
                odd_feat["occlusion_low_points_ratio_dyn"] = np.nan
                odd_feat["occlusion_low_points_ratio_dyn_30m"] = np.nan
            # camera-label traffic light visibility (frame-level; state is NOT available in Perception raw)
            tl_count = 0
            try:
                for cam in getattr(frame, "camera_labels", []):
                    for lab2 in getattr(cam, "labels", []):
                        try:
                            tname = open_dataset.Label.Type.Name(int(getattr(lab2, "type", -1)))
                        except Exception:
                            tname = ""
                        if "TRAFFIC_LIGHT" in str(tname):
                            tl_count += 1
            except Exception:
                tl_count = 0
            odd_feat["traffic_light_visible_count"] = int(tl_count)
            odd_feat["has_traffic_light_visible"] = int(tl_count > 0)

            odd_feat["is_near_signalized_intersection"] = int(bool(odd_feat["has_traffic_light_visible"]) and bool(odd_feat["is_near_intersection"]))
            odd_feat["dist_to_signalized_intersection_lane_m"] = float(odd_feat["dist_to_intersection_lane_m"]) if odd_feat["has_traffic_light_visible"] and np.isfinite(odd_feat["dist_to_intersection_lane_m"]) else np.nan

            odd_rows.append(odd_feat)

            # objects (GLOBAL/MAP primary)
            for lab in frame.laser_labels:
                r = _extract_laser_label_row(lab, T_map, ego_yaw)
                r.update({
                    "segment_id": seg_id,
                    "frame_label": int(seg_frame_idx),
                    "timestamp_micros": ts,
                })
                obj_rows.append(r)

            seg_frame_idx += 1
            flush_if_too_big(seg_id)

    # final flush
    if seg_id is not None:
        if not static_map_written:
            write_missing_static_map_row(seg_id)
        flush_rows(seg_id)
        report["processed_segments"] += 1

    report["status"] = "success"
    return report


# -------------------------
# Manifest helpers
# -------------------------
def discover_tfrecords(wod_root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(wod_root, "**", "*.tfrecord*"), recursive=True))


def write_manifest(tfrecords: List[str], manifest_path: str):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"tfrecords": tfrecords}, f, ensure_ascii=False, indent=2)


def read_manifest(manifest_path: str) -> List[str]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    tfrecords = payload.get("tfrecords", [])
    if not isinstance(tfrecords, list):
        raise ValueError("manifest['tfrecords'] must be a list")
    return tfrecords


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wod_root", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--write_manifest", type=str, default=None)
    parser.add_argument("--out_root", type=str, required=False)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--radius", type=float, default=50.0)
    parser.add_argument("--lane_match_max_dist", type=float, default=5.0)
    parser.add_argument("--near_radius", type=float, default=20.0,
                        help="Distance threshold (m) for per-frame is_near_* infra flags.")
    parser.add_argument("--intersection_near_dist", type=float, default=15.0,
                        help="Distance threshold (m) for per-frame is_near_intersection (to interpolating lane points).")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--flush_every_rows", type=int, default=300_000)
    parser.add_argument("--report_path", type=str, default=None)
    args = parser.parse_args()

    if args.write_manifest:
        if not args.wod_root:
            raise ValueError("--wod_root is required for --write_manifest")
        tfrecords = discover_tfrecords(args.wod_root)
        write_manifest(tfrecords, args.write_manifest)
        print(f"[OK] Wrote manifest: {args.write_manifest} (files={len(tfrecords)})")
        return

    if not args.out_root:
        raise ValueError("--out_root is required")

    if args.manifest:
        tfrecords = read_manifest(args.manifest)
    else:
        if not args.wod_root:
            raise ValueError("Either --manifest or --wod_root must be provided")
        tfrecords = discover_tfrecords(args.wod_root)

    report = build_from_wod_v7(
        tfrecords=tfrecords,
        out_root=args.out_root,
        overwrite=args.overwrite,
        radius=args.radius,
        lane_match_max_dist=args.lane_match_max_dist,
        near_radius=args.near_radius,
        intersection_near_dist=args.intersection_near_dist,
        max_files=args.max_files,
        flush_every_rows=args.flush_every_rows,
    )

    print(f"[Finished] written under: {os.path.join(args.out_root, 'staging')}")
    print(f"  processed_files={report['processed_files']}, processed_segments={report['processed_segments']}, lane_nn_mode={report.get('lane_nn_mode')}")

    if args.report_path:
        os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
        with open(args.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[Report] {args.report_path}")


if __name__ == "__main__":
    main()

