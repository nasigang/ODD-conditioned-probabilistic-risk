#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
09_render_risk_videos.py

Create top-view videos for "risky" frames listed in debug_ttci10_report.csv.
Data sources (from your pipeline):
  - <OUT>/staging/base_frames/segment_id=.../*.parquet
  - <OUT>/staging/base_objects/segment_id=.../*.parquet
  - <OUT>/staging/interaction_pairs_v7/segment_id=.../*.parquet

The video shows ego + object bounding boxes (no map lanes).
Highlights the object that yields min TTC among candidate edges (src_is_ego==1) at each frame.

Requirements:
  - pandas, numpy, matplotlib
  - opencv-python (cv2) for AVI writing (MJPG)

Run:
python scripts/09_render_risk_videos.py --out_root "$OUT" \
  --report_csv debug_ttci10_report.csv \
  --only_ttci10 \
  --max_segments 20 \
  --pad_frames 30 \
  --fps 10
"""

from __future__ import annotations

import os
import glob
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

try:
    import cv2
except ImportError as e:
    raise SystemExit("cv2 (opencv-python) is required for video writing. Please install opencv-python.") from e


# -----------------------------
# IO helpers
# -----------------------------
def _list_parquets(part_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(part_dir, "*.parquet")))

def _read_parquet_partition_filtered(
    part_dir: str,
    columns: List[str],
    frame_min: Optional[int] = None,
    frame_max: Optional[int] = None,
) -> pd.DataFrame:
    files = _list_parquets(part_dir)
    if not files:
        return pd.DataFrame()

    dfs = []
    for fp in files:
        try:
            df = pd.read_parquet(fp, columns=columns)
        except Exception:
            df = pd.read_parquet(fp)
            keep = [c for c in columns if c in df.columns]
            df = df[keep].copy()

        if "frame_label" in df.columns:
            df["frame_label"] = pd.to_numeric(df["frame_label"], errors="coerce").astype("Int64")
            df = df[df["frame_label"].notna()]
            if frame_min is not None:
                df = df[df["frame_label"].astype(int) >= int(frame_min)]
            if frame_max is not None:
                df = df[df["frame_label"].astype(int) <= int(frame_max)]
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    return out


def _pick_obj_id_col(df: pd.DataFrame) -> str:
    for c in ["obj_id", "track_id", "id"]:
        if c in df.columns:
            return c
    raise RuntimeError("base_objects must contain one of: obj_id / track_id / id")


# -----------------------------
# geometry
# -----------------------------
def _rot2d(x: np.ndarray, y: np.ndarray, yaw: float) -> Tuple[np.ndarray, np.ndarray]:
    # rotate by yaw (rad)
    c, s = np.cos(yaw), np.sin(yaw)
    xr = c * x - s * y
    yr = s * x + c * y
    return xr, yr

def _to_ego_frame(xw: float, yw: float, ego_x: float, ego_y: float, ego_yaw: float) -> Tuple[float, float]:
    # translate then rotate by -ego_yaw (so ego faces +x)
    dx = xw - ego_x
    dy = yw - ego_y
    xr, yr = _rot2d(np.array([dx]), np.array([dy]), -ego_yaw)
    return float(xr[0]), float(yr[0])

def _rect_corners(cx: float, cy: float, length: float, width: float, yaw: float) -> np.ndarray:
    # rectangle centered at (cx,cy) with heading yaw
    # local corners
    hl = 0.5 * float(length)
    hw = 0.5 * float(width)
    pts = np.array([
        [ hl,  hw],
        [ hl, -hw],
        [-hl, -hw],
        [-hl,  hw],
    ], dtype=np.float32)
    # rotate then translate
    xr, yr = _rot2d(pts[:, 0], pts[:, 1], yaw)
    return np.stack([xr + cx, yr + cy], axis=1)


# -----------------------------
# per-segment video rendering
# -----------------------------
def _build_min_ttc_target_map(
    pairs_df: pd.DataFrame,
    *,
    ego_only: bool = True,
) -> Dict[int, Dict[str, object]]:
    """
    Returns:
      frame_label -> {"dst_id": str, "min_ttc": float, "overlap_now": int, "hit_future": int}
    """
    if pairs_df.empty:
        return {}

    # expected columns (from your 04 patched)
    # frame_label, src_is_ego, dst_id, ttc_2d, overlap_now, hit_future
    for c in ["frame_label", "ttc_2d"]:
        if c not in pairs_df.columns:
            raise RuntimeError(f"interaction_pairs missing required column: {c}")

    if ego_only and "src_is_ego" in pairs_df.columns:
        pairs_df["src_is_ego"] = pd.to_numeric(pairs_df["src_is_ego"], errors="coerce").fillna(0).astype(int)
        pairs_df = pairs_df[pairs_df["src_is_ego"] == 1]

    # ensure numeric
    pairs_df["frame_label"] = pd.to_numeric(pairs_df["frame_label"], errors="coerce").astype("Int64")
    pairs_df = pairs_df[pairs_df["frame_label"].notna()]
    pairs_df["ttc_2d"] = pd.to_numeric(pairs_df["ttc_2d"], errors="coerce")

    pairs_df = pairs_df[np.isfinite(pairs_df["ttc_2d"].values)]
    if pairs_df.empty:
        return {}

    if "dst_id" not in pairs_df.columns:
        # older schema fallback
        if "dst_track_id" in pairs_df.columns:
            pairs_df = pairs_df.rename(columns={"dst_track_id": "dst_id"})
        else:
            # if not available, we can't highlight a target
            pairs_df["dst_id"] = "UNKNOWN"

    if "overlap_now" not in pairs_df.columns:
        pairs_df["overlap_now"] = 0
    if "hit_future" not in pairs_df.columns:
        pairs_df["hit_future"] = 0

    pairs_df["overlap_now"] = pd.to_numeric(pairs_df["overlap_now"], errors="coerce").fillna(0).astype(int)
    pairs_df["hit_future"] = pd.to_numeric(pairs_df["hit_future"], errors="coerce").fillna(0).astype(int)

    out: Dict[int, Dict[str, object]] = {}
    for fl, g in pairs_df.groupby(pairs_df["frame_label"].astype(int), sort=False):
        idx = g["ttc_2d"].values.argmin()
        row = g.iloc[int(idx)]
        out[int(fl)] = {
            "dst_id": str(row["dst_id"]),
            "min_ttc": float(row["ttc_2d"]),
            "overlap_now": int(row["overlap_now"]),
            "hit_future": int(row["hit_future"]),
        }
    return out


def render_segment_video(
    out_root: str,
    segment_id: str,
    frame_min: int,
    frame_max: int,
    *,
    out_dir: str,
    fps: int = 10,
    view_radius_m: float = 60.0,
    ego_centric: bool = True,
    pad_title: bool = True,
    ego_only_pairs: bool = True,
    video_w: int = 900,
    video_h: int = 900,
) -> str:
    staging = os.path.join(out_root, "staging")
    base_frames_dir = os.path.join(staging, "base_frames", f"segment_id={segment_id}")
    base_objects_dir = os.path.join(staging, "base_objects", f"segment_id={segment_id}")
    pairs_dir = os.path.join(staging, "interaction_pairs_v7", f"segment_id={segment_id}")

    if not os.path.exists(base_frames_dir):
        raise RuntimeError(f"Missing base_frames partition: {base_frames_dir}")
    if not os.path.exists(base_objects_dir):
        raise RuntimeError(f"Missing base_objects partition: {base_objects_dir}")
    if not os.path.exists(pairs_dir):
        raise RuntimeError(f"Missing interaction_pairs_v7 partition: {pairs_dir}")

    # load frames
    frm_cols = ["frame_label", "ego_x", "ego_y", "ego_vx", "ego_vy", "ego_yaw", "timestamp_micros"]
    frames = _read_parquet_partition_filtered(base_frames_dir, frm_cols, frame_min, frame_max)
    if frames.empty:
        raise RuntimeError(f"No base_frames rows for {segment_id} in range [{frame_min},{frame_max}]")
    frames["frame_label"] = pd.to_numeric(frames["frame_label"], errors="coerce").astype("Int64")
    frames = frames[frames["frame_label"].notna()]
    frames["frame_label_int"] = frames["frame_label"].astype(int)
    frames = frames.sort_values("frame_label_int")

    # if ego_yaw missing, derive from velocity (fallback)
    if "ego_yaw" not in frames.columns:
        frames["ego_yaw"] = np.arctan2(pd.to_numeric(frames["ego_vy"], errors="coerce").fillna(0.0).values,
                                       pd.to_numeric(frames["ego_vx"], errors="coerce").fillna(0.0).values)

    # load objects
    obj_cols = ["frame_label", "x", "y", "length", "width", "heading", "speed_x", "speed_y", "obj_id", "track_id", "id"]
    objs = _read_parquet_partition_filtered(base_objects_dir, obj_cols, frame_min, frame_max)
    if objs.empty:
        raise RuntimeError(f"No base_objects rows for {segment_id} in range [{frame_min},{frame_max}]")

    oid_col = _pick_obj_id_col(objs)

    # sanitize numeric
    for c in ["x", "y", "length", "width", "heading"]:
        if c in objs.columns:
            objs[c] = pd.to_numeric(objs[c], errors="coerce")
    objs["frame_label"] = pd.to_numeric(objs["frame_label"], errors="coerce").astype("Int64")
    objs = objs[objs["frame_label"].notna()]
    objs["frame_label_int"] = objs["frame_label"].astype(int)
    objs[oid_col] = objs[oid_col].astype(str)

    # set default size if missing
    if "length" not in objs.columns:
        objs["length"] = 4.5
    if "width" not in objs.columns:
        objs["width"] = 1.8
    if "heading" not in objs.columns:
        objs["heading"] = 0.0

    # load pairs (only needed cols)
    pair_cols = ["frame_label", "src_is_ego", "dst_id", "ttc_2d", "overlap_now", "hit_future"]
    pairs = _read_parquet_partition_filtered(pairs_dir, pair_cols, frame_min, frame_max)
    target_map = _build_min_ttc_target_map(pairs, ego_only=ego_only_pairs)

    # video setup
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{segment_id}_fl{frame_min:04d}-{frame_max:04d}.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (video_w, video_h))
    if not vw.isOpened():
        raise RuntimeError("Failed to open VideoWriter. Try another codec/fourcc.")

    # matplotlib figure (offscreen)
    dpi = 100
    fig_w = video_w / dpi
    fig_h = video_h / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    # pre-group objects by frame for speed
    obj_groups = {k: g for k, g in objs.groupby("frame_label_int", sort=False)}

    for _, fr in frames.iterrows():
        fl = int(fr["frame_label_int"])
        ego_x = float(fr.get("ego_x", 0.0))
        ego_y = float(fr.get("ego_y", 0.0))
        ego_yaw = float(fr.get("ego_yaw", 0.0))
        ts = int(fr.get("timestamp_micros", 0))

        ax.clear()
        ax.set_aspect("equal", adjustable="box")

        # bounds
        ax.set_xlim(-view_radius_m, view_radius_m)
        ax.set_ylim(-view_radius_m, view_radius_m)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, linewidth=0.3, alpha=0.3)

        # ego box at origin if ego-centric
        if ego_centric:
            ego_cx, ego_cy = 0.0, 0.0
            ego_h = 0.0  # heading aligned to +x
        else:
            ego_cx, ego_cy = ego_x, ego_y
            ego_h = ego_yaw

        ego_poly = _rect_corners(ego_cx, ego_cy, length=4.8, width=2.0, yaw=ego_h)
        ax.add_patch(Polygon(ego_poly, closed=True, fill=False, linewidth=2.5))
        ax.text(ego_cx, ego_cy, "EGO", fontsize=10, ha="center", va="center")

        # target info for this frame
        tinfo = target_map.get(fl, None)
        target_id = str(tinfo["dst_id"]) if tinfo else None
        min_ttc = float(tinfo["min_ttc"]) if tinfo else np.inf
        is_overlap = int(tinfo["overlap_now"]) if tinfo else 0
        is_hit = int(tinfo["hit_future"]) if tinfo else 0

        # draw objects
        g = obj_groups.get(fl, None)
        if g is not None and not g.empty:
            for _, ob in g.iterrows():
                ox = float(ob.get("x", np.nan))
                oy = float(ob.get("y", np.nan))
                if not np.isfinite(ox) or not np.isfinite(oy):
                    continue
                L = float(ob.get("length", 4.5)) if np.isfinite(ob.get("length", 4.5)) else 4.5
                W = float(ob.get("width", 1.8)) if np.isfinite(ob.get("width", 1.8)) else 1.8
                hd = float(ob.get("heading", 0.0)) if np.isfinite(ob.get("heading", 0.0)) else 0.0
                oid = str(ob.get(oid_col, "UNK"))

                if ego_centric:
                    cx, cy = _to_ego_frame(ox, oy, ego_x, ego_y, ego_yaw)
                    yaw = hd - ego_yaw
                else:
                    cx, cy = ox, oy
                    yaw = hd

                # style
                lw = 1.2
                color = "0.4"  # gray
                label = None

                if target_id is not None and oid == target_id:
                    # highlight target
                    lw = 2.8
                    if is_overlap:
                        color = "m"   # magenta (CONTACT)
                        label = "TARGET (CONTACT)"
                    elif is_hit:
                        color = "y"   # yellow (HIT FUTURE)
                        label = "TARGET (HIT)"
                    else:
                        color = "y"
                        label = "TARGET"

                poly = _rect_corners(cx, cy, L, W, yaw)
                ax.add_patch(Polygon(poly, closed=True, fill=False, linewidth=lw, edgecolor=color))

                if label:
                    ax.text(cx, cy, label, fontsize=7, ha="center", va="center")

        # overlay text
        if pad_title:
            title = f"{segment_id} | frame={fl} | ts={ts}"
            if np.isfinite(min_ttc):
                title += f" | min_ttc={min_ttc:.3f}s"
                if is_overlap:
                    title += " | OVERLAP_NOW"
                elif is_hit:
                    title += " | HIT_FUTURE"
            ax.set_title(title, fontsize=10)

        # render to image buffer
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        vw.write(img_bgr)

    vw.release()
    plt.close(fig)
    return out_path


# -----------------------------
# main: choose segments/frames from report
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--report_csv", type=str, required=True)

    ap.add_argument("--out_dir", type=str, default="", help="default: <OUT>/risk_videos")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--view_radius_m", type=float, default=60.0)
    ap.add_argument("--pad_frames", type=int, default=30)
    ap.add_argument("--ego_centric", action="store_true", help="default True", default=True)

    ap.add_argument("--max_segments", type=int, default=20)
    ap.add_argument("--max_videos", type=int, default=0, help="0=all selected segments")
    ap.add_argument("--only_ttci10", action="store_true", help="(kept) not used; report already filtered upstream")
    ap.add_argument("--only_overlap", action="store_true")
    ap.add_argument("--only_hit", action="store_true")

    # ✅ NEW: render only specified segment(s)
    ap.add_argument("--segment_id", action="append", default=[],
                    help="Render only this segment_id. Repeatable. Example: --segment_id <id> --segment_id <id2>")

    # ✅ NEW: sort strategy for segments when using max_segments
    ap.add_argument("--sort_segments", type=str, default="first_seen",
                    choices=["first_seen", "most_frames", "worst_ttc"],
                    help="How to pick first max_segments segments.")

    args = ap.parse_args()

    out_root = os.path.abspath(args.out_root)
    report = pd.read_csv(args.report_csv)

    for c in ["segment_id", "frame_label"]:
        if c not in report.columns:
            raise SystemExit(f"[ERROR] report_csv missing column: {c}")

    # sanitize
    report["segment_id"] = report["segment_id"].astype(str).str.strip()
    report["frame_label"] = pd.to_numeric(report["frame_label"], errors="coerce")
    report = report[report["frame_label"].notna()]
    report["frame_label"] = report["frame_label"].astype(int)

    # optional filters
    if args.only_overlap and "min_from_overlap" in report.columns:
        report = report[report["min_from_overlap"] == 1]
    if args.only_hit and "min_from_hit_future" in report.columns:
        report = report[report["min_from_hit_future"] == 1]

    # ✅ if specific segment_id requested
    if args.segment_id:
        want = set([s.strip() for s in args.segment_id if s.strip()])
        report = report[report["segment_id"].isin(want)]
        if report.empty:
            raise SystemExit(f"[ERROR] no rows for requested --segment_id in report_csv: {sorted(list(want))}")

    # group by segment
    segs = []
    for sid, g in report.groupby("segment_id", sort=False):
        fls = sorted(g["frame_label"].tolist())
        # metric for sorting
        n_frames = len(fls)
        worst_ttc = float(pd.to_numeric(g.get("min_ttc", pd.Series([np.inf]*len(g))), errors="coerce").min()) if "min_ttc" in g.columns else np.inf
        segs.append((sid, fls, n_frames, worst_ttc))

    if not segs:
        raise SystemExit("[ERROR] no segments selected from report (after filters).")

    # sort segments if not explicitly requested
    if not args.segment_id:
        if args.sort_segments == "most_frames":
            segs.sort(key=lambda x: (-x[2], x[0]))  # many risky frames first
        elif args.sort_segments == "worst_ttc":
            segs.sort(key=lambda x: (x[3], -x[2], x[0]))  # smallest min_ttc first
        else:
            # first_seen: keep order
            pass

        # cap segments
        if args.max_segments and args.max_segments > 0:
            segs = segs[:args.max_segments]

    out_dir = args.out_dir.strip() or os.path.join(out_root, "risk_videos")
    os.makedirs(out_dir, exist_ok=True)

    made = 0
    for sid, fls, n_frames, worst_ttc in segs:
        f0 = max(min(fls) - int(args.pad_frames), 0)
        f1 = max(fls) + int(args.pad_frames)
        print(f"[Render] {sid}  risky_frames={min(fls)}..{max(fls)}(n={n_frames})  -> video range {f0}..{f1}")

        try:
            out_path = render_segment_video(
                out_root, sid, f0, f1,
                out_dir=out_dir,
                fps=args.fps,
                view_radius_m=args.view_radius_m,
                ego_centric=True,
            )
            print(f"  -> saved: {out_path}")
            made += 1
        except Exception as e:
            print(f"  [SKIP] {sid} failed: {e}")

        if args.max_videos and args.max_videos > 0 and made >= args.max_videos:
            break

    print(f"[Done] videos created: {made}  output_dir={out_dir}")


if __name__ == "__main__":
    main()

