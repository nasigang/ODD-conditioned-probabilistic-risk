#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract still PNG frames from videos produced by 09_render_risk_videos.py
for the exact (segment_id, frame_label) rows in report_csv.

Assumption:
  09_render_risk_videos.py names files:
    <video_dir>/<segment_id>_fl{f0:04d}-{f1:04d}.avi
  where f0 = min(selected_frames) - pad_frames (clipped at 0)
        f1 = max(selected_frames) + pad_frames

Run:
  python 10_extract_frames_from_risk_videos.py \
    --report_csv $OUT/fp_reports/fp_top50_val_tau0.5_pg0.15.csv \
    --video_dir  $OUT/risk_videos \
    --out_dir    $OUT/risk_frames_fp \
    --pad_frames 30
"""

import argparse
from pathlib import Path
import pandas as pd

try:
    import cv2
except ImportError as e:
    raise SystemExit("cv2 (opencv-python) is required. pip install opencv-python") from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_csv", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pad_frames", type=int, default=30)
    ap.add_argument("--max_per_segment", type=int, default=20, help="limit saved PNGs per segment (0=unlimited)")
    args = ap.parse_args()

    df = pd.read_csv(args.report_csv)
    for c in ["segment_id", "frame_label"]:
        if c not in df.columns:
            raise SystemExit(f"[ERROR] report_csv missing column: {c}")

    df["segment_id"] = df["segment_id"].astype(str).str.strip()
    df["frame_label"] = pd.to_numeric(df["frame_label"], errors="coerce")
    df = df[df["frame_label"].notna()]
    df["frame_label"] = df["frame_label"].astype(int)

    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for sid, g in df.groupby("segment_id", sort=False):
        fls = sorted(g["frame_label"].tolist())
        if not fls:
            continue

        f0 = max(min(fls) - int(args.pad_frames), 0)
        f1 = max(fls) + int(args.pad_frames)

        vid = video_dir / f"{sid}_fl{f0:04d}-{f1:04d}.avi"
        if not vid.exists():
            print(f"[SKIP] video not found: {vid}")
            continue

        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f"[SKIP] cannot open video: {vid}")
            continue

        seg_out = out_dir / sid
        seg_out.mkdir(parents=True, exist_ok=True)

        saved = 0
        for fl in fls:
            idx = int(fl - f0)
            if idx < 0:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                print(f"[WARN] read failed: {sid} frame_label={fl} (idx={idx})")
                continue

            out_png = seg_out / f"{sid}_frame{fl:04d}.png"
            cv2.imwrite(str(out_png), frame)
            saved += 1
            total += 1

            if args.max_per_segment > 0 and saved >= int(args.max_per_segment):
                break

        cap.release()
        print(f"[OK] {sid}: saved {saved} pngs  (video={vid.name})")

    print(f"[DONE] total png saved: {total} -> {out_dir}")


if __name__ == "__main__":
    main()

