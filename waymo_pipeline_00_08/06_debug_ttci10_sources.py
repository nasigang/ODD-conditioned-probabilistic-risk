#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08_debug_ttci10_sources.py

Goal:
- Find frames with TTCI≈10 in processed shards
- For those (segment_id, frame_label), open interaction_pairs_v7 and compute:
  - how many candidate edges are overlap_now vs hit_future
  - whether min TTC comes from overlap_now or hit_future
- Optionally apply the same filters as 07 (ego speed, rel speed, static-close exclusion)

Run example:
python scripts/08_debug_ttci10_sources.py --out_root "$OUT" --split_name train \
  --ego_only --distance_max 70 \
  --min_ego_speed_mps 3.0 --min_rel_speed_mps 0.5 \
  --exclude_static_close --static_close_range_m 3.0 --static_close_rel_speed_mps 0.5
"""

from __future__ import annotations

import os
import glob
import argparse
from typing import Dict, Set, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


def _load_ego_speed_map(base_frames_root: str, sid: str) -> Dict[int, float]:
    part_dir = os.path.join(base_frames_root, f"segment_id={sid}")
    files = sorted(glob.glob(os.path.join(part_dir, "*.parquet")))
    if not files:
        return {}
    dfs = []
    for fp in files:
        try:
            df = pd.read_parquet(fp, columns=["frame_label", "ego_vx", "ego_vy"])
        except Exception:
            df = pd.read_parquet(fp)
            if not {"frame_label", "ego_vx", "ego_vy"}.issubset(df.columns):
                return {}
            df = df[["frame_label", "ego_vx", "ego_vy"]]
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df["frame_label"] = pd.to_numeric(df["frame_label"], errors="coerce")
    df["ego_vx"] = pd.to_numeric(df["ego_vx"], errors="coerce")
    df["ego_vy"] = pd.to_numeric(df["ego_vy"], errors="coerce")
    df = df[df["frame_label"].notna()]
    if df.empty:
        return {}

    spd = np.sqrt(df["ego_vx"].values**2 + df["ego_vy"].values**2)
    df["ego_speed_mps"] = spd
    g = df.groupby(df["frame_label"].astype(int), sort=False)["ego_speed_mps"].max()
    return {int(k): float(v) for k, v in g.items()}


def _safe_read_parquet_columns(fp: str, cols: List[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(fp, columns=cols)
    except Exception:
        df = pd.read_parquet(fp)
        keep = [c for c in cols if c in df.columns]
        return df[keep].copy()


def _pick_collision_flag(cols: List[str]) -> Optional[str]:
    if "collision_course" in cols:
        return "collision_course"
    if "collision_pred" in cols:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--split_name", type=str, default="train")
    ap.add_argument("--processed_subdir", type=str, default="processed_v7")
    ap.add_argument("--pairs_subdir", type=str, default="interaction_pairs_v7")

    # frame selection: from shard ttci
    ap.add_argument("--ttci_ge", type=float, default=9.999, help="select frames where best_ttci >= this")
    ap.add_argument("--max_frames", type=int, default=0, help="0=all, else cap number of frames")

    # match 07 filters
    ap.add_argument("--ego_only", action="store_true")
    ap.add_argument("--distance_max", type=float, default=70.0)

    ap.add_argument("--min_ego_speed_mps", type=float, default=0.0)
    ap.add_argument("--min_rel_speed_mps", type=float, default=0.0)
    ap.add_argument("--exclude_static_close", action="store_true")
    ap.add_argument("--static_close_range_m", type=float, default=3.0)
    ap.add_argument("--static_close_rel_speed_mps", type=float, default=0.5)

    ap.add_argument("--report_csv", type=str, default="/workspace/out/waymo_wod_parquet/debug_ttci10_report.csv")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out_root)
    processed_dir = os.path.join(out_root, args.processed_subdir, args.split_name)
    pairs_root = os.path.join(out_root, "staging", args.pairs_subdir)
    base_frames_root = os.path.join(out_root, "staging", "base_frames")

    shard_files = sorted(glob.glob(os.path.join(processed_dir, "shard_*.pt")))
    if not shard_files:
        raise SystemExit(f"[ERROR] no shards under: {processed_dir}")

    # 1) Collect (segment_id -> set(frame_label)) where ttci>=threshold
    target: Dict[str, Set[int]] = {}
    n_items = 0
    n_hit = 0

    for sf in shard_files:
        data = torch.load(sf, weights_only=False)
        for it in data:
            n_items += 1
            ttci = float(it["x"][0])
            if ttci >= args.ttci_ge:
                sid = str(it["segment_id"])
                fl = int(it["frame_label"])
                target.setdefault(sid, set()).add(fl)
                n_hit += 1

    print(f"[Shard scan] total_samples={n_items:,}  ttci>={args.ttci_ge} frames(samples)={n_hit:,}  segments={len(target):,}")

    # cap frames if requested
    if args.max_frames and args.max_frames > 0:
        # deterministic cap: iterate segments in sorted order
        capped: Dict[str, Set[int]] = {}
        cnt = 0
        for sid in sorted(target.keys()):
            fls = sorted(target[sid])
            for fl in fls:
                capped.setdefault(sid, set()).add(fl)
                cnt += 1
                if cnt >= args.max_frames:
                    break
            if cnt >= args.max_frames:
                break
        target = capped
        print(f"[Cap] using first {args.max_frames} frames across segments -> segments={len(target)}")

    # 2) For each segment, read interaction_pairs and compute overlap/hit breakdown for those frames
    cols_need = [
        "segment_id", "frame_label",
        "src_is_ego",
        "range_m", "ttc_2d",
        "collision_course", "collision_pred",
        "overlap_now", "hit_future",
        "approaching",
        "src_speed_mps",
        "rel_speed_mps",
        "rel_vel_x", "rel_vel_y",
    ]

    rows = []
    seg_count = 0

    for sid in sorted(target.keys()):
        part_dir = os.path.join(pairs_root, f"segment_id={sid}")
        files = sorted(glob.glob(os.path.join(part_dir, "*.parquet")))
        if not files:
            continue

        # ego speed map (only if needed and src_speed_mps not available)
        ego_speed_map = {}
        if args.min_ego_speed_mps > 0 and os.path.exists(base_frames_root):
            ego_speed_map = _load_ego_speed_map(base_frames_root, sid)

        frames_set = target[sid]
        seg_count += 1

        for fp in files:
            df = _safe_read_parquet_columns(fp, cols_need)

            if "frame_label" not in df.columns or "ttc_2d" not in df.columns or "range_m" not in df.columns:
                continue

            df["frame_label"] = pd.to_numeric(df["frame_label"], errors="coerce").astype("Int64")
            df = df[df["frame_label"].notna()]
            if df.empty:
                continue

            # only target frames
            fl_int = df["frame_label"].astype(int)
            mask_frames = fl_int.isin(frames_set)
            df = df[mask_frames]
            if df.empty:
                continue

            # distance filter
            df["range_m"] = pd.to_numeric(df["range_m"], errors="coerce")
            df = df[np.isfinite(df["range_m"].values) & (df["range_m"].values <= float(args.distance_max))]
            if df.empty:
                continue

            # ego_only
            if args.ego_only:
                if "src_is_ego" not in df.columns:
                    raise SystemExit("[ERROR] --ego_only but src_is_ego missing in interaction_pairs")
                df["src_is_ego"] = pd.to_numeric(df["src_is_ego"], errors="coerce").fillna(0).astype(int)
                df = df[df["src_is_ego"] == 1]
                if df.empty:
                    continue

            # determine collision flag
            coll_flag = _pick_collision_flag(list(df.columns))
            has_approach = "approaching" in df.columns
            if coll_flag is not None:
                df[coll_flag] = pd.to_numeric(df[coll_flag], errors="coerce").fillna(0).astype(int)
            if has_approach:
                df["approaching"] = pd.to_numeric(df["approaching"], errors="coerce").fillna(0).astype(int)

            # candidates: prefer collision edges
            if coll_flag is not None:
                df_cand = df[df[coll_flag] == 1].copy()
            elif has_approach:
                df_cand = df[df["approaching"] == 1].copy()
            else:
                df_cand = df.copy()

            if df_cand.empty:
                continue

            # ego speed filter
            if args.min_ego_speed_mps > 0:
                if "src_speed_mps" in df_cand.columns:
                    sspd = pd.to_numeric(df_cand["src_speed_mps"], errors="coerce")
                    df_cand = df_cand[sspd.isna() | (sspd.values >= float(args.min_ego_speed_mps))]
                elif ego_speed_map:
                    spd_series = df_cand["frame_label"].astype(int).map(ego_speed_map)
                    df_cand = df_cand[spd_series.isna() | (spd_series.values >= float(args.min_ego_speed_mps))]
                if df_cand.empty:
                    continue

            # rel speed filter + static-close exclusion
            rel_speed, ok_rel = _get_rel_speed(df_cand)
            if (args.min_rel_speed_mps > 0 or args.exclude_static_close) and ok_rel:
                if args.min_rel_speed_mps > 0:
                    mask = np.isfinite(rel_speed) & (rel_speed >= float(args.min_rel_speed_mps))
                    df_cand = df_cand[mask].copy()
                    rel_speed = rel_speed[mask]
                    if df_cand.empty:
                        continue
                if args.exclude_static_close:
                    rng = df_cand["range_m"].values
                    mask_sc = (rng < float(args.static_close_range_m)) & (rel_speed < float(args.static_close_rel_speed_mps))
                    df_cand = df_cand[~mask_sc].copy()
                    if df_cand.empty:
                        continue

            # ensure numeric ttc
            df_cand["ttc_2d"] = pd.to_numeric(df_cand["ttc_2d"], errors="coerce")
            df_cand = df_cand[np.isfinite(df_cand["ttc_2d"].values)]
            if df_cand.empty:
                continue

            # overlap/hit columns (if present)
            if "overlap_now" in df_cand.columns:
                df_cand["overlap_now"] = pd.to_numeric(df_cand["overlap_now"], errors="coerce").fillna(0).astype(int)
            else:
                df_cand["overlap_now"] = 0
            if "hit_future" in df_cand.columns:
                df_cand["hit_future"] = pd.to_numeric(df_cand["hit_future"], errors="coerce").fillna(0).astype(int)
            else:
                df_cand["hit_future"] = 0

            # per-frame summarize
            for fl, g in df_cand.groupby(df_cand["frame_label"].astype(int), sort=False):
                min_ttc = float(g["ttc_2d"].min())
                # min-edge source
                gmin = g[g["ttc_2d"].values == min_ttc]
                min_from_overlap = int((gmin["overlap_now"].values == 1).any())
                min_from_hit = int((gmin["hit_future"].values == 1).any())

                rows.append({
                    "segment_id": sid,
                    "frame_label": int(fl),
                    "n_cand_edges": int(len(g)),
                    "min_ttc": float(min_ttc),
                    "n_overlap_now": int((g["overlap_now"].values == 1).sum()),
                    "n_hit_future": int((g["hit_future"].values == 1).sum()),
                    "min_from_overlap": int(min_from_overlap),
                    "min_from_hit_future": int(min_from_hit),
                })

    if not rows:
        print("[Result] No matching frames found (after filters).")
        return

    rep = pd.DataFrame(rows)
    rep.to_csv(args.report_csv, index=False)
    print(f"[Saved] {args.report_csv}  rows={len(rep):,}")

    # global summary
    total_frames = rep.shape[0]
    n_min_overlap = int(rep["min_from_overlap"].sum())
    n_min_hit = int(rep["min_from_hit_future"].sum())
    print("\n=== Summary for TTCI≈10 frames (filtered candidates) ===")
    print(f"frames_analyzed: {total_frames:,}")
    print(f"min_from_overlap_now: {n_min_overlap:,} ({n_min_overlap/total_frames:.3f})")
    print(f"min_from_hit_future: {n_min_hit:,} ({n_min_hit/total_frames:.3f})")

    print("\n=== Overlap/Hit edge counts (sum across analyzed frames) ===")
    print(f"sum_overlap_now_edges: {int(rep['n_overlap_now'].sum()):,}")
    print(f"sum_hit_future_edges: {int(rep['n_hit_future'].sum()):,}")

    # top frames by overlap edges
    top = rep.sort_values(["n_overlap_now", "n_cand_edges"], ascending=False).head(20)
    print("\n=== Top-20 frames by overlap_now edges ===")
    print(top[["segment_id", "frame_label", "n_cand_edges", "n_overlap_now", "n_hit_future", "min_ttc"]].to_string(index=False))


if __name__ == "__main__":
    main()

