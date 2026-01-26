#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mine False Positives from preds_*.npz produced by diagnose_tau_gate_flow_v6_5.py --save_preds.

Output CSV columns:
  segment_id, frame_label, risk, p_gate, p_event, y_event, y_gate, rank

Typical use:
  python 10_mine_false_positives.py \
    --preds_npz $OUT/diag/run_flow_dropc/preds_val_ttc_sstar_tau0.5.npz \
    --out_csv   $OUT/fp_reports/fp_top50_tau0.5.csv \
    --topk 50 \
    --min_p_gate 0.15 \
    --per_segment_k 3 \
    --min_frame_gap 5
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_npz", required=True)
    ap.add_argument("--out_csv", required=True)

    ap.add_argument("--topk", type=int, default=50, help="Total FP cases to export")
    ap.add_argument("--min_risk", type=float, default=None, help="Optional hard threshold for risk")
    ap.add_argument("--min_p_gate", type=float, default=0.0, help="Filter: p_gate >= this (e.g., 0.15)")
    ap.add_argument("--per_segment_k", type=int, default=3, help="Max selected frames per segment (0=unlimited)")
    ap.add_argument("--min_frame_gap", type=int, default=5, help="Avoid near-duplicate frames within segment")

    args = ap.parse_args()

    npz = np.load(args.preds_npz, allow_pickle=True)

    def _must(k):
        if k not in npz:
            raise SystemExit(f"[ERROR] missing key in npz: {k}. Available={list(npz.keys())}")
        return npz[k]

    segment_id = _must("segment_id").astype(str)
    frame_label = _must("frame_label").astype(np.int64)

    y_gate  = _must("y_gate").astype(np.int64) if "y_gate" in npz else None
    y_event = _must("y_event").astype(np.int64)
    p_gate  = _must("p_gate").astype(np.float32)
    p_event = _must("p_event").astype(np.float32)
    risk    = _must("risk").astype(np.float32)

    df = pd.DataFrame({
        "segment_id": segment_id,
        "frame_label": frame_label,
        "y_event": y_event,
        "p_gate": p_gate,
        "p_event": p_event,
        "risk": risk,
    })
    if y_gate is not None:
        df["y_gate"] = y_gate
    else:
        df["y_gate"] = -1

    # FP condition: label=0 but score high
    m = (df["y_event"] == 0) & (df["p_gate"] >= float(args.min_p_gate))
    if args.min_risk is not None:
        m = m & (df["risk"] >= float(args.min_risk))

    cand = df[m].copy()
    if cand.empty:
        raise SystemExit("[ERROR] No FP candidates after filtering. Try lowering min_p_gate/min_risk or check labels.")

    cand = cand.sort_values(["risk"], ascending=False).reset_index(drop=True)

    # Diversify: cap per segment and avoid near-duplicate frames
    picked = []
    per_seg_cnt = {}
    last_frame = {}

    for _, row in cand.iterrows():
        sid = row["segment_id"]
        fl = int(row["frame_label"])

        if args.per_segment_k > 0:
            if per_seg_cnt.get(sid, 0) >= args.per_segment_k:
                continue

        if sid in last_frame and abs(fl - last_frame[sid]) < int(args.min_frame_gap):
            continue

        picked.append(row)
        per_seg_cnt[sid] = per_seg_cnt.get(sid, 0) + 1
        last_frame[sid] = fl

        if len(picked) >= int(args.topk):
            break

    out = pd.DataFrame(picked)
    if out.empty:
        raise SystemExit("[ERROR] Selection became empty after per-segment/gap filtering. Relax constraints.")

    out.insert(0, "rank", np.arange(1, len(out) + 1, dtype=np.int64))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"[OK] wrote FP report: {out_path}  (n={len(out)})")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()

