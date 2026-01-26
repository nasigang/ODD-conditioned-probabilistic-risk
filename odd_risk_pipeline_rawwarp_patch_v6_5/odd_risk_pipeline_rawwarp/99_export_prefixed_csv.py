#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export Stage-07 shards (.pt) into a training-ready CSV.

This script bridges the Waymo-processing pipeline (00-07) and the training pipeline
(odd_risk_pipeline_rawwarp).

Input:
  - index.json written by Step 07 (contains x_schema, shard list)
  - odd_input_schema_v7.json written by Step 04 (contains c vector_order)

Output:
  - CSV with fixed, explicit columns:
      x__* : frame-level raw features
      c__* : ODD / segment condition vector entries

We stream rows to avoid large in-memory DataFrames.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, required=True, help="Step-07 index.json path")
    ap.add_argument("--odd_schema", type=str, required=True, help="Step-04 odd_input_schema_v7.json path")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--max_rows", type=int, default=0, help="0 means no limit")
    args = ap.parse_args()

    index_p = Path(args.index)
    odd_p = Path(args.odd_schema)
    out_p = Path(args.out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    index = _load_json(str(index_p))
    odd = _load_json(str(odd_p))

    x_schema: List[str] = index.get("x_schema", [])
    if not x_schema:
        raise RuntimeError(f"index.json missing x_schema: {args.index}")

    c_schema: List[str] = odd.get("vector_order", [])
    if not c_schema:
        raise RuntimeError(f"odd_input_schema missing vector_order: {args.odd_schema}")

    # Step-07 index.json may store shards either as:
    #   - list[str] (paths)
    #   - list[dict] with keys like {"path": <path>, "n": <count>}
    shards_raw = index.get("shards", [])
    if not shards_raw:
        raise RuntimeError(f"index.json missing shards: {args.index}")

    shard_paths: List[str] = []
    for ent in shards_raw:
        if isinstance(ent, str):
            shard_paths.append(ent)
        elif isinstance(ent, dict) and "path" in ent:
            shard_paths.append(str(ent["path"]))
        else:
            raise RuntimeError(f"Unsupported shard entry in index.json: {type(ent)} -> {ent}")

    # Build header
    meta_cols = ["segment_id", "frame_label", "y_soft", "label", "n_edges", "n_flagged"]
    x_cols = [f"x__{n}" for n in x_schema]
    c_cols = [f"c__{n}" for n in c_schema]
    header = meta_cols + x_cols + c_cols

    base_dir = index_p.parent
    n_written = 0

    with open(out_p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for rel in shard_paths:
            shard_path = base_dir / rel
            if not shard_path.exists():
                # allow absolute shard paths in index
                shard_path = Path(rel)
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing shard: {rel} (resolved {shard_path})")

            items = torch.load(shard_path, map_location="cpu")
            if not isinstance(items, list):
                raise RuntimeError(f"Shard is not a list: {shard_path}")

            for it in items:
                seg = it.get("segment_id", "")
                fl = int(it.get("frame_label", -1))
                y_soft = float(it.get("y_soft", 0.0))
                label = int(it.get("label", -1))
                n_edges = int(it.get("n_edges", 0))
                n_flagged = int(it.get("n_flagged", 0))

                x = it.get("x", [])
                c = it.get("c", [])

                if len(x) != len(x_schema):
                    raise RuntimeError(f"x length mismatch at {seg}/{fl}: {len(x)} vs {len(x_schema)}")
                if len(c) != len(c_schema):
                    raise RuntimeError(f"c length mismatch at {seg}/{fl}: {len(c)} vs {len(c_schema)}")

                row = [seg, fl, y_soft, label, n_edges, n_flagged]
                row.extend([float(v) for v in x])
                row.extend([float(v) for v in c])
                w.writerow(row)

                n_written += 1
                if args.max_rows and n_written >= args.max_rows:
                    print(f"[OK] reached max_rows={args.max_rows}")
                    print(f"Wrote: {out_p} rows={n_written}")
                    return

    print(f"[OK] Wrote: {out_p} rows={n_written}")


if __name__ == "__main__":
    main()
