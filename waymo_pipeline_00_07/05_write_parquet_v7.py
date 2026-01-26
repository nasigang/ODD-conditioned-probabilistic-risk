#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_write_parquet_v7_fixed.py
- 04_v7 산출물을 "학습용 데이터셋 패키지"로 정리한다.
- 핵심: 전체를 concat/merge 해서 단일 파일로 만들지 않는다(데이터 폭발 방지).
- 대신:
  1) manifest.json에 경로/스키마/컬럼 기록
  2) split manifest(train/val/test) 생성(세그먼트 기반)
  3) 필요 시 preview 샘플만 병합 저장

입력(기본):
  {out_root}/staging/interaction_pairs_v7/  (parquet dataset, segment_id 파티션)
  {out_root}/staging/segment_metadata.parquet
  {out_root}/staging/segment_condition_vectors_v7.parquet (있으면)
  {out_root}/staging/odd_input_schema_v7.json (있으면)

출력:
  {out_root}/dataset_v7/manifest.json
  {out_root}/dataset_v7/splits/train.json, val.json, test.json
  {out_root}/dataset_v7/preview.parquet (옵션)
"""

import os
import glob
import json
import argparse
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd


def _hash01(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    # 0..1
    return int(h[:8], 16) / float(0xFFFFFFFF)


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _find_first_parquet(dataset_root: str) -> Optional[str]:
    files = sorted(glob.glob(os.path.join(dataset_root, "**", "*.parquet"), recursive=True))
    return files[0] if files else None


def _seg_from_path(p: str) -> Optional[str]:
    # .../segment_id=XXXX/part-....parquet
    parts = p.replace("\\", "/").split("/")
    for x in parts:
        if x.startswith("segment_id="):
            return x.split("=", 1)[-1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True, help="프로젝트 out_root (staging 상위)")
    ap.add_argument("--staging_name", type=str, default="staging")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--preview_rows", type=int, default=0, help="0이면 preview 미생성, >0이면 샘플 병합 저장")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out_root)
    staging = os.path.join(out_root, args.staging_name)

    pairs_root = os.path.join(staging, "interaction_pairs_v7")
    meta_path = os.path.join(staging, "segment_metadata.parquet")
    cond_path = os.path.join(staging, "segment_condition_vectors_v7.parquet")
    schema_path = os.path.join(staging, "odd_input_schema_v7.json")

    if not os.path.exists(pairs_root):
        raise SystemExit(f"[ERROR] not found: {pairs_root}")
    if not os.path.exists(meta_path):
        raise SystemExit(f"[ERROR] not found: {meta_path}")

    ds_root = os.path.join(out_root, "dataset_v7")
    splits_dir = os.path.join(ds_root, "splits")
    _ensure_dir(ds_root)
    _ensure_dir(splits_dir)

    # segment list: from partition folders
    seg_dirs = sorted(glob.glob(os.path.join(pairs_root, "segment_id=*")))
    seg_ids = [os.path.basename(d).split("=", 1)[-1] for d in seg_dirs]
    if not seg_ids:
        raise SystemExit(f"[ERROR] no segment partitions under: {pairs_root}")

    # split (deterministic)
    tr, va, te = [], [], []
    for sid in seg_ids:
        r = _hash01(sid)
        if r < args.train_ratio:
            tr.append(sid)
        elif r < args.train_ratio + args.val_ratio:
            va.append(sid)
        else:
            te.append(sid)

    with open(os.path.join(splits_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump({"split": "train", "segment_ids": tr}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(splits_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump({"split": "val", "segment_ids": va}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(splits_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump({"split": "test", "segment_ids": te}, f, ensure_ascii=False, indent=2)

    # schema/columns snapshot
    sample_file = _find_first_parquet(pairs_root)
    if sample_file is None:
        raise SystemExit("[ERROR] no parquet parts found in interaction_pairs_v7")

    df0 = pd.read_parquet(sample_file)
    # segment_id 복원(파티션 컬럼이 파일 내부에 없을 수 있음)
    if "segment_id" not in df0.columns:
        sid0 = _seg_from_path(sample_file)
        if sid0 is not None:
            df0["segment_id"] = sid0

    manifest: Dict[str, Any] = {
        "version": "v7",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "paths": {
            "interaction_pairs_v7": pairs_root,
            "segment_metadata": meta_path,
            "segment_condition_vectors": cond_path if os.path.exists(cond_path) else None,
            "odd_input_schema": schema_path if os.path.exists(schema_path) else None,
        },
        "splits": {
            "train": os.path.join(splits_dir, "train.json"),
            "val": os.path.join(splits_dir, "val.json"),
            "test": os.path.join(splits_dir, "test.json"),
        },
        "counts": {
            "n_segments": len(seg_ids),
            "n_train": len(tr),
            "n_val": len(va),
            "n_test": len(te),
        },
        "pair_columns_sample": [{"name": c, "dtype": str(df0[c].dtype)} for c in df0.columns],
        "notes": [
            "학습용 full 데이터는 concat/merge 단일 파일로 만들지 말 것(메모리/용량 폭발).",
            "07_v7는 interaction_pairs_v7를 스트리밍 처리하면서 프레임 단위 샘플로 집계한다.",
        ],
    }

    with open(os.path.join(ds_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[OK] wrote:")
    print(f" - {os.path.join(ds_root, 'manifest.json')}")
    print(f" - {splits_dir}/*.json")
    print(f" - segments: {len(seg_ids)} (train={len(tr)}, val={len(va)}, test={len(te)})")

    # preview (optional): small sample merge for sanity check
    if args.preview_rows and args.preview_rows > 0:
        meta_df = pd.read_parquet(meta_path)
        # 일부 파티션 파일에서만 샘플링
        some_files = sorted(glob.glob(os.path.join(pairs_root, "**", "*.parquet"), recursive=True))[:10]
        dfs = []
        for fp in some_files:
            d = pd.read_parquet(fp)
            if "segment_id" not in d.columns:
                sid = _seg_from_path(fp)
                if sid is not None:
                    d["segment_id"] = sid
            dfs.append(d)
        pairs_preview = pd.concat(dfs, ignore_index=True)
        pairs_preview = pairs_preview.head(args.preview_rows)
        preview = pairs_preview.merge(meta_df, on="segment_id", how="left")
        preview_path = os.path.join(ds_root, "preview.parquet")
        preview.to_parquet(preview_path, index=False)
        print(f"[OK] wrote preview: {preview_path} (rows={len(preview)})")


if __name__ == "__main__":
    main()

