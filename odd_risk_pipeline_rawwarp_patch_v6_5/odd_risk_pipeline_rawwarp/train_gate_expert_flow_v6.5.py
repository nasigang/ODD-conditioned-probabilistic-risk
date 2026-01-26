from __future__ import annotations
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch
import argparse
import os

from risk_pipeline.schema import build_schema_from_columns, save_schema
from risk_pipeline.preprocess import build_preprocess_state, transform_dataframe, save_preprocess_state
from risk_pipeline.data import RiskCSVDataset, SegmentBalancedBatchSampler, SegmentBalancedPosBatchSampler, make_dataloader
from risk_pipeline.warp import RawWarpConfig
from risk_pipeline.models import GateMLP, ConditionalSpline1DFlow
from risk_pipeline.train import TrainConfig, train_gate_one_epoch_raw, eval_gate_raw, train_expert_one_epoch_raw, eval_expert_raw
from risk_pipeline.risk import RiskConfig, compute_risk

def split_by_segment(df: pd.DataFrame, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    segs = df["segment_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(segs)
    n = len(segs)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_segs = set(segs[:n_test])
    val_segs = set(segs[n_test:n_test+n_val])
    train_segs = set(segs[n_test+n_val:])
    return (df[df["segment_id"].isin(train_segs)].copy(),
            df[df["segment_id"].isin(val_segs)].copy(),
            df[df["segment_id"].isin(test_segs)].copy())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out", type=str, default="runs/odd_risk_rawwarp")
    ap.add_argument("--m_segments", type=int, default=64)
    ap.add_argument("--k_frames", type=int, default=4)
    ap.add_argument("--min_frame_gap", type=int, default=5, help="Minimum gap (frame_label units) between samples drawn within a segment.")
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--ttc_floor", type=float, default=0.05)
    ap.add_argument("--ttc_cap", type=float, default=10.0)
    ap.add_argument("--dropout", type=float, default=0.2, help="Callback dropout rate")
    ap.add_argument("--hidden", type=int, default=64, help="Model hidden dim")
    ap.add_argument("--depth", type=int, default=2, help="Model depth")
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--input_noise", type=float, default=0.02, help="Std of Gaussian noise added to inputs during training")
    ap.add_argument(
        "--expert_keep_c_cont",
        action="store_true",
        help="Expert/Flow conditioning: KEEP c__ continuous segment-level summary features. By default, the script DROPS them (sets to train mean -> scaled=0) because they often act as a segment fingerprint. Use this flag only if you have binned/regularized c__ continuous features and validated generalization.",
    )
    ap.add_argument("--expert_ctx_block_drop_prob", type=float, default=0.10,
                    help="Expert/Flow training only: with this probability per-sample, drop (set-to-mean) the entire c__ context block to reduce segment fingerprinting.")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--gate_epochs", type=int, default=30)
    ap.add_argument("--expert_epochs", type=int, default=50)
    ap.add_argument(
        "--drop_leakage_cols",
        action="store_true",
        help="Drop known leakage columns if present (label, x__best_ttci, best_ttci).",
    )
    ap.add_argument(
        "--context_mode",
        type=str,
        default="all",
        choices=["all", "odd_only", "none"],
        help=(
            "Which c__ (segment/context) features to include in Gate/Expert inputs. "
            "'all' = keep all c__*, 'odd_only' = keep only ODD one-hots + c__has_*, 'none' = drop all c__* (x__ only)."
        ),
    )
    ap.add_argument(
        "--expert_drop_all_context",
        action="store_true",
        help=(
            "For the Expert/Flow only, hard-drop all c__ context features by zeroing them out in both train+eval. "
            "Gate will still see c__ features (controlled by --context_mode)."
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # Memory-friendly read: treat all numeric columns as float32.
    header = pd.read_csv(args.csv, nrows=0)
    dtypes = {c: np.float32 for c in header.columns if c != "segment_id"}
    dtypes["segment_id"] = str
    df = pd.read_csv(args.csv, dtype=dtypes)

    if args.drop_leakage_cols:
        leak_cols = [c for c in ["label", "x__best_ttci", "best_ttci"] if c in df.columns]
        if leak_cols:
            df = df.drop(columns=leak_cols)

    df_train, df_val, df_test = split_by_segment(
        df, seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio
    )

    # Persist the segment split mapping for reproducibility
    os.makedirs(args.out, exist_ok=True)
    split_map = {
        "train": sorted(df_train["segment_id"].unique().tolist()),
        "val": sorted(df_val["segment_id"].unique().tolist()),
        "test": sorted(df_test["segment_id"].unique().tolist()),
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
    }
    with open(os.path.join(args.out, "segment_split.json"), "w", encoding="utf-8") as f:
        json.dump(split_map, f, indent=2)

    # 1) Schema is inferred from CSV headers.
    schema0 = build_schema_from_columns(df_train.columns, context_mode=args.context_mode)
    if (len(schema0.x_gate_cont) + len(schema0.x_gate_bin) + len(schema0.x_gate_onehot)) == 0:
        raise RuntimeError(
            "Schema matched zero features. Check x__/c__ prefixes and that ODD one-hot columns look like c__odd_weather=Rain."
        )

    # 2) Preprocess state *prunes* the schema by variance-filter (Train-only) and
    #    learns scaler + target standardization.
    state = build_preprocess_state(df_train, schema0, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)
    save_schema(state.schema, str(out / "feature_schema.json"))
    save_preprocess_state(state, str(out / "preprocess_state.json"))

    df_train_t = transform_dataframe(df_train, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)
    df_val_t = transform_dataframe(df_val, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)
    df_test_t = transform_dataframe(df_test, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)

    ds_train = RiskCSVDataset(df_train_t, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)
    ds_val = RiskCSVDataset(df_val_t, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)
    ds_test = RiskCSVDataset(df_test_t, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)

    gate_sampler = SegmentBalancedBatchSampler(
        ds_train.get_segment_indices(),
        ds_train.get_frame_labels(),
        args.m_segments,
        args.k_frames,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
        min_frame_gap=args.min_frame_gap,
    )
    gate_val_sampler = SegmentBalancedBatchSampler(
        ds_val.get_segment_indices(),
        ds_val.get_frame_labels(),
        args.m_segments,
        args.k_frames,
        steps_per_epoch=max(10, args.steps_per_epoch // 5),
        seed=args.seed + 100,
        min_frame_gap=args.min_frame_gap,
    )
    gate_loader = make_dataloader(ds_train, gate_sampler)
    gate_val_loader = make_dataloader(ds_val, gate_val_sampler)

    # Expert sampling: maximize segment diversity.
    # If Gate uses (M segments x K frames), Expert uses (M*K segments x 1 frame)
    # -> same batch size, far less duplication / fingerprint memorization.
    expert_m = args.m_segments * args.k_frames
    expert_k = 1
    expert_gap = max(args.min_frame_gap, 30)  # 3s @10Hz, reduce autocorrelation
    # ds_train.tensors is a dataclass (DatasetTensors), not a dict.
    n_pos = int(ds_train.tensors.expert_mask.sum().item())
    expert_steps = max(10, int((n_pos / max(1, expert_m)) + 0.999))

    expert_sampler = SegmentBalancedPosBatchSampler(
        ds_train.get_segment_pos_indices(),
        ds_train.get_frame_labels(),
        expert_m,
        expert_k,
        steps_per_epoch=expert_steps,
        seed=args.seed + 1,
        min_frame_gap=expert_gap,
    )
    expert_val_sampler = SegmentBalancedPosBatchSampler(
        ds_val.get_segment_pos_indices(),
        ds_val.get_frame_labels(),
        expert_m,
        expert_k,
        steps_per_epoch=max(10, expert_steps // 5),
        seed=args.seed + 101,
        min_frame_gap=expert_gap,
    )
    expert_loader = make_dataloader(ds_train, expert_sampler)
    expert_val_loader = make_dataloader(ds_val, expert_val_sampler)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TrainConfig in risk_pipeline/train.py is intentionally minimal. Keep seeds separate.
    cfg = TrainConfig(
        device=device,
        gate_epochs=args.gate_epochs,
        expert_epochs=args.expert_epochs,
        weight_decay=args.weight_decay,
        input_noise=args.input_noise,
    )

    scale_tensors = ds_train.get_scale_tensors(device)
    gate_mean, gate_std = scale_tensors["gate_mean"], scale_tensors["gate_std"]
    expert_mean, expert_std = scale_tensors["expert_mean"], scale_tensors["expert_std"]

    # --- Gate Training ---
    gate = GateMLP(ds_train.tensors.x_gate_raw.shape[1], hidden=args.hidden, depth=args.depth, dropout=args.dropout).to(device)
    feature_index = ds_train.gate_feature_index
    warp_cfg = RawWarpConfig()
    
    optimizer = torch.optim.AdamW(gate.parameters(), lr=cfg.gate_lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best = 1e9
    patience_counter = 0
    patience_limit = 5
    
    for ep in range(cfg.gate_epochs):
        tr = train_gate_one_epoch_raw(gate, gate_loader, optimizer, cfg, warp_cfg, feature_index, gate_mean, gate_std)
        va = eval_gate_raw(gate, gate_val_loader, device, gate_mean, gate_std)
        scheduler.step(va)
        
        print(f"[Gate] epoch {ep+1}/{cfg.gate_epochs} train={tr:.4f} val={va:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")
        if va < best:
            best = va
            patience_counter = 0
            torch.save(gate.state_dict(), out / "gate.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"[Gate] Early stopping at epoch {ep+1}")
                break

    # --- Expert Training ---
    flow = ConditionalSpline1DFlow(cond_dim=ds_train.tensors.x_expert_raw.shape[1], hidden=args.hidden, depth=args.depth, dropout=args.dropout).to(device)

    # Expert fingerprint guard: optionally drop c__ continuous segment-stat features (set to train mean => scaled 0)
    expert_colnames = ds_train.get_x_expert_colnames()
    ctx_all_idx = [i for i, c in enumerate(expert_colnames) if c.startswith("c__")]
    drop_idx = []
    drop_c_cont = not args.expert_keep_c_cont
    # Expert fingerprint suppression:
    # 1) optionally drop ALL c__ continuous cols (legacy behavior)
    # 2) ALWAYS apply auto-detected drop list from preprocess (per-segment uniqueness)
    drop_idx = []
    if drop_c_cont:
        drop_idx.extend(
            i for i, c in enumerate(expert_colnames)
            if c.startswith("c__") and ("=" not in c) and (not c.startswith("c__has_"))
        )
    if getattr(state, "drop_context_idx_expert", None):
        drop_idx.extend(list(state.drop_context_idx_expert))

    if args.expert_drop_all_context:
        # Permanently remove all segment-level context c__* from Expert (both train & eval).
        drop_idx.extend(ctx_all_idx)
        ctx_all_idx = []  # disable stochastic block-drop when we're already hard-dropping
    drop_idx = sorted(set(drop_idx))

    drop_idx_t = torch.tensor(drop_idx, dtype=torch.long, device=device) if len(drop_idx) else None
    ctx_all_idx_t = torch.tensor(ctx_all_idx, dtype=torch.long, device=device) if len(ctx_all_idx) else None
    if len(drop_idx):
        print(f"[Expert] context drop idx: {len(drop_idx)}/{len(expert_colnames)} (drop_c_cont={drop_c_cont}, auto_drop={len(getattr(state,'drop_context_idx_expert',[]))})")
    
    optimizer = torch.optim.AdamW(flow.parameters(), lr=cfg.expert_lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best = 1e9
    patience_counter = 0
    patience_limit = 10

    for ep in range(cfg.expert_epochs):
        tr = train_expert_one_epoch_raw(
            flow, expert_loader, optimizer, cfg, expert_mean, expert_std,
            drop_idx=drop_idx_t,
            ctx_all_idx=ctx_all_idx_t,
            ctx_block_drop_prob=args.expert_ctx_block_drop_prob,
        )
        va = eval_expert_raw(flow, expert_val_loader, device, expert_mean, expert_std, drop_idx=drop_idx_t)
        scheduler.step(va)
        
        print(f"[Expert] epoch {ep+1}/{cfg.expert_epochs} train_nll={tr:.4f} val_nll={va:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")
        if va < best:
            best = va
            patience_counter = 0
            torch.save(flow.state_dict(), out / "expert_flow.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"[Expert] Early stopping at epoch {ep+1}")
                break

    # risk demo
    gate.load_state_dict(torch.load(out / "gate.pt", map_location=device))
    flow.load_state_dict(torch.load(out / "expert_flow.pt", map_location=device))
    gate.eval(); flow.eval()

    batch = next(iter(gate_val_loader))
    xg_raw = batch["x_gate_raw"].to(device)
    xe_raw = batch["x_expert_raw"].to(device)
    v_close = (batch.get("raw_closing_speed_mps") if isinstance(batch, dict) and "raw_closing_speed_mps" in batch else batch["raw_speed_mps"]).to(device)

    xg = (xg_raw - gate_mean) / (gate_std + 1e-6)
    if drop_idx_t is not None and drop_idx_t.numel() > 0:
        xe_raw = xe_raw.clone()
        xe_raw[:, drop_idx_t] = expert_mean[drop_idx_t]
    xe = (xe_raw - expert_mean) / (expert_std + 1e-6)
    logits = gate(xg)

    rcfg = RiskConfig(tau=0.7, a_max=6.0, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)
    risk, p_gate = compute_risk(logits, flow, xe, v_close, state.target_std, rcfg)

    stats = {
        "risk_mean": float(risk.mean().item()),
        "risk_p95": float(torch.quantile(risk, 0.95).item()),
        "p_gate_mean": float(p_gate.mean().item()),
    }
    print("Risk demo:", stats)
    with open(out / "risk_demo.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
