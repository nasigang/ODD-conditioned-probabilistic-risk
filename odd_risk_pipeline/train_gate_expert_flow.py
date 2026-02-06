from __future__ import annotations
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch
import os
import re
from typing import Dict, List, Tuple, Iterable, Optional


# -----------------------------
# Gate temperature scaling helpers
# -----------------------------
@torch.no_grad()
def _collect_gate_logits_and_targets(
    gate: GateMLP,
    loader,
    gate_mean: torch.Tensor,
    gate_std: torch.Tensor,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect uncalibrated gate logits and targets from a loader."""
    gate.eval()
    logits_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []
    for batch in loader:
        x_raw = batch["x_gate_raw"].to(device)
        y = batch["y_gate"].to(device).float()
        x = (x_raw - gate_mean) / (gate_std + 1e-12)
        logits = gate(x).view(-1)
        logits_all.append(logits.detach().cpu())
        y_all.append(y.view(-1).detach().cpu())
    return torch.cat(logits_all, dim=0), torch.cat(y_all, dim=0)

def _fit_gate_temperature(
    logits: torch.Tensor,
    y: torch.Tensor,
    init_temperature: float = 1.0,
    max_iter: int = 50,
) -> float:
    """Fit a single temperature T>0 minimizing BCE(logits/T, y) on validation data."""
    logits = logits.detach().float()
    y = y.detach().float()
    # Optimize log(T) to enforce positivity.
    log_T = torch.nn.Parameter(torch.log(torch.tensor([init_temperature], dtype=torch.float32)))
    bce = torch.nn.BCEWithLogitsLoss(reduction="mean")

    optimizer = torch.optim.LBFGS([log_T], lr=0.2, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        T = torch.exp(log_T)
        loss = bce(logits / T, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = float(torch.exp(log_T).item())
    # Clamp to a reasonable range to avoid pathological outputs if val set is tiny.
    T = float(max(0.05, min(20.0, T)))
    return T

def _parse_csv_list(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(',') if x.strip()]

def _build_x_to_c_rename_map(columns: Iterable[str], patterns: List[str]) -> Dict[str, str]:
    """Rename x__* columns to c__* if any regex pattern matches the column name."""
    if not patterns:
        return {}
    regs = [re.compile(p) for p in patterns]
    cols = list(columns)
    colset = set(cols)
    rename: Dict[str, str] = {}
    for col in cols:
        if not col.startswith('x__'):
            continue
        if any(r.search(col) for r in regs):
            new = 'c__' + col[len('x__'):]
            if new in colset and new != col:
                raise ValueError(f"[x_to_c] rename collision: {col} -> {new} already exists in dataframe.")
            rename[col] = new
    return rename

def _apply_rename(df: 'pd.DataFrame', rename: Dict[str, str]) -> 'pd.DataFrame':
    return df.rename(columns=rename) if rename else df

def _schema_expected_columns(schema) -> List[str]:
    expected = set()
    for attr in dir(schema):
        if 'cols' not in attr:
            continue
        try:
            v = getattr(schema, attr)
        except Exception:
            continue
        if isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v):
            expected.update(v)
    return sorted(expected)

def _auto_align_xc_prefix_swap(df: 'pd.DataFrame', schema) -> Tuple['pd.DataFrame', Dict[str, str]]:
    """If schema expects c__foo but df has x__foo (or vice versa), rename automatically."""
    expected = _schema_expected_columns(schema)
    cols = set(df.columns)
    rename: Dict[str, str] = {}
    for exp in expected:
        if exp in cols:
            continue
        if exp.startswith('c__'):
            alt = 'x__' + exp[len('c__'):]
        elif exp.startswith('x__'):
            alt = 'c__' + exp[len('x__'):]
        else:
            continue
        if alt in cols and exp not in cols:
            rename[alt] = exp
    if rename:
        df = df.rename(columns=rename)
    return df, rename

def _match_cols_by_regex(colnames: List[str], patterns: List[str]) -> List[str]:
    if not patterns:
        return []
    regs = [re.compile(p) for p in patterns]
    return [c for c in colnames if any(r.search(c) for r in regs)]


from risk_pipeline.schema import build_schema_from_columns, save_schema
from risk_pipeline.preprocess import build_preprocess_state, transform_dataframe, save_preprocess_state, GateLabelConfig
from risk_pipeline.data import RiskCSVDataset, SegmentBalancedBatchSampler, SegmentBalancedPosBatchSampler, make_dataloader
from risk_pipeline.warp import RawWarpConfig
from risk_pipeline.models import GateMLP, ConditionalSpline1DFlow
from risk_pipeline.train import (
    TrainConfig,
    train_gate_one_epoch_raw, eval_gate_raw,
    train_expert_one_epoch_raw, eval_expert_raw,
)

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

def _set_finetune_head_only(flow: ConditionalSpline1DFlow) -> None:
    """Freeze all params except the final projection (head).

    concat mode: only the last Linear in `net`
    film mode: only `head`
    """
    for p in flow.parameters():
        p.requires_grad = False
    if getattr(flow, "cond_mode", "concat") == "concat":
        # net is Sequential; last layer is Linear
        last = list(flow.net.modules())[-1]  # includes Sequential itself; safer: index
        # safer: direct
        for p in flow.net[-1].parameters():
            p.requires_grad = True
    else:
        for p in flow.head.parameters():
            p.requires_grad = True

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out", type=str, default="runs/odd_risk_two_stage")
    ap.add_argument("--m_segments", type=int, default=64)
    ap.add_argument("--k_frames", type=int, default=4)
    ap.add_argument("--min_frame_gap", type=int, default=5, help="Minimum gap (frame_label units) between samples drawn within a segment.")
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--ttc_floor", type=float, default=0.05)
    ap.add_argument("--ttc_cap", type=float, default=10.0)
    ap.add_argument("--schema_profile", type=str, default="minimal_v2", choices=["auto","minimal_v2"],
                    help="Which feature-role schema to use. minimal_v2 enforces disjoint Gate/Expert features.")

    # Gate-label proxy thresholds (must match preprocess + warp labeling)
    ap.add_argument("--gate_candidate_range_m", type=float, default=50.0)
    ap.add_argument("--gate_closing_thr_mps", type=float, default=0.5)
    ap.add_argument("--gate_ttc_max_s", type=float, default=10.0)

    # --- Warp controls ---
    ap.add_argument("--warp_p", type=float, default=0.35, help="(Backward compatible) default warp probability used for both Gate & Expert unless overridden.")
    ap.add_argument("--gate_warp_p", type=float, default=None, help="Override Gate warp probability only.")
    ap.add_argument("--expert_warp_p", type=float, default=None, help="Override Expert warp probability only (pretrain stage).")

    ap.add_argument("--warp_closing_add_min", type=float, default=0.5, help="Warp: additive closing speed min (m/s)")
    ap.add_argument("--warp_closing_add_max", type=float, default=6.0, help="Warp: additive closing speed max (m/s)")
    ap.add_argument("--warp_speed_scale_min", type=float, default=1.05)
    ap.add_argument("--warp_speed_scale_max", type=float, default=1.25)

    # model/opt
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--input_noise", type=float, default=0.02)

    ap.add_argument(
        "--expert_keep_c_cont",
        action="store_true",
        help="Expert/Flow conditioning: KEEP c__ continuous segment-level summary features. By default, the script DROPS them.",
    )
    ap.add_argument("--expert_ctx_block_drop_prob", type=float, default=0.10)

    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)

    ap.add_argument("--gate_epochs", type=int, default=30)


    # Gate calibration (temperature scaling on VAL)
    ap.add_argument("--disable_gate_temperature_scaling", action="store_true",
                    help="Disable post-hoc temperature scaling for Gate logits.")
    ap.add_argument("--gate_temperature_init", type=float, default=1.0,
                    help="Initial temperature value for scaling (T>0).")
    ap.add_argument("--gate_temperature_max_iter", type=int, default=50,
                    help="Max LBFGS iterations for temperature scaling fit.")
    ap.add_argument("--expert_epochs", type=int, default=1, help="Expert pretrain epochs (WITH warp if expert_warp_p>0). Set to 1 for Strategy A.")
    ap.add_argument("--expert_finetune_epochs", type=int, default=3, help="Additional Expert fine-tuning epochs on CLEAN data only (warp disabled).")

    ap.add_argument("--expert_finetune_lr_mult", type=float, default=0.2, help="Multiply expert lr by this factor for fine-tuning.")
    ap.add_argument("--expert_finetune_head_only", action="store_true", help="Fine-tune only the last head/projection layer (recommended to preserve mapping learned in pretrain).")

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
        help=("Which c__ (segment/context) features to include in Gate/Expert inputs."),
    )
    ap.add_argument(
        "--expert_drop_all_context",
        action="store_true",
        help=("For the Expert/Flow only, hard-drop all c__ context features by zeroing them out in both train+eval."),
    )
    ap.add_argument("--x_to_c_regex", type=str, default="")
    ap.add_argument("--expert_drop_feature_regex", type=str, default="")
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

    # Optional role remap: x__ -> c__
    x_to_c_patterns = _parse_csv_list(args.x_to_c_regex)
    rename_map = _build_x_to_c_rename_map(df.columns, x_to_c_patterns)
    if rename_map:
        print(f"[x_to_c] renaming {len(rename_map)} columns (x__ -> c__). Example: {list(rename_map.items())[:5]}")
        df = _apply_rename(df, rename_map)

    df_train, df_val, df_test = split_by_segment(df, seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # Persist split mapping
    split_map = {
        "train": sorted(df_train["segment_id"].unique().tolist()),
        "val": sorted(df_val["segment_id"].unique().tolist()),
        "test": sorted(df_test["segment_id"].unique().tolist()),
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
    }
    with open(out / "segment_split.json", "w", encoding="utf-8") as f:
        json.dump(split_map, f, indent=2)

    # Schema + preprocess
    schema0 = build_schema_from_columns(df_train.columns, context_mode=args.context_mode, schema_profile=args.schema_profile)
    if (len(schema0.x_gate_cont) + len(schema0.x_gate_bin) + len(schema0.x_gate_onehot)) == 0:
        raise RuntimeError("Schema matched zero features. Check x__/c__ prefixes.")

    state = build_preprocess_state(
        df_train,
        schema0,
        ttc_floor=args.ttc_floor,
        ttc_cap=args.ttc_cap,
        gate_label_cfg=GateLabelConfig(
            candidate_range_m=args.gate_candidate_range_m,
            closing_thr_mps=args.gate_closing_thr_mps,
            ttc_max_s=args.gate_ttc_max_s,
        ),
    )
    save_schema(state.schema, str(out / "feature_schema.json"))
    save_preprocess_state(state, str(out / "preprocess_state.json"))

    df_train_t = transform_dataframe(df_train, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)
    df_val_t = transform_dataframe(df_val, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)
    df_test_t = transform_dataframe(df_test, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)

    ds_train = RiskCSVDataset(df_train_t, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)
    ds_val = RiskCSVDataset(df_val_t, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)

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

    # Expert sampling (pos-only)
    expert_m = args.m_segments * args.k_frames
    expert_k = 1
    expert_gap = max(args.min_frame_gap, 30)
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

    # --------------------
    # Gate training (warp)
    # --------------------
    gate = GateMLP(ds_train.tensors.x_gate_raw.shape[1], hidden=args.hidden, depth=args.depth, dropout=args.dropout).to(device)
    gate_feature_index = ds_train.gate_feature_index

    gate_p = float(args.gate_warp_p) if args.gate_warp_p is not None else float(args.warp_p)
    gate_warp_cfg = RawWarpConfig(
        p_warp=gate_p,
        closing_add_min=float(args.warp_closing_add_min),
        closing_add_max=float(args.warp_closing_add_max),
        speed_scale_min=float(args.warp_speed_scale_min),
        speed_scale_max=float(args.warp_speed_scale_max),
        candidate_range_m=float(args.gate_candidate_range_m),
        closing_thr_mps=float(args.gate_closing_thr_mps),
        ttc_max_s=float(args.gate_ttc_max_s),
    )

    optimizer_g = torch.optim.AdamW(gate.parameters(), lr=cfg.gate_lr, weight_decay=cfg.weight_decay)
    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=3, verbose=True)

    best = 1e9
    patience_counter = 0
    patience_limit = 5

    for ep in range(cfg.gate_epochs):
        tr = train_gate_one_epoch_raw(gate, gate_loader, optimizer_g, cfg, gate_warp_cfg, gate_feature_index, gate_mean, gate_std)
        va = eval_gate_raw(gate, gate_val_loader, device, gate_mean, gate_std)
        scheduler_g.step(va)

        print(f"[Gate] epoch {ep+1}/{cfg.gate_epochs} train={tr:.4f} val={va:.4f} lr={optimizer_g.param_groups[0]['lr']:.2e}")
        if va < best:
            best = va
            patience_counter = 0
            torch.save(gate.state_dict(), out / "gate.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"[Gate] Early stopping at epoch {ep+1}")
                break



    # -----------------------------
    # Gate temperature scaling (VAL)
    # -----------------------------
    # Calibrates probabilities without changing ranking/AUC.
    if not args.disable_gate_temperature_scaling:
        # Load best checkpoint saved during early stopping.
        if (out / "gate.pt").exists():
            gate.load_state_dict(torch.load(out / "gate.pt", map_location=device))
        gate.eval()

        logits_val, y_val = _collect_gate_logits_and_targets(gate, gate_val_loader, gate_mean, gate_std, device)
        nll_before = float(torch.nn.functional.binary_cross_entropy_with_logits(logits_val, y_val, reduction="mean").item())

        T = _fit_gate_temperature(
            logits_val,
            y_val,
            init_temperature=float(args.gate_temperature_init),
            max_iter=int(args.gate_temperature_max_iter),
        )
        nll_after = float(torch.nn.functional.binary_cross_entropy_with_logits(logits_val / T, y_val, reduction="mean").item())
        print(f"[Gate-Cal] temperature T={T:.3f}  NLL(before)={nll_before:.4f}  NLL(after)={nll_after:.4f}")

        with open(out / "gate_temperature.json", "w", encoding="utf-8") as f:
            json.dump({"temperature": float(T), "nll_before": nll_before, "nll_after": nll_after}, f, indent=2)
    else:
        print("[Gate-Cal] disabled (using raw logits)")

    # -----------------------------
    # Expert training: Two-stage A
    # -----------------------------
    # Build expert feature index (name -> column idx in x_expert_raw)
    expert_colnames = ds_train.get_x_expert_colnames()
    expert_feature_index = {name: i for i, name in enumerate(expert_colnames)}

    flow = ConditionalSpline1DFlow(
        cond_dim=ds_train.tensors.x_expert_raw.shape[1],
        hidden=args.hidden, depth=args.depth, dropout=args.dropout
    ).to(device)

    # Expert context drop indices
    ctx_all_idx = [i for i, c in enumerate(expert_colnames) if c.startswith("c__")]
    drop_idx = []
    drop_c_cont = not args.expert_keep_c_cont

    if drop_c_cont:
        drop_idx.extend(
            i for i, c in enumerate(expert_colnames)
            if c.startswith("c__") and ("=" not in c) and (not c.startswith("c__has_"))
        )
    if getattr(state, "drop_context_idx_expert", None):
        drop_idx.extend(list(state.drop_context_idx_expert))

    if args.expert_drop_all_context:
        drop_idx.extend(ctx_all_idx)
        ctx_all_idx = []

    extra_drop_patterns = _parse_csv_list(args.expert_drop_feature_regex)
    if extra_drop_patterns:
        extra_cols = _match_cols_by_regex(expert_colnames, extra_drop_patterns)
        if extra_cols:
            extra_idx = [expert_colnames.index(c) for c in extra_cols]
            print(f"[expert_drop_feature_regex] hard-drop {len(extra_idx)} expert cols. Example: {extra_cols[:10]}")
            drop_idx.extend(extra_idx)

    drop_idx = sorted(set(drop_idx))
    drop_idx_t = torch.tensor(drop_idx, dtype=torch.long, device=device) if len(drop_idx) else None
    ctx_all_idx_t = torch.tensor(ctx_all_idx, dtype=torch.long, device=device) if len(ctx_all_idx) else None

    if len(drop_idx):
        print(f"[Expert] context drop idx: {len(drop_idx)}/{len(expert_colnames)} (drop_c_cont={drop_c_cont}, auto_drop={len(getattr(state,'drop_context_idx_expert',[]))})")

    # Warp config for Expert pretrain
    expert_p = float(args.expert_warp_p) if args.expert_warp_p is not None else float(args.warp_p)
    expert_warp_cfg = RawWarpConfig(
        p_warp=expert_p,
        closing_add_min=float(args.warp_closing_add_min),
        closing_add_max=float(args.warp_closing_add_max),
        speed_scale_min=float(args.warp_speed_scale_min),
        speed_scale_max=float(args.warp_speed_scale_max),
        candidate_range_m=float(args.gate_candidate_range_m),
        closing_thr_mps=float(args.gate_closing_thr_mps),
        ttc_max_s=float(args.gate_ttc_max_s),
    )

    # Stage 1: Pretrain with Warp (v5_1 behavior)
    optimizer_e = torch.optim.AdamW(flow.parameters(), lr=cfg.expert_lr, weight_decay=cfg.weight_decay)
    scheduler_e = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_e, mode='min', factor=0.5, patience=5, verbose=True)

    best_pre = 1e9
    for ep in range(int(args.expert_epochs)):
        tr = train_expert_one_epoch_raw(
            flow, expert_loader, optimizer_e, cfg, expert_mean, expert_std,
            drop_idx=drop_idx_t, ctx_all_idx=ctx_all_idx_t, ctx_block_drop_prob=float(args.expert_ctx_block_drop_prob),
            flow_x_idx=getattr(state, "flow_x_idx", None), flow_c_idx=getattr(state, "flow_c_idx", None),
            warp_cfg=expert_warp_cfg,
            expert_feature_index=expert_feature_index,
            ttc_floor_s=float(args.ttc_floor), ttc_cap_s=float(args.ttc_cap),
            target_mu_y=float(state.target_std.mu_y), target_sigma_y=float(state.target_std.sigma_y),
        )
        va = eval_expert_raw(
            flow, expert_val_loader, device, expert_mean, expert_std,
            drop_idx=drop_idx_t,
            flow_x_idx=getattr(state, "flow_x_idx", None), flow_c_idx=getattr(state, "flow_c_idx", None),
        )
        scheduler_e.step(va)
        print(f"[Expert-Pre] epoch {ep+1}/{int(args.expert_epochs)} train_nll={tr:.4f} val_nll={va:.4f} lr={optimizer_e.param_groups[0]['lr']:.2e}")
        if va < best_pre:
            best_pre = va
            torch.save(flow.state_dict(), out / "expert_flow_pre.pt")

    # Load best pretrain checkpoint before fine-tuning
    if (out / "expert_flow_pre.pt").exists():
        flow.load_state_dict(torch.load(out / "expert_flow_pre.pt", map_location=device))

    # Stage 2: Fine-tune on CLEAN only (warp disabled)
    if int(args.expert_finetune_epochs) > 0:
        if args.expert_finetune_head_only:
            _set_finetune_head_only(flow)

        # Rebuild optimizer for trainable params only, with reduced LR
        trainable = [p for p in flow.parameters() if p.requires_grad]
        lr_ft = float(cfg.expert_lr) * float(args.expert_finetune_lr_mult)
        optimizer_ft = torch.optim.AdamW(trainable, lr=lr_ft, weight_decay=float(cfg.weight_decay))
        scheduler_ft = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=2, verbose=True)

        best_ft = 1e9
        for ep in range(int(args.expert_finetune_epochs)):
            tr = train_expert_one_epoch_raw(
                flow, expert_loader, optimizer_ft, cfg, expert_mean, expert_std,
                drop_idx=drop_idx_t, ctx_all_idx=ctx_all_idx_t, ctx_block_drop_prob=float(args.expert_ctx_block_drop_prob),
                flow_x_idx=getattr(state, "flow_x_idx", None), flow_c_idx=getattr(state, "flow_c_idx", None),
                # CLEAN: no warp
                warp_cfg=None,
            )
            va = eval_expert_raw(
                flow, expert_val_loader, device, expert_mean, expert_std,
                drop_idx=drop_idx_t,
                flow_x_idx=getattr(state, "flow_x_idx", None), flow_c_idx=getattr(state, "flow_c_idx", None),
            )
            scheduler_ft.step(va)
            print(f"[Expert-FT] epoch {ep+1}/{int(args.expert_finetune_epochs)} train_nll={tr:.4f} val_nll={va:.4f} lr={optimizer_ft.param_groups[0]['lr']:.2e}")
            if va < best_ft:
                best_ft = va
                torch.save(flow.state_dict(), out / "expert_flow_ft.pt")
                # also overwrite the canonical checkpoint for eval scripts
                torch.save(flow.state_dict(), out / "expert_flow.pt")

        print(f"[Expert-FT] best val_nll={best_ft:.4f} saved to expert_flow.pt (and expert_flow_ft.pt)")

    else:
        # No fine-tuning: use pretrain checkpoint as canonical
        torch.save(flow.state_dict(), out / "expert_flow.pt")

    print("[Done] checkpoints:")
    print(" -", out / "gate.pt")
    print(" -", out / "expert_flow_pre.pt")
    print(" -", out / "expert_flow.pt")

if __name__ == "__main__":
    main()
