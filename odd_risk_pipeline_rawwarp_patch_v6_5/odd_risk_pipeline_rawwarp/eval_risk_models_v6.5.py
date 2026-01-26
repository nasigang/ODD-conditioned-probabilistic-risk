#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Gate/Expert/Risk models on held-out segment splits (ODD risk raw-warp pipeline compatible).

Key compat fixes vs v1:
- segment_split.json may contain metadata ints (seed, counts) -> ignore non-split keys
- Always call transform_dataframe() to guarantee y_gate/y_expert/expert_mask exist
- Dataset keeps RAW features (for raw-warp ordering) -> apply z-score scaling in eval (like training)
- Apply expert fingerprint suppression via state.drop_context_idx_expert
- Infer GateMLP / ConditionalSpline1DFlow hyperparams from checkpoint state_dict shapes
- Flow p_event computed via expert.cdf (exact), not MC sampling
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import torch

# sklearn is convenient; fallback gracefully if missing
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:
    roc_auc_score = None
    average_precision_score = None


# ----------------------------
# Metrics
# ----------------------------
def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if roc_auc_score is None:
        return float("nan")
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_prauc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if average_precision_score is None:
        return float("nan")
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_prob = np.clip(y_prob.astype(np.float64), 1e-7, 1 - 1e-7)
    return float(np.mean((y_prob - y_true) ** 2))


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    y_true = y_true.astype(np.float64)
    y_prob = np.clip(y_prob.astype(np.float64), 1e-7, 1 - 1e-7)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    N = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if not np.any(m):
            continue
        acc = float(np.mean(y_true[m]))
        conf = float(np.mean(y_prob[m]))
        ece_val += (float(np.sum(m)) / float(N)) * abs(acc - conf)
    return float(ece_val)


# ----------------------------
# Labels / thresholds
# ----------------------------
def compute_ttc(df: pd.DataFrame, eps: float = 1e-6) -> np.ndarray:
    """Prefer min_ttc_est if exists, else TTC from y_soft (=1/TTC)."""
    if "min_ttc_est" in df.columns:
        ttc_raw = pd.to_numeric(df["min_ttc_est"], errors="coerce").to_numpy(np.float64)
        ttc_raw = np.where(np.isfinite(ttc_raw) & (ttc_raw > 0.0), ttc_raw, np.inf)
        return ttc_raw
    y_soft = pd.to_numeric(df["y_soft"], errors="coerce").to_numpy(np.float64)
    valid = np.isfinite(y_soft) & (y_soft > 0.0)
    ttc = np.where(valid, 1.0 / (y_soft + eps), np.inf)
    return ttc


def s_star(v: np.ndarray, tau: float, amax: float, cap: float) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    s = tau + np.maximum(v, 0.0) / max(amax, 1e-6)
    return np.clip(s, 0.0, cap)


def build_event_label(
    df: pd.DataFrame,
    *,
    ttc_floor: float,
    ttc_cap: float,
    label_mode: str,
    sstar_mode: str,
    tau: float,
    amax: float,
    fixed_ttc: float = 2.0,
    require_edges: bool = True,
    y_gate_proxy: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Returns binary event label y_event for risk scoring.

    label_mode:
      - ttc_sstar: 1 if TTC <= s*(v)
      - ttc_fixed: 1 if TTC <= fixed_ttc
      - gate_clean: 1 if y_gate_proxy==1 (proxy of "non-empty interaction")
    """
    ttc = compute_ttc(df, eps=1e-6)
    ttc = np.clip(ttc, ttc_floor, ttc_cap)

    if label_mode == "gate_clean":
        if y_gate_proxy is None:
            # conservative fallback
            y_gate_proxy = (pd.to_numeric(df.get("n_edges", 0), errors="coerce").fillna(0.0).to_numpy() > 0.0).astype(
                np.int64
            )
        return (y_gate_proxy > 0).astype(np.int64)

    if label_mode == "ttc_fixed":
        y = (ttc <= float(fixed_ttc)).astype(np.int64)
    elif label_mode == "ttc_sstar":
        if sstar_mode == "ego_speed":
            v = pd.to_numeric(df.get("x__ego_speed_mps", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float64)
        else:
            # closing_speed preferred; fallback to ego speed
            if "x__max_closing_speed_any_mps" in df.columns:
                v = pd.to_numeric(df["x__max_closing_speed_any_mps"], errors="coerce").fillna(0.0).to_numpy(np.float64)
            elif "x__closing_speed_mps_max" in df.columns:
                v = pd.to_numeric(df["x__closing_speed_mps_max"], errors="coerce").fillna(0.0).to_numpy(np.float64)
            else:
                v = pd.to_numeric(df.get("x__ego_speed_mps", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float64)

        thr = s_star(v, tau=float(tau), amax=float(amax), cap=float(ttc_cap))
        y = (ttc <= thr).astype(np.int64)
    else:
        raise ValueError(f"Unknown label_mode={label_mode}")

    if require_edges and y_gate_proxy is not None:
        y = (y.astype(bool) & (y_gate_proxy > 0)).astype(np.int64)
    return y


def build_event_threshold_ttc(
    df: pd.DataFrame,
    *,
    ttc_cap: float,
    label_mode: str,
    sstar_mode: str,
    tau: float,
    amax: float,
    fixed_ttc: float,
) -> np.ndarray:
    """Return per-sample TTC threshold in seconds."""
    if label_mode == "ttc_fixed":
        return np.full((len(df),), float(fixed_ttc), dtype=np.float64)
    if label_mode == "gate_clean":
        # Not used for expert thresholding, but keep defined
        return np.full((len(df),), float(ttc_cap), dtype=np.float64)
    if label_mode != "ttc_sstar":
        raise ValueError(label_mode)

    if sstar_mode == "ego_speed":
        v = pd.to_numeric(df.get("x__ego_speed_mps", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float64)
    else:
        if "x__max_closing_speed_any_mps" in df.columns:
            v = pd.to_numeric(df["x__max_closing_speed_any_mps"], errors="coerce").fillna(0.0).to_numpy(np.float64)
        elif "x__closing_speed_mps_max" in df.columns:
            v = pd.to_numeric(df["x__closing_speed_mps_max"], errors="coerce").fillna(0.0).to_numpy(np.float64)
        else:
            v = pd.to_numeric(df.get("x__ego_speed_mps", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float64)

    return s_star(v, tau=float(tau), amax=float(amax), cap=float(ttc_cap))


# ----------------------------
# Load run artifacts
# ----------------------------
def load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_id_list(v: Any) -> Optional[List[str]]:
    if isinstance(v, (list, tuple, set)):
        return [str(x) for x in v]
    # allow nested dict schema: {"segment_ids":[...]} etc.
    if isinstance(v, dict):
        for kk in ("segment_ids", "segments", "ids"):
            vv = v.get(kk, None)
            if isinstance(vv, (list, tuple, set)):
                return [str(x) for x in vv]
    return None


def load_segment_split(run_dir: Path) -> Dict[str, List[str]]:
    """
    segment_split.json is required for fair eval.
    This loader ignores metadata keys (seed, counts, etc.) safely.
    """
    p = run_dir / "segment_split.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p} (need segment_split.json for fair eval)")
    d = load_json(p)

    out: Dict[str, List[str]] = {}
    # case1: {"train":[...], "val":[...], "test":[...], "seed":123}
    for k in ("train", "val", "test"):
        if k in d:
            ids = _as_id_list(d[k])
            if ids is not None:
                out[k] = ids

    # case2: {"train_segments":[...], ...}
    if not out and any((k.endswith("_segments") for k in d.keys())):
        for k in ("train", "val", "test"):
            kk = f"{k}_segments"
            if kk in d:
                ids = _as_id_list(d[kk])
                if ids is not None:
                    out[k] = ids

    if not out:
        raise TypeError(f"{p} does not contain usable split lists. Keys={list(d.keys())[:20]}")
    return out


def pick_split_df(df: pd.DataFrame, segs: List[str]) -> pd.DataFrame:
    segset = set(map(str, segs))
    return df[df["segment_id"].astype(str).isin(segset)].copy()


def detect_expert_type(run_dir: Path) -> str:
    if (run_dir / "expert_gauss.pt").exists():
        return "gauss"
    if (run_dir / "expert_flow.pt").exists() or (run_dir / "expert.pt").exists():
        return "flow"
    raise FileNotFoundError(f"Cannot find expert checkpoint in {run_dir}")


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in sd.keys()):
        return sd
    return {k[len("module.") :]: v for k, v in sd.items()}


def _torch_load_sd(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Load a pure state_dict robustly.
    - supports torch >= 2.0 weights_only
    - supports checkpoints wrapped as {"state_dict": ...}
    - strips DDP 'module.' prefix
    """
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location=device)

    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint {path} is not a state_dict (type={type(obj)})")
    obj = _strip_module_prefix(obj)
    return obj


def _infer_mlp_dims_from_sd(sd: Dict[str, torch.Tensor], *, prefix: Optional[str] = None) -> Tuple[int, int, int]:
    """
    Infer (in_dim, hidden, depth) for a plain MLP that ends with 1-dim output.
    We count 2D weight matrices.
    depth is #hidden layers (not counting final output linear).
    """
    items = sd.items()
    if prefix is not None:
        items = [(k, v) for k, v in items if k.startswith(prefix)]

    w2 = [(k, v) for k, v in items if k.endswith("weight") and getattr(v, "ndim", 0) == 2]
    if not w2:
        raise RuntimeError("Could not infer MLP dims: no 2D weights found.")
    # first linear
    first_w = w2[0][1]
    hidden = int(first_w.shape[0])
    in_dim = int(first_w.shape[1])
    n_linear = len(w2)
    depth = max(1, n_linear - 1)  # n_linear = depth + 1
    return in_dim, hidden, depth


def _infer_flow_hparams_from_sd(flow_sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    """
    Infer (hidden, depth, num_bins) for ConditionalSpline1DFlow based on its `net.*.weight` shapes.
    out_dim should be 3*num_bins + 1.
    depth is #hidden layers (not counting final output linear).
    """
    lin_keys = [k for k, v in flow_sd.items() if k.startswith("net.") and k.endswith(".weight") and getattr(v, "ndim", 0) == 2]
    if not lin_keys:
        raise RuntimeError("Could not find flow linear weights under net.*.weight")

    def _net_idx(k: str) -> int:
        # "net.6.weight" -> 6
        try:
            return int(k.split(".")[1])
        except Exception:
            return -1

    lin_keys_sorted = sorted(lin_keys, key=_net_idx)
    first_w = flow_sd[lin_keys_sorted[0]]
    last_w = flow_sd[lin_keys_sorted[-1]]

    hidden = int(first_w.shape[0])
    out_dim = int(last_w.shape[0])
    num_bins = int((out_dim - 1) // 3)
    if (3 * num_bins + 1) != out_dim:
        raise RuntimeError(f"Flow out_dim={out_dim} not compatible with num_bins (expected out_dim=3K+1).")
    depth = max(1, len(lin_keys_sorted) - 1)
    return hidden, depth, num_bins


def load_models_for_run(run_dir: Path, state, device: torch.device):
    """
    Build Gate + Expert models compatible with this pipeline.
    - Gate: GateMLP(in_dim, hidden, depth)
    - Flow: ConditionalSpline1DFlow(cond_dim, num_bins, hidden, depth)
    """
    from risk_pipeline.models import GateMLP, ConditionalSpline1DFlow  # type: ignore

    # Gate
    gate_ckpt = run_dir / "gate.pt"
    if not gate_ckpt.exists():
        raise FileNotFoundError(f"Missing {gate_ckpt}")
    gate_sd = _torch_load_sd(gate_ckpt, device=torch.device("cpu"))
    in_dim, hidden, depth = _infer_mlp_dims_from_sd(gate_sd)
    gate = GateMLP(in_dim, hidden=hidden, depth=depth, dropout=0.0).to(device)
    gate.load_state_dict(gate_sd, strict=True)
    gate.eval()

    # Expert
    flow_ckpt = run_dir / "expert_flow.pt"
    if not flow_ckpt.exists():
        # legacy name
        legacy = run_dir / "expert.pt"
        if legacy.exists():
            flow_ckpt = legacy

    if flow_ckpt.exists():
        flow_sd = _torch_load_sd(flow_ckpt, device=torch.device("cpu"))
        ctx_dim = len(state.schema.x_expert_cols_in_order())
        f_hidden, f_depth, num_bins = _infer_flow_hparams_from_sd(flow_sd)
        expert = ConditionalSpline1DFlow(
            cond_dim=ctx_dim, num_bins=num_bins, hidden=f_hidden, depth=f_depth, dropout=0.0
        ).to(device)
        expert.load_state_dict(flow_sd, strict=True)
        expert.eval()
        return gate, expert, "flow"

    # Gaussian expert support (only if your codebase provides it)
    gauss_ckpt = run_dir / "expert_gauss.pt"
    if gauss_ckpt.exists():
        import risk_pipeline.models as M  # type: ignore

        # Try common class names
        cand = [
            "HeteroscedasticGaussianExpert",
            "HeteroscedasticGaussianMLP",
            "ExpertGaussianMLP",
            "ExpertGaussian",
            "GaussianExpert",
        ]
        ExpertCls = None
        for n in cand:
            if hasattr(M, n):
                ExpertCls = getattr(M, n)
                break
        if ExpertCls is None:
            raise RuntimeError(f"Found {gauss_ckpt} but no Gaussian Expert class in risk_pipeline.models")

        gauss_sd = _torch_load_sd(gauss_ckpt, device=torch.device("cpu"))
        # infer dims
        in_dim, hidden, depth = _infer_mlp_dims_from_sd(gauss_sd)
        try:
            expert = ExpertCls(input_dim=in_dim, hidden_dim=hidden, depth=depth, dropout=0.0).to(device)
        except TypeError:
            # alternate signature
            expert = ExpertCls(in_dim, hidden=hidden, depth=depth, dropout=0.0).to(device)

        expert.load_state_dict(gauss_sd, strict=True)
        expert.eval()
        return gate, expert, "gauss"

    raise FileNotFoundError(f"Cannot find expert checkpoint in {run_dir}")


# ----------------------------
# Inference helpers
# ----------------------------
@torch.no_grad()
def infer_gate_probs(
    gate: torch.nn.Module,
    x_gate_raw: torch.Tensor,
    gate_mean: torch.Tensor,
    gate_std: torch.Tensor,
    *,
    device: torch.device,
    batch: int,
) -> np.ndarray:
    n = x_gate_raw.shape[0]
    out = np.empty((n,), dtype=np.float32)
    gate.eval()

    gate_mean = gate_mean.to(device)
    gate_std = gate_std.to(device)

    for i0 in range(0, n, batch):
        i1 = min(n, i0 + batch)
        xg_raw = x_gate_raw[i0:i1].to(device)
        xg = (xg_raw - gate_mean) / (gate_std + 1e-6)
        logits = gate(xg).reshape(-1)
        out[i0:i1] = torch.sigmoid(logits).float().detach().cpu().numpy()
    return out


@torch.no_grad()
def infer_flow_cdf_probs(
    flow,
    y_thr_std: torch.Tensor,          # [N]
    x_ctx_raw: torch.Tensor,          # [N, C] raw
    expert_mean: torch.Tensor,
    expert_std: torch.Tensor,
    *,
    drop_idx: Optional[torch.Tensor],
    device: torch.device,
    batch: int,
) -> np.ndarray:
    n = x_ctx_raw.shape[0]
    out = np.empty((n,), dtype=np.float32)
    flow.eval()

    expert_mean = expert_mean.to(device)
    expert_std = expert_std.to(device)
    y_thr_std = y_thr_std.to(device)

    if drop_idx is not None and drop_idx.numel() > 0:
        drop_idx = drop_idx.to(device)

    for i0 in range(0, n, batch):
        i1 = min(n, i0 + batch)
        xe_raw = x_ctx_raw[i0:i1].to(device)

        if drop_idx is not None and drop_idx.numel() > 0:
            xe_raw = xe_raw.clone()
            xe_raw[:, drop_idx] = expert_mean[drop_idx]

        xe = (xe_raw - expert_mean) / (expert_std + 1e-6)

        yb = y_thr_std[i0:i1]
        cdf = flow.cdf(yb, xe).reshape(-1)  # P(Y <= y_thr | x)
        out[i0:i1] = cdf.float().detach().cpu().numpy()
    return out


@torch.no_grad()
def eval_flow_nll_and_pit(
    flow,
    y_std: torch.Tensor,
    x_ctx_raw: torch.Tensor,
    expert_mean: torch.Tensor,
    expert_std: torch.Tensor,
    expert_mask: torch.Tensor,
    censored_mask: torch.Tensor,
    *,
    drop_idx: Optional[torch.Tensor],
    device: torch.device,
    batch: int,
) -> Tuple[float, float, float]:
    """
    Match training loss logic:
    - uncensored: -log_prob(y|x)
    - censored:   -log(1 - CDF(y)) using u = y_to_u -> survival = 0.5*erfc(u/sqrt(2))
    PIT computed on uncensored only: PIT = CDF(y|x)
    """
    n = x_ctx_raw.shape[0]
    flow.eval()

    expert_mean = expert_mean.to(device)
    expert_std = expert_std.to(device)
    y_std = y_std.to(device)
    expert_mask = expert_mask.to(device)
    censored_mask = censored_mask.to(device)

    if drop_idx is not None and drop_idx.numel() > 0:
        drop_idx = drop_idx.to(device)

    losses = []
    pits = []

    for i0 in range(0, n, batch):
        i1 = min(n, i0 + batch)
        m = (expert_mask[i0:i1] > 0.5)
        if int(m.sum().item()) < 2:
            continue

        xe_raw = x_ctx_raw[i0:i1].to(device)
        if drop_idx is not None and drop_idx.numel() > 0:
            xe_raw = xe_raw.clone()
            xe_raw[:, drop_idx] = expert_mean[drop_idx]

        xe = (xe_raw - expert_mean) / (expert_std + 1e-6)

        yb = y_std[i0:i1][m]
        xb = xe[m]
        is_c = (censored_mask[i0:i1][m] > 0.5)

        loss_val = torch.zeros_like(yb)

        # uncensored: -log_prob
        mask_u = ~is_c
        if mask_u.any():
            loss_val[mask_u] = -flow.log_prob(yb[mask_u], xb[mask_u])

            # PIT on uncensored
            pit = flow.cdf(yb[mask_u], xb[mask_u])
            pits.append(pit.detach().cpu())

        # censored: -log(survival)
        mask_c = is_c
        if mask_c.any():
            u, _ = flow.y_to_u(yb[mask_c], xb[mask_c])
            surv = 0.5 * torch.erfc(u / 1.41421356237)
            surv = torch.clamp(surv, min=1e-6)
            loss_val[mask_c] = -torch.log(surv)

        losses.append(loss_val.mean().detach().cpu())

    if not losses:
        return float("nan"), float("nan"), float("nan")

    nll = float(torch.stack(losses).mean().item())
    if pits:
        pit_all = torch.cat(pits, dim=0).numpy()
        return nll, float(np.mean(pit_all)), float(np.std(pit_all))
    return nll, float("nan"), float("nan")


# ----------------------------
# Main evaluation
# ----------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV with segment_id, y_soft, and features.")
    ap.add_argument("--run", action="append", required=True, help="Run directory (can pass multiple).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"], help="Which split to evaluate.")
    ap.add_argument("--ttc_floor", type=float, default=0.05)
    ap.add_argument("--ttc_cap", type=float, default=8.0)

    # event label definition
    ap.add_argument("--label_mode", default="ttc_sstar", choices=["ttc_sstar", "ttc_fixed", "gate_clean"])
    ap.add_argument("--sstar_mode", default="closing_speed", choices=["closing_speed", "ego_speed"])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--amax", type=float, default=6.0)
    ap.add_argument("--fixed_ttc", type=float, default=2.0)

    # recommended: avoid labeling "events" where there is no interaction edge
    ap.add_argument("--require_edges", action="store_true", default=True)
    ap.add_argument("--no_require_edges", dest="require_edges", action="store_false")

    # subgroup eval
    ap.add_argument(
        "--odd_group_prefix",
        action="append",
        default=[],
        help="Evaluate subgroup metrics for columns starting with this prefix. Example: c__odd_weather=",
    )

    # TTC bin diagnostic (NOT 'accuracy', just behavior)
    ap.add_argument("--ttc_bins", type=int, default=6)

    # performance / output
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--save_dir", default=None, help="If set, save outputs here under <save_dir>/<run_name>/ ...")
    ap.add_argument("--save_preds", action="store_true", help="Also save per-sample preds (.npz)")
    ap.add_argument("--no_save", action="store_true", help="Disable saving json/csv")

    args = ap.parse_args()

    df_all = pd.read_csv(args.csv)
    assert "segment_id" in df_all.columns, "CSV must contain segment_id"
    assert ("y_soft" in df_all.columns) or ("min_ttc_est" in df_all.columns), "CSV must contain y_soft or min_ttc_est"

    device = torch.device(args.device)

    # Import project modules
    from risk_pipeline.schema import load_schema
    from risk_pipeline.preprocess import load_preprocess_state, transform_dataframe
    from risk_pipeline.data import RiskCSVDataset

    all_results: Dict[str, dict] = {}

    for run in args.run:
        run_dir = Path(run)
        split = load_segment_split(run_dir)
        segs = split[args.split]
        df_split = pick_split_df(df_all, segs)

        # ensure frame_label exists (Dataset expects it)
        if "frame_label" not in df_split.columns:
            df_split["frame_label"] = np.arange(len(df_split), dtype=np.int64)

        # load schema/state
        _ = load_schema(str(run_dir / "feature_schema.json"))
        state = load_preprocess_state(str(run_dir / "preprocess_state.json"))

        # transform -> adds y_gate/y_expert/expert_mask (and keeps RAW features)
        df_t = transform_dataframe(df_split, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)

        # dataset (raw tensors)
        ds = RiskCSVDataset(df_t, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)

        # scaling tensors (train-based mean/std; binary/onehot are identity)
        scale = ds.get_scale_tensors("cpu")
        gate_mean, gate_std = scale["gate_mean"], scale["gate_std"]
        expert_mean, expert_std = scale["expert_mean"], scale["expert_std"]

        # expert fingerprint suppression indices
        drop_idx = sorted(set(getattr(state, "drop_context_idx_expert", []) or []))
        drop_idx_t = torch.tensor(drop_idx, dtype=torch.long) if len(drop_idx) else None

        # load models
        gate, expert, expert_type = load_models_for_run(run_dir, state, device=device)

        # labels
        y_gate_true = ds.tensors.y_gate.detach().cpu().numpy().astype(np.int64)
        y_event = build_event_label(
            df_t,
            ttc_floor=args.ttc_floor,
            ttc_cap=args.ttc_cap,
            label_mode=args.label_mode,
            sstar_mode=args.sstar_mode,
            tau=args.tau,
            amax=args.amax,
            fixed_ttc=args.fixed_ttc,
            require_edges=args.require_edges,
            y_gate_proxy=y_gate_true,
        )

        # threshold (TTC seconds -> standardized log TTC for expert CDF query)
        thr_ttc = build_event_threshold_ttc(
            df_t,
            ttc_cap=args.ttc_cap,
            label_mode=args.label_mode,
            sstar_mode=args.sstar_mode,
            tau=args.tau,
            amax=args.amax,
            fixed_ttc=args.fixed_ttc,
        )
        thr_ttc = np.clip(thr_ttc, args.ttc_floor, args.ttc_cap)
        thr_log = np.log(thr_ttc + 1e-9)
        mu_y = float(state.target_std.mu_y)
        sig_y = float(state.target_std.sigma_y)
        eps_y = float(getattr(state.target_std, "eps", 1e-6))
        y_thr_std = (thr_log - mu_y) / (sig_y + eps_y)
        y_thr_std_t = torch.tensor(y_thr_std, dtype=torch.float32)

        # inference: gate
        p_gate = infer_gate_probs(
            gate,
            ds.tensors.x_gate_raw,
            gate_mean,
            gate_std,
            device=device,
            batch=args.batch,
        )

        # inference: expert -> p_event
        if expert_type == "flow":
            p_event = infer_flow_cdf_probs(
                expert,
                y_thr_std_t,
                ds.tensors.x_expert_raw,
                expert_mean,
                expert_std,
                drop_idx=drop_idx_t,
                device=device,
                batch=args.batch,
            )

            # expert nll + PIT on expert training subset
            expert_nll, pit_mean, pit_std = eval_flow_nll_and_pit(
                expert,
                ds.tensors.y_expert,
                ds.tensors.x_expert_raw,
                expert_mean,
                expert_std,
                ds.tensors.expert_mask,
                ds.tensors.censored_mask,
                drop_idx=drop_idx_t,
                device=device,
                batch=args.batch,
            )
        else:
            # gaussian expert path (optional if your codebase has it)
            # We can't guarantee output format, so we keep these as NaN unless you standardize it.
            p_event = np.full((len(ds),), np.nan, dtype=np.float32)
            expert_nll, pit_mean, pit_std = float("nan"), float("nan"), float("nan")

        # final risk prob
        risk_prob = np.clip(p_gate * p_event, 0.0, 1.0)

        # metrics
        res = {
            "run": str(run_dir),
            "run_name": run_dir.name,
            "split": args.split,
            "N": int(len(df_t)),
            "pos_rate": float(np.mean(y_event)),

            # gate metrics (uses training gate label)
            "gate_pr_auc": _safe_prauc(y_gate_true, p_gate),
            "gate_roc_auc": _safe_auc(y_gate_true, p_gate),
            "gate_brier": brier(y_gate_true, p_gate),
            "gate_ece": ece(y_gate_true, p_gate),

            # risk metrics
            "risk_label_mode": args.label_mode,
            "risk_pr_auc": _safe_prauc(y_event, risk_prob),
            "risk_roc_auc": _safe_auc(y_event, risk_prob),
            "risk_brier": brier(y_event, risk_prob),
            "risk_ece": ece(y_event, risk_prob),

            # expert diagnostics (flow only)
            "expert_type": expert_type,
            "expert_nll": float(expert_nll),
            "pit_mean": float(pit_mean),
            "pit_std": float(pit_std),

            # summary stats
            "risk_mean": float(np.mean(risk_prob)),
            "risk_p95": float(np.quantile(risk_prob, 0.95)),
            "p_gate_mean": float(np.mean(p_gate)),
            "p_event_mean": float(np.mean(p_event)),
        }

        # subgroup metrics (one-hot columns)
        subgroup = {}
        for pref in args.odd_group_prefix:
            cols = [c for c in df_t.columns if c.startswith(pref)]
            if not cols:
                continue
            for c in cols:
                m = pd.to_numeric(df_t[c], errors="coerce").fillna(0.0).to_numpy(np.float64) > 0.5
                if int(np.sum(m)) < 100:
                    continue
                subgroup[c] = {
                    "N": int(np.sum(m)),
                    "risk_pr_auc": _safe_prauc(y_event[m], risk_prob[m]),
                    "risk_brier": brier(y_event[m], risk_prob[m]),
                    "risk_ece": ece(y_event[m], risk_prob[m]),
                    "risk_mean": float(np.mean(risk_prob[m])),
                    "pos_rate": float(np.mean(y_event[m])),
                }
        res["subgroup"] = subgroup

        # TTC-bin diagnostic
        ttc_clip = np.clip(compute_ttc(df_t, eps=1e-6), args.ttc_floor, args.ttc_cap)
        logttc = np.log(ttc_clip + 1e-9)
        bins = np.quantile(logttc, np.linspace(0, 1, args.ttc_bins + 1))
        bin_rows = []
        for i in range(args.ttc_bins):
            lo, hi = float(bins[i]), float(bins[i + 1])
            m = (logttc >= lo) & (logttc <= hi if i == args.ttc_bins - 1 else logttc < hi)
            if int(np.sum(m)) < 200:
                continue
            bin_rows.append(
                {
                    "bin": int(i),
                    "N": int(np.sum(m)),
                    "logttc_lo": lo,
                    "logttc_hi": hi,
                    "event_rate": float(np.mean(y_event[m])),
                    "risk_mean": float(np.mean(risk_prob[m])),
                    "p_gate_mean": float(np.mean(p_gate[m])),
                    "p_event_mean": float(np.mean(p_event[m])),
                }
            )
        res["ttc_bin_diag"] = bin_rows

        # save
        if not args.no_save:
            if args.save_dir:
                out_dir = Path(args.save_dir) / run_dir.name
            else:
                out_dir = run_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            out_json = out_dir / f"eval_{args.split}_{args.label_mode}.json"
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)

            # optional preds dump
            if args.save_preds:
                seg_id = df_t["segment_id"].astype(str).to_numpy()
                frame = pd.to_numeric(df_t["frame_label"], errors="coerce").fillna(-1).to_numpy(np.int64)
                out_npz = out_dir / f"preds_{args.split}_{args.label_mode}.npz"
                np.savez_compressed(
                    out_npz,
                    segment_id=seg_id,
                    frame_label=frame,
                    y_gate=y_gate_true.astype(np.int8),
                    y_event=y_event.astype(np.int8),
                    p_gate=p_gate.astype(np.float32),
                    p_event=p_event.astype(np.float32),
                    risk=risk_prob.astype(np.float32),
                )

        all_results[str(run_dir)] = res

        print(
            f"[OK] {run_dir.name} ({expert_type}) split={args.split} N={res['N']} pos={res['pos_rate']:.4f} "
            f"PR-AUC={res['risk_pr_auc']:.4f} ROC-AUC={res['risk_roc_auc']:.4f} "
            f"Brier={res['risk_brier']:.4f} risk_mean={res['risk_mean']:.4f} risk_p95={res['risk_p95']:.4f}"
        )
        if not args.no_save:
            if args.save_dir:
                print("saved:", str((Path(args.save_dir) / run_dir.name / f"eval_{args.split}_{args.label_mode}.json")))
            else:
                print("saved:", str(run_dir / f"eval_{args.split}_{args.label_mode}.json"))

    # summary table (printed)
    print("\n\n===== SUMMARY TABLE (risk_pr_auc / risk_brier / expert_nll) =====")
    rows = []
    for run, r in all_results.items():
        rows.append(
            {
                "run": run,
                "split": r["split"],
                "N": r["N"],
                "pos_rate": r["pos_rate"],
                "risk_pr_auc": r["risk_pr_auc"],
                "risk_roc_auc": r["risk_roc_auc"],
                "risk_brier": r["risk_brier"],
                "risk_ece": r["risk_ece"],
                "expert_nll": r["expert_nll"],
                "risk_mean": r["risk_mean"],
                "risk_p95": r["risk_p95"],
                "p_gate_mean": r["p_gate_mean"],
                "p_event_mean": r["p_event_mean"],
            }
        )
        print(
            f"- {run}: risk_pr_auc={r['risk_pr_auc']:.4f}  risk_brier={r['risk_brier']:.4f}  "
            f"expert_nll={r['expert_nll']}  expert={r['expert_type']}"
        )

    # save summary as well
    if rows and not args.no_save:
        df_sum = pd.DataFrame(rows)
        if args.save_dir:
            base = Path(args.save_dir)
        else:
            base = Path(args.run[0])
        base.mkdir(parents=True, exist_ok=True)
        df_sum.to_csv(base / f"eval_summary_{args.split}_{args.label_mode}.csv", index=False)
        (base / f"eval_summary_{args.split}_{args.label_mode}.json").write_text(df_sum.to_json(orient="records", indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

