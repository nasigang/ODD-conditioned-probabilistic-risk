# ============================================================
# diagnose_tau_gate_flow_antigravity_final5.py
#
# VERSION: antigravity_final5
#
# Fixes your recurring error in _plot_ttc_pdf_low_vs_high():
#   RuntimeError: Size does not match at dimension 0 expected index [240, 1] ...
#
# Root cause:
#   When plotting PDF, y_std is a grid (G=240) but cond_low/cond_high is a single
#   condition (batch=1). Flow expects y and cond to have matching batch size.
#
# This file:
#   - repeats cond to match grid batch size BEFORE calling flow.log_prob
#   - adds --skip_flow_pdf (if you want to bypass the plot entirely)
#   - prints version banner so you can verify the correct file is running
# ============================================================

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from risk_pipeline.preprocess import load_preprocess_state, transform_dataframe
from risk_pipeline.data import RiskCSVDataset
from risk_pipeline.models import GateMLP, ConditionalSpline1DFlow
from risk_pipeline.train import _make_flow_cond
from risk_pipeline.risk import RiskConfig, compute_risk


def safe_torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


# -------------------------
# Metrics (no sklearn dependency)
# -------------------------

def pr_auc_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")

    order = np.argsort(-y_score)
    y = y_true[order]
    P = float(np.sum(y == 1))
    if P <= 0:
        return float("nan")

    tp = 0.0
    fp = 0.0
    ap = 0.0
    prev_recall = 0.0

    for i in range(len(y)):
        if y[i] == 1:
            tp += 1.0
        else:
            fp += 1.0
        recall = tp / P
        precision = tp / max(1.0, tp + fp)
        if y[i] == 1:
            ap += (recall - prev_recall) * precision
            prev_recall = recall
    return float(ap)


def roc_auc_trapezoid(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")

    order = np.argsort(-y_score)
    y = y_true[order]
    P = float(np.sum(y == 1))
    N = float(np.sum(y == 0))
    if P <= 0 or N <= 0:
        return float("nan")

    tps = 0.0
    fps = 0.0
    fpr = [0.0]
    tpr = [0.0]

    for i in range(len(y)):
        if y[i] == 1:
            tps += 1.0
        else:
            fps += 1.0
        fpr.append(fps / N)
        tpr.append(tps / P)

    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) * 0.5
    return float(auc)


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    y_true = y_true.astype(np.float64)
    y_prob = np.clip(y_prob.astype(np.float64), 0.0, 1.0)
    if y_true.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(m):
            continue
        acc = float(np.mean(y_true[m]))
        conf = float(np.mean(y_prob[m]))
        out += float(np.mean(m)) * abs(acc - conf)
    return float(out)


# -------------------------
# Helpers
# -------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _load_split_map(run_dir: str) -> Optional[Dict]:
    p = os.path.join(run_dir, "segment_split.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _filter_df_by_split(df: pd.DataFrame, split_map: Optional[Dict], split: str) -> pd.DataFrame:
    if split_map is None or split == "all":
        return df
    segs = set(map(str, split_map.get(split, [])))
    if not segs or "segment_id" not in df.columns:
        return df
    return df[df["segment_id"].astype(str).isin(segs)].copy()


def _infer_mlp_dims_from_sd(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    w0 = None
    for k, v in sd.items():
        if k.endswith("net.0.weight"):
            w0 = v
            break
    if w0 is None:
        cand = [(k, v) for k, v in sd.items() if k.startswith("net.") and k.endswith(".weight")]
        if not cand:
            raise RuntimeError("Could not infer GateMLP dims: no net.*.weight keys.")
        cand.sort(key=lambda kv: int(kv[0].split(".")[1]))
        w0 = cand[0][1]

    hidden = int(w0.shape[0])
    in_dim = int(w0.shape[1])

    idxs = []
    for k in sd.keys():
        if k.startswith("net.") and k.endswith(".weight"):
            try:
                idxs.append(int(k.split(".")[1]))
            except Exception:
                pass
    num_linears = len(set(idxs))
    depth = max(1, num_linears - 1)
    return in_dim, hidden, depth


def _infer_flow_arch_and_hparams_from_sd(sd: Dict[str, torch.Tensor]) -> Tuple[str, int, int, int, int, int]:
    # concat
    if any(k.startswith("net.") for k in sd.keys()):
        cand = [(k, v) for k, v in sd.items() if k.startswith("net.") and k.endswith(".weight")]
        if not cand:
            raise RuntimeError("Could not infer Flow (concat): no net.*.weight keys.")
        cand.sort(key=lambda kv: int(kv[0].split(".")[1]))

        w0 = cand[0][1]
        hidden = int(w0.shape[0])
        cond_dim = int(w0.shape[1])

        last_w = cand[-1][1]
        out_dim = int(last_w.shape[0])
        if (out_dim - 1) % 3 != 0:
            raise RuntimeError(f"Could not infer num_bins from out_dim={out_dim} (expected 3*K+1).")
        num_bins = int((out_dim - 1) // 3)

        idxs = []
        for k, _ in cand:
            try:
                idxs.append(int(k.split(".")[1]))
            except Exception:
                pass
        depth = max(1, len(set(idxs)) - 1)
        return "concat", hidden, depth, num_bins, cond_dim, 0

    # film
    if "trunk.0.weight" not in sd or "film.0.weight" not in sd or "head.weight" not in sd:
        raise RuntimeError("Could not infer Flow (film): expected trunk.0.weight, film.0.weight, head.weight in state_dict.")

    trunk0 = sd["trunk.0.weight"]
    film0 = sd["film.0.weight"]
    head = sd["head.weight"]

    hidden = int(trunk0.shape[0])
    x_dim = int(trunk0.shape[1])
    c_dim = int(film0.shape[1])
    out_dim = int(head.shape[0])
    if (out_dim - 1) % 3 != 0:
        raise RuntimeError(f"Could not infer num_bins from out_dim={out_dim} (expected 3*K+1).")
    num_bins = int((out_dim - 1) // 3)

    idxs = []
    for k in sd.keys():
        if k.startswith("trunk.") and k.endswith(".weight"):
            try:
                idxs.append(int(k.split(".")[1]))
            except Exception:
                pass
    depth = max(1, len(set(idxs)))
    return "film", hidden, depth, num_bins, x_dim, c_dim


def _load_models_for_run(run_dir: Path, state, device: torch.device):
    gate_ckpt = run_dir / "gate.pt"
    if not gate_ckpt.exists():
        raise FileNotFoundError(f"Missing {gate_ckpt}")
    gate_sd = safe_torch_load(str(gate_ckpt), map_location="cpu")
    in_dim, hidden, depth = _infer_mlp_dims_from_sd(gate_sd)
    gate = GateMLP(in_dim=in_dim, hidden=hidden, depth=depth, dropout=0.0).to(device)
    gate.load_state_dict(gate_sd, strict=True)
    gate.eval()

    flow_ckpt = run_dir / "expert_flow.pt"
    if not flow_ckpt.exists():
        legacy = run_dir / "expert.pt"
        if legacy.exists():
            flow_ckpt = legacy
    if not flow_ckpt.exists():
        raise FileNotFoundError(f"Missing {flow_ckpt}")
    flow_sd = safe_torch_load(str(flow_ckpt), map_location="cpu")

    cond_mode, f_hidden, f_depth, num_bins, inf_x, inf_c = _infer_flow_arch_and_hparams_from_sd(flow_sd)

    expert_cols = state.schema.x_expert_cols_in_order()
    cond_dim = int(len(expert_cols))

    if cond_mode == "film":
        if getattr(state, "flow_x_idx", None) and getattr(state, "flow_c_idx", None):
            x_dim = int(len(state.flow_x_idx))
            c_dim = int(len(state.flow_c_idx))
        elif inf_x > 0 and inf_c > 0 and (inf_x + inf_c == cond_dim):
            x_dim, c_dim = int(inf_x), int(inf_c)
        else:
            raise RuntimeError("FiLM flow detected but could not determine x_dim/c_dim. Ensure preprocess_state has flow_x_idx/flow_c_idx.")

        flow = ConditionalSpline1DFlow(
            x_dim=x_dim,
            c_dim=c_dim,
            cond_mode="film",
            num_bins=num_bins,
            hidden=f_hidden,
            depth=f_depth,
            dropout=0.0,
        ).to(device)
    else:
        flow = ConditionalSpline1DFlow(
            cond_dim=cond_dim,
            cond_mode="concat",
            num_bins=num_bins,
            hidden=f_hidden,
            depth=f_depth,
            dropout=0.0,
        ).to(device)

    flow.load_state_dict(flow_sd, strict=True)
    flow.eval()
    return gate, flow


def _make_event_label(
    y_expert_std: torch.Tensor,
    censored_mask: torch.Tensor,
    raw_closing_speed_mps: torch.Tensor,
    target_mu: float,
    target_sigma: float,
    tau: float,
    amax: float,
    ttc_floor: float,
    ttc_cap: float,
) -> torch.Tensor:
    y_log = y_expert_std * (target_sigma + 1e-6) + target_mu
    ttc = torch.exp(y_log).clamp(min=ttc_floor, max=ttc_cap)

    v = torch.clamp(raw_closing_speed_mps, min=0.0)
    s = torch.clamp(tau + v / max(amax, 1e-6), min=ttc_floor, max=ttc_cap)

    is_cens = (censored_mask > 0.5)
    return ((ttc <= s) & (~is_cens)).float()


def _choose_gate_thr_for_recall(p_gate: np.ndarray, y_true: np.ndarray, thresholds: List[float], target_recall: float) -> float:
    pos = (y_true > 0.5)
    P = int(np.sum(pos))
    if P <= 0:
        return float("nan")
    best = None
    for thr in sorted(thresholds):
        pred = (p_gate >= thr)
        tp = int(np.sum(pred & pos))
        rec = tp / max(1, P)
        if rec >= target_recall:
            best = thr
    if best is None:
        best = float(min(thresholds)) if thresholds else float("nan")
    return float(best)


def _plot_pit(u: np.ndarray, out_path: str) -> Dict[str, float]:
    u = np.clip(u, 0.0, 1.0)
    plt.figure(figsize=(5, 4))
    plt.hist(u, bins=20, range=(0, 1), density=True)
    plt.axhline(1.0, linestyle="--")
    plt.title("PIT histogram (uniform = calibrated density)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return {"u_mean": float(np.mean(u)) if u.size else float("nan"),
            "u_std": float(np.std(u)) if u.size else float("nan")}



def _save_calibration_curve(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    out_png: str,
    out_csv: str,
    n_bins: int = 15,
) -> Dict[str, float]:
    """
    Reliability diagram for predicted probability p_pred vs empirical event rate.
    Returns summary stats including ECE (same binning), Brier.
    """
    y_true = y_true.astype(np.float64).reshape(-1)
    p_pred = np.clip(p_pred.astype(np.float64).reshape(-1), 0.0, 1.0)
    if y_true.size == 0:
        return {"ece": float("nan"), "brier": float("nan")}

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p_pred >= lo) & (p_pred < hi) if i < n_bins - 1 else (p_pred >= lo) & (p_pred <= hi)
        if not np.any(m):
            rows.append({"bin": i, "lo": lo, "hi": hi, "n": 0, "p_mean": float("nan"), "y_rate": float("nan")})
            continue
        p_mean = float(np.mean(p_pred[m]))
        y_rate = float(np.mean(y_true[m]))
        n = int(np.sum(m))
        rows.append({"bin": i, "lo": lo, "hi": hi, "n": n, "p_mean": p_mean, "y_rate": y_rate})
        ece_val += (n / y_true.size) * abs(y_rate - p_mean)

    brier_val = float(np.mean((p_pred - y_true) ** 2))

    # Save CSV
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Plot reliability
    xs = [r["p_mean"] for r in rows if r["n"] > 0]
    ys = [r["y_rate"] for r in rows if r["n"] > 0]
    ns = [r["n"] for r in rows if r["n"] > 0]

    plt.figure(figsize=(5, 4))
    if len(xs) > 0:
        plt.plot(xs, ys, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted p_event")
    plt.ylabel("Empirical event rate")
    plt.title(f"p_event calibration (bins={n_bins})")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    return {"ece": float(ece_val), "brier": float(brier_val)}


def _ks_cdf_u_vs_uniform(u: np.ndarray) -> float:
    """KS distance between empirical CDF of u and Uniform(0,1) CDF (which is u)."""
    u = np.clip(u.astype(np.float64).reshape(-1), 0.0, 1.0)
    if u.size == 0:
        return float("nan")
    us = np.sort(u)
    ecdf = (np.arange(1, us.size + 1, dtype=np.float64) / us.size)
    ks = float(np.max(np.abs(ecdf - us)))
    return ks


def _save_ks_plot_u_vs_uniform(u: np.ndarray, out_png: str) -> Dict[str, float]:
    u = np.clip(u.astype(np.float64).reshape(-1), 0.0, 1.0)
    if u.size == 0:
        plt.figure(figsize=(5, 4))
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("Empirical CDF(u) vs Uniform CDF (empty)")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        return {"ks": float("nan"), "n": 0}

    us = np.sort(u)
    ecdf = (np.arange(1, us.size + 1, dtype=np.float64) / us.size)
    ks = float(np.max(np.abs(ecdf - us)))

    plt.figure(figsize=(5, 4))
    plt.plot(us, ecdf, label="Empirical CDF(u)")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Uniform CDF")
    plt.xlabel("u = F_model(y|cond)")
    plt.ylabel("CDF")
    plt.title(f"KS(CDF_u, Uniform) = {ks:.4f} (n={us.size})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    return {"ks": float(ks), "n": int(us.size), "u_mean": float(np.mean(u)), "u_std": float(np.std(u))}


def _repeat_cond_for_grid(cond, n: int):
    """Repeat a single-sample condition to match a grid of size n."""
    if isinstance(cond, (tuple, list)):
        x, c = cond
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if c.dim() == 1:
            c = c.unsqueeze(0)
        return (x.repeat(n, 1), c.repeat(n, 1))
    else:
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        return cond.repeat(n, 1)


def _plot_ttc_pdf_low_vs_high(
    flow: ConditionalSpline1DFlow,
    cond_low,
    cond_high,
    target_mu: float,
    target_sigma: float,
    ttc_floor: float,
    ttc_cap: float,
    out_path: str,
) -> None:
    # Grid of TTC, converted to standardized log(TTC)
    device = next(flow.parameters()).device
    t = torch.linspace(float(ttc_floor), float(ttc_cap), 240, device=device)
    y_log = torch.log(t + 1e-9)
    y_std = (y_log - target_mu) / (target_sigma + 1e-6)  # (G,)

    # IMPORTANT: match batch sizes (G) for y_std and cond
    G = int(y_std.numel())
    cond_low_g = _repeat_cond_for_grid(cond_low, G)
    cond_high_g = _repeat_cond_for_grid(cond_high, G)

    with torch.no_grad():
        # >>> MUST pass cond_low_g / cond_high_g (NOT cond_low/cond_high)
        logp_low = flow.log_prob(y_std, cond_low_g)
        logp_high = flow.log_prob(y_std, cond_high_g)
        pz_low = logp_low.exp().detach().cpu().numpy().reshape(-1)
        pz_high = logp_high.exp().detach().cpu().numpy().reshape(-1)

    # Convert PDF over standardized log(TTC) to approximate PDF over TTC
    sigma = float(target_sigma + 1e-6)
    t_np = t.detach().cpu().numpy().reshape(-1)
    jac = 1.0 / (t_np * sigma)
    pt_low = pz_low * jac
    pt_high = pz_high * jac

    plt.figure(figsize=(5, 4))
    plt.plot(t_np, pt_low, label="low-TTC cond")
    plt.plot(t_np, pt_high, label="high-TTC cond")
    plt.xlabel("TTC (s)")
    plt.ylabel("PDF (approx)")
    plt.title("Flow implied TTC PDF (2 conditions)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


@torch.no_grad()
def main():
    print("[diagnose] VERSION: antigravity_final5")

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    ap.add_argument("--ttc_floor", type=float, default=0.05)
    ap.add_argument("--ttc_cap", type=float, default=10.0)
    ap.add_argument("--label_mode", default="ttc_sstar")
    ap.add_argument("--sstar_mode", default="closing_speed")
    ap.add_argument("--amax", type=float, default=6.0)
    ap.add_argument("--tau_sweep", default="0.5")
    ap.add_argument("--gate_thresholds", default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5")
    ap.add_argument("--gate_target_recall", type=float, default=0.90)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--flow_diag_dir", required=True)
    ap.add_argument("--save_preds", action="store_true")
    ap.add_argument("--calib_bins", type=int, default=15, help="Number of bins for p_event calibration curve")
    ap.add_argument("--calib_on_gated_only", action="store_true", help="If set, compute calibration only on gated subset (p_gate>=thr)")
    ap.add_argument("--ks_on_gated_only", action="store_true", help="If set, compute PIT-KS only on gated subset (p_gate>=thr)")
    ap.add_argument("--skip_flow_pdf", action="store_true", help="Skip TTC PDF plot (only PIT)")
    ap.add_argument("--expert_keep_c_cont", action="store_true")
    ap.add_argument("--expert_drop_all_context", action="store_true")
    ap.add_argument("--expert_drop_feature_regex", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run)
    _ensure_dir(args.out_dir)
    _ensure_dir(args.flow_diag_dir)

    taus = [float(x) for x in args.tau_sweep.split(",") if str(x).strip()]
    gate_thresholds = [float(x) for x in args.gate_thresholds.split(",") if str(x).strip()]

    # preprocess_state.json file path
    state_path = run_dir / "preprocess_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing preprocess_state.json: {state_path}")
    state = load_preprocess_state(str(state_path))

    # load CSV + split
    header = pd.read_csv(args.csv, nrows=0)
    dtypes = {c: np.float32 for c in header.columns if c != "segment_id"}
    dtypes["segment_id"] = str
    df = pd.read_csv(args.csv, dtype=dtypes)

    split_map = _load_split_map(str(run_dir))
    df = _filter_df_by_split(df, split_map, args.split)

    if args.max_rows and args.max_rows > 0 and len(df) > args.max_rows:
        df = df.sample(n=int(args.max_rows), random_state=0).copy()

    df_t = transform_dataframe(df, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)
    ds = RiskCSVDataset(df_t, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)

    device = torch.device(args.device)
    gate, flow = _load_models_for_run(run_dir, state, device=device)

    scale = ds.get_scale_tensors(device=device)
    gate_mean, gate_std = scale["gate_mean"], scale["gate_std"]
    expert_mean, expert_std = scale["expert_mean"], scale["expert_std"]

    xg_raw = ds.tensors.x_gate_raw.to(device)
    xe_raw = ds.tensors.x_expert_raw.to(device)

    # y_expert must be 1D (N,)
    y_expert = ds.tensors.y_expert.to(device).reshape(-1)
    censored_mask = ds.tensors.censored_mask.to(device).reshape(-1)
    v_close = ds.tensors.raw_closing_speed_mps.to(device).reshape(-1)

    # scale gate
    xg = (xg_raw - gate_mean) / (gate_std + 1e-6)

    # apply expert drops (must match train)
    expert_colnames = ds.get_x_expert_colnames()
    drop_idx: List[int] = []

    if not args.expert_keep_c_cont:
        drop_idx.extend(
            i for i, c in enumerate(expert_colnames)
            if c.startswith("c__") and ("=" not in c) and (not c.startswith("c__has_"))
        )

    if getattr(state, "drop_context_idx_expert", None):
        drop_idx.extend(list(state.drop_context_idx_expert))

    if args.expert_drop_all_context:
        drop_idx.extend([i for i, c in enumerate(expert_colnames) if c.startswith("c__")])

    if args.expert_drop_feature_regex:
        import re
        regs = [re.compile(p.strip()) for p in args.expert_drop_feature_regex.split(",") if p.strip()]
        for i, c in enumerate(expert_colnames):
            if any(r.search(c) for r in regs):
                drop_idx.append(i)

    drop_idx = sorted(set(int(i) for i in drop_idx))
    if drop_idx:
        idx_t = torch.tensor(drop_idx, dtype=torch.long, device=device)
        xe_raw = xe_raw.clone()
        xe_raw[:, idx_t] = expert_mean[idx_t]

    xe = (xe_raw - expert_mean) / (expert_std + 1e-6)

    # cond for flow (concat or film)
    cond = _make_flow_cond(
        xe,
        flow,
        flow_x_idx=getattr(state, "flow_x_idx", None),
        flow_c_idx=getattr(state, "flow_c_idx", None),
    )

    # gate forward
    gate_logits = gate(xg)
    p_gate = torch.sigmoid(gate_logits).reshape(-1)

    # -------------------------
    # Flow diagnostics (tau-independent)
    # -------------------------
    diag_subdir = os.path.join(args.flow_diag_dir, run_dir.name, args.split)
    _ensure_dir(diag_subdir)

    u, _ = flow.y_to_u(y_expert, cond)
    u_np = u.detach().cpu().numpy().reshape(-1)

    pit_png = os.path.join(diag_subdir, "pit_hist.png")
    pit_stats = _plot_pit(u_np, pit_png)
    with open(os.path.join(diag_subdir, "pit_stats.json"), "w", encoding="utf-8") as f:
        json.dump(pit_stats, f, indent=2)

    if not args.skip_flow_pdf:
        y_np = y_expert.detach().cpu().numpy()
        low_i = int(np.argmin(y_np))
        high_i = int(np.argmax(y_np))

        if isinstance(cond, (tuple, list)):
            cond_low = (cond[0][low_i:low_i+1], cond[1][low_i:low_i+1])
            cond_high = (cond[0][high_i:high_i+1], cond[1][high_i:high_i+1])
        else:
            cond_low = cond[low_i:low_i+1]
            cond_high = cond[high_i:high_i+1]

        pdf_png = os.path.join(diag_subdir, "ttc_pdf_low_vs_high.png")
        _plot_ttc_pdf_low_vs_high(
            flow, cond_low, cond_high,
            target_mu=float(state.target_std.mu_y),
            target_sigma=float(state.target_std.sigma_y),
            ttc_floor=float(args.ttc_floor),
            ttc_cap=float(args.ttc_cap),
            out_path=pdf_png,
        )

    # -------------------------
    # Tau sweep
    # -------------------------
    rows = []

    for tau in taus:
        cfg = RiskConfig(tau=float(tau), a_max=float(args.amax), ttc_floor=float(args.ttc_floor), ttc_cap=float(args.ttc_cap))

        risk, p_gate_out = compute_risk(
            gate_logits=gate_logits,
            expert_flow=flow,
            x_expert_scaled=cond,
            raw_closing_speed_mps=v_close,
            target_std=state.target_std,
            cfg=cfg,
        )
        risk = risk.reshape(-1)
        p_gate_out = p_gate_out.reshape(-1)

        v = torch.clamp(v_close, min=0.0)
        s = torch.clamp(cfg.tau + v / max(cfg.a_max, 1e-6), min=cfg.ttc_floor, max=cfg.ttc_cap)
        y_thr = torch.log(s + 1e-9)  # 1D
        y_thr_std = (y_thr - float(state.target_std.mu_y)) / (float(state.target_std.sigma_y) + 1e-6)
        p_event = flow.cdf(y_thr_std, cond).reshape(-1)

        y_true = _make_event_label(
            y_expert_std=y_expert,
            censored_mask=censored_mask,
            raw_closing_speed_mps=v_close,
            target_mu=float(state.target_std.mu_y),
            target_sigma=float(state.target_std.sigma_y),
            tau=float(tau),
            amax=float(args.amax),
            ttc_floor=float(args.ttc_floor),
            ttc_cap=float(args.ttc_cap),
        ).reshape(-1)

        y_true_np = y_true.detach().cpu().numpy().astype(np.int64)
        risk_np = risk.detach().cpu().numpy()
        p_gate_np = p_gate_out.detach().cpu().numpy()
        p_event_np = p_event.detach().cpu().numpy()

        pos_rate = float(np.mean(y_true_np)) if y_true_np.size else float("nan")

        pr = pr_auc_average_precision(y_true_np, risk_np)
        roc = roc_auc_trapezoid(y_true_np, risk_np)
        brier = float(np.mean((risk_np - y_true_np) ** 2)) if y_true_np.size else float("nan")
        e = ece(y_true_np, risk_np, n_bins=15)

        gate_thr = _choose_gate_thr_for_recall(p_gate_np, y_true_np, gate_thresholds, float(args.gate_target_recall))

        out = {
            "run_name": run_dir.name,
            "split": args.split,
            "tau": float(tau),
            "amax": float(args.amax),
            "N": int(y_true_np.size),
            "pos_rate": float(pos_rate),
            "risk_pr_auc": float(pr),
            "risk_roc_auc": float(roc),
            "risk_brier": float(brier),
            "risk_ece": float(e),
            "p_gate_mean": float(np.mean(p_gate_np)) if p_gate_np.size else float("nan"),
            "p_event_mean": float(np.mean(p_event_np)) if p_event_np.size else float("nan"),
            "gate_thr_for_target_recall": float(gate_thr),
        }

        # ---- tau-specific diagnostics (p_event calibration + KS) ----
        # Define gating mask using target-recall threshold (tau-dependent because y_true depends on tau)
        if np.isfinite(gate_thr):
            gated_mask = (p_gate_np >= gate_thr)
        else:
            gated_mask = np.ones_like(p_gate_np, dtype=bool)

        # Calibration curve target: p_event vs y_true (optionally on gated subset only)
        if args.calib_on_gated_only:
            y_cal = y_true_np[gated_mask]
            p_cal = p_event_np[gated_mask]
        else:
            y_cal = y_true_np
            p_cal = p_event_np

        tau_dir = os.path.join(diag_subdir, f"tau_{tau}")
        _ensure_dir(tau_dir)

        calib_png = os.path.join(tau_dir, "p_event_calibration.png")
        calib_csv = os.path.join(tau_dir, "p_event_calibration.csv")
        calib_stats = _save_calibration_curve(y_cal, p_cal, calib_png, calib_csv, n_bins=int(args.calib_bins))

        # KS distance: empirical CDF of PIT u vs Uniform (optionally on gated subset only)
        if args.ks_on_gated_only:
            u_use = u_np[gated_mask]
        else:
            u_use = u_np

        ks_png = os.path.join(tau_dir, "pit_ks_cdf.png")
        ks_stats = _save_ks_plot_u_vs_uniform(u_use, ks_png)

        # Attach these tau diagnostics to the summary json
        out["p_event_calib_ece"] = float(calib_stats.get("ece", float("nan")))
        out["p_event_calib_brier"] = float(calib_stats.get("brier", float("nan")))
        out["pit_ks_u_uniform"] = float(ks_stats.get("ks", float("nan")))
        out["pit_ks_n"] = int(ks_stats.get("n", 0))

        out_path = os.path.join(args.out_dir, f"{run_dir.name}_diag_{args.split}_tau{tau}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        if args.save_preds:
            pred_df = pd.DataFrame({
                "segment_id": ds.tensors.segment_ids,
                "frame_label": ds.tensors.frame_label.cpu().numpy(),
                "y_event": y_true_np,
                "p_gate": p_gate_np,
                "p_event": p_event_np,
                "risk": risk_np,
                "raw_closing_speed_mps": ds.tensors.raw_closing_speed_mps.cpu().numpy(),
                "censored": (ds.tensors.censored_mask.cpu().numpy() > 0.5).astype(np.int64),
            })
            pred_path = os.path.join(args.out_dir, f"preds_{run_dir.name}_{args.split}_tau{tau}.csv")
            pred_df.to_csv(pred_path, index=False)

        rows.append(out)

        print(
            f"[OK] {run_dir.name} split={args.split} tau={tau} "
            f"pos={pos_rate:.4f} PR-AUC={pr:.4f} ROC-AUC={roc:.4f} Brier={brier:.4f} "
            f"p_gate_mean={out['p_gate_mean']:.3f} p_event_mean={out['p_event_mean']:.3f} "
            f"gate_thr@recall{args.gate_target_recall:.2f}={gate_thr}"
        )

    if rows:
        s = pd.DataFrame(rows)
        s_path = os.path.join(args.out_dir, f"summary_{run_dir.name}_{args.split}.csv")
        s.to_csv(s_path, index=False)
        print("Saved summary:", s_path)


if __name__ == "__main__":
    main()
