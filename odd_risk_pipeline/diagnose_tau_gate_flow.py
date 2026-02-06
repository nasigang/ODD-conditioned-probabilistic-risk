# ============================================================
# diagnose_tau_gate_flow.py  (ITSC submission patch)
#
# Key changes for ITSC-ready reporting:
#   1) Gate pass/budget is defined ONLY via target-recall threshold (--gate_target_recall).
#      We intentionally avoid reporting p_gate>=0.5 pass rates.
#   2) PIT is unified to censoring-aware Randomized PIT (CDF-based) and exported as:
#        - pit_hist.png
#        - pit_stats.json
#      (latent-u PIT is NOT used for submission)
#   3) Density sensitivity (Strategy B) is reported with 95% CI:
#        - density_sensitivity.csv / .png
#   4) Reliability diagram includes Wilson 95% CI (helps explain sparsity in high-risk bins).
#   5) Optional multivariate proxy regression to mitigate proxy inconsistency:
#        - sigma_proxy_regression.csv
# ============================================================

from __future__ import annotations

import os
import json
import argparse
import math
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
# Flow sampling uncertainty (Strategy A)
# -------------------------

@torch.no_grad()
def infer_flow_sigma_log(
    flow: ConditionalSpline1DFlow,
    cond,
    *,
    target_mu: float,
    target_sigma: float,
    num_samples: int = 512,
    batch: int = 1024,
    device: torch.device,
) -> np.ndarray:
    """Per-sample uncertainty sigma in logTTC space via sampling.

    Returns
    -------
    sigma_log : np.ndarray, shape [N]
        std over sampled logTTC.
    """

    N = int(cond[0].shape[0]) if isinstance(cond, (tuple, list)) else int(cond.shape[0])
    S = int(num_samples)
    out = np.zeros((N,), dtype=np.float32)

    mu = float(target_mu)
    sig = float(target_sigma)

    for i0 in range(0, N, int(batch)):
        i1 = min(N, i0 + int(batch))
        if isinstance(cond, (tuple, list)):
            c_b = (cond[0][i0:i1], cond[1][i0:i1])
        else:
            c_b = cond[i0:i1]

        y_s = flow.sample(c_b, num_samples=S)  # [B,S] in standardized logTTC
        log_ttc_s = y_s * (sig + 1e-6) + mu
        sigma_log_b = torch.std(log_ttc_s, dim=1, unbiased=False)
        out[i0:i1] = sigma_log_b.detach().cpu().numpy().astype(np.float32)

    return out


# -------------------------
# Subgroup splitting (Strategy B)
# -------------------------

def _make_bins(values: np.ndarray, method: str) -> List[Tuple[str, np.ndarray]]:
    v = values.astype(np.float64)
    finite = np.isfinite(v)
    if finite.sum() < 20:
        return [("all", finite.copy())]

    if method == "median":
        thr = float(np.nanmedian(v))
        lo = finite & (v <= thr)
        hi = finite & (v > thr)
        if lo.sum() == 0 or hi.sum() == 0:
            return [("all", finite.copy())]
        return [("low", lo), ("high", hi)]

    # quantile3
    q1, q2 = np.nanquantile(v, [1.0 / 3.0, 2.0 / 3.0])
    q1, q2 = float(q1), float(q2)
    if not np.isfinite(q1) or not np.isfinite(q2) or q1 >= q2:
        # fallback to median split if quantiles are degenerate
        thr = float(np.nanmedian(v))
        lo = finite & (v <= thr)
        hi = finite & (v > thr)
        if lo.sum() == 0 or hi.sum() == 0:
            return [("all", finite.copy())]
        return [("low", lo), ("high", hi)]

    low = finite & (v <= q1)
    mid = finite & (v > q1) & (v <= q2)
    high = finite & (v > q2)
    # guard: if any bucket is empty, fallback to median
    if low.sum() == 0 or mid.sum() == 0 or high.sum() == 0:
        thr = float(np.nanmedian(v))
        lo = finite & (v <= thr)
        hi = finite & (v > thr)
        if lo.sum() == 0 or hi.sum() == 0:
            return [("all", finite.copy())]
        return [("low", lo), ("high", hi)]
    return [("low", low), ("mid", mid), ("high", high)]



def _mean_diff_ci(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Return mean(a)-mean(b) with a normal-approx 95% CI (Welch-style SE)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    na = int(a.size)
    nb = int(b.size)
    if na < 2 or nb < 2:
        return {
            "na": float(na), "nb": float(nb),
            "mean_a": float(np.nanmean(a)) if na else float("nan"),
            "mean_b": float(np.nanmean(b)) if nb else float("nan"),
            "diff": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    ma = float(np.mean(a))
    mb = float(np.mean(b))
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    se = math.sqrt(va / na + vb / nb)
    z = 1.96
    diff = ma - mb
    return {
        "na": float(na), "nb": float(nb),
        "mean_a": float(ma), "mean_b": float(mb),
        "diff": float(diff),
        "ci_low": float(diff - z * se),
        "ci_high": float(diff + z * se),
    }
def _get_feature_array(ds: RiskCSVDataset, feature: str) -> np.ndarray:
    """Fetch raw feature values from dataset tensors by column name."""
    g_cols = ds.get_x_gate_colnames()
    if feature in g_cols:
        j = g_cols.index(feature)
        return ds.tensors.x_gate_raw[:, j].cpu().numpy().astype(np.float32)

    e_cols = ds.get_x_expert_colnames()
    if feature in e_cols:
        j = e_cols.index(feature)
        return ds.tensors.x_expert_raw[:, j].cpu().numpy().astype(np.float32)

    raise KeyError(f"Feature not found in dataset tensors: {feature}")


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
    """Return the *exact* gate threshold that hits the requested positive recall.

    We define gated = (p_gate >= thr). Among positives, recall(thr) = P[p_gate>=thr | y=1].
    The smallest thr that achieves recall >= target_recall is the (1-target_recall) quantile
    of positive scores (with a conservative ceil to guarantee recall).

    Notes
    -----
    - We intentionally IGNORE the `thresholds` grid (kept only for backward CLI compatibility).
    - This avoids the common anti-pattern of reporting p_gate>=0.5 pass rates.
    """
    pos = (y_true > 0.5)
    P = int(np.sum(pos))
    if P <= 0:
        return float("nan")

    pos_scores = np.asarray(p_gate[pos], dtype=np.float64)
    # Guarantee recall >= target_recall with a conservative ceil.
    k = int(np.ceil(float(target_recall) * float(P)))
    k = max(1, min(k, P))
    pos_sorted = np.sort(pos_scores)  # ascending
    thr = float(pos_sorted[P - k])    # k-th largest
    return thr
def _plot_pit(u: np.ndarray, out_path: str) -> Dict[str, float]:
    # NOTE: this function expects PIT values in [0,1] (NOT latent u ~ N(0,1)).
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
    """Reliability diagram (with bin-wise confidence intervals).

    We bin by predicted probability, then compute:
      - p_mean: mean predicted probability in bin
      - y_rate: empirical event rate in bin
      - Wilson 95% CI for y_rate (helps reviewers understand sparsity at high-risk bins)

    Returns summary stats including ECE (same binning) and Brier score.
    """
    y_true = y_true.astype(np.float64).reshape(-1)
    p_pred = np.clip(p_pred.astype(np.float64).reshape(-1), 0.0, 1.0)
    if y_true.size == 0:
        return {"ece": float("nan"), "brier": float("nan")}

    z = 1.96  # ~95% CI
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece_val = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p_pred >= lo) & (p_pred < hi) if i < n_bins - 1 else (p_pred >= lo) & (p_pred <= hi)
        n = int(np.sum(m))
        if n == 0:
            rows.append({
                "bin": i, "lo": lo, "hi": hi, "n": 0,
                "p_mean": float("nan"), "y_rate": float("nan"),
                "y_ci_low": float("nan"), "y_ci_high": float("nan"),
            })
            continue

        p_mean = float(np.mean(p_pred[m]))
        y_rate = float(np.mean(y_true[m]))

        # Wilson interval for proportion
        phat = y_rate
        denom = 1.0 + (z * z) / n
        center = (phat + (z * z) / (2.0 * n)) / denom
        half = (z * math.sqrt((phat * (1.0 - phat)) / n + (z * z) / (4.0 * n * n))) / denom
        y_lo = float(max(0.0, center - half))
        y_hi = float(min(1.0, center + half))

        rows.append({
            "bin": i, "lo": lo, "hi": hi, "n": n,
            "p_mean": p_mean, "y_rate": y_rate,
            "y_ci_low": y_lo, "y_ci_high": y_hi,
        })
        ece_val += (n / y_true.size) * abs(y_rate - p_mean)

    brier_val = float(np.mean((p_pred - y_true) ** 2))

    # Save CSV
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Plot reliability + CI
    xs = np.array([r["p_mean"] for r in rows if r["n"] > 0], dtype=np.float64)
    ys = np.array([r["y_rate"] for r in rows if r["n"] > 0], dtype=np.float64)
    ylo = np.array([r["y_ci_low"] for r in rows if r["n"] > 0], dtype=np.float64)
    yhi = np.array([r["y_ci_high"] for r in rows if r["n"] > 0], dtype=np.float64)

    plt.figure(figsize=(5, 4))
    if xs.size > 0:
        plt.errorbar(xs, ys, yerr=[ys - ylo, yhi - ys], fmt="o", capsize=3)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical event rate")
    plt.title(f"Calibration (bins={n_bins}, Wilson 95% CI)")
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
    print("[diagnose] VERSION: itsc_v6_2_4_hotfix (quantile-tail main, physical appendix)")

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test", "all"])

    ap.add_argument("--ttc_floor", type=float, default=0.05)
    ap.add_argument("--ttc_cap", type=float, default=10.0)

    # Main (paper) risk definition
    ap.add_argument("--main_risk_def", choices=["quantile_tail", "physical_sstar"], default="quantile_tail",
                    help="Main risk used in paper. 'quantile_tail' = P(TTC <= t_q). 'physical_sstar' = P(TTC <= s*(v)).")
    ap.add_argument("--tail_q", type=float, default=0.10, help="Quantile q for tail-event threshold t_q (default 0.10).")
    ap.add_argument("--tail_q_source", choices=["train_candidates", "train_uncensored", "train_gatepos"], default="train_candidates",
                    help="Which TRAIN subset defines t_q.")
    ap.add_argument("--require_edges", action="store_true",
                    help="If set, define events only when expert_mask=1 (recommended for reviewer-proof evaluation).")

    # Physical appendix (optional)
    ap.add_argument("--appendix_physical", action="store_true",
                    help="Also export physical risk (tau sweep) under appendix_physical/.")
    ap.add_argument("--sstar_mode", default="closing_speed")
    ap.add_argument("--amax", type=float, default=6.0)
    ap.add_argument("--tau_sweep", default="0.5")

    # Gate thresholding: ONLY recall-defined budgets (no p_gate>=0.5)
    ap.add_argument("--gate_target_recall", type=float, default=0.90)

    # Randomized PIT
    ap.add_argument("--pit_seed", type=int, default=0)

    # Outputs
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--flow_diag_dir", required=True)
    ap.add_argument("--paper_mode", action="store_true",
                    help="Paper mode: main=quantile_tail(q=0.1), require_edges=True, calib/KS on gated-only, appendix physical on.")
    ap.add_argument("--save_preds", action="store_true")

    # Calibration / KS
    ap.add_argument("--calib_bins", type=int, default=15)
    ap.add_argument("--calib_on_gated_only", action="store_true")
    ap.add_argument("--ks_on_gated_only", action="store_true")

    # Strategy A: uncertainty
    ap.add_argument("--uncertainty_samples", type=int, default=0)
    ap.add_argument("--uncertainty_scope", choices=["all", "gated", "both"], default="both")
    ap.add_argument("--uncertainty_batch", type=int, default=1024)

    # Strategy B: subgroup + sensitivity
    ap.add_argument("--subgroup_features", type=str,
                    default="x__density,x__dist_to_intersection_lane_m,x__dyn_label_count_30m,x__occlusion_low_points_ratio_dyn_30m",
                    help="Comma-separated feature list for subgroup splits.")
    ap.add_argument("--subgroup_split", choices=["median", "quantile3"], default="median")
    ap.add_argument("--subgroup_on", choices=["all", "gated", "both"], default="both")

    # ITSC package: partial-effect regression + interaction terms + density CI
    ap.add_argument("--itsc_package", action="store_true", help="Export ITSC-ready tables/figures (regression + CI plots).")

    # Feature ablations (keep for compatibility)
    ap.add_argument("--expert_keep_c_cont", action="store_true")
    ap.add_argument("--expert_drop_all_context", action="store_true")
    ap.add_argument("--expert_drop_feature_regex", type=str, default="")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_rows", type=int, default=0)

    args = ap.parse_args()

    # -------------------------
    # Paper mode defaults
    # -------------------------
    if getattr(args, 'paper_mode', False):
        # main risk
        args.main_risk_def = 'quantile_tail'
        # reviewer-proof: only evaluate on edges by default
        args.require_edges = True
        # consistent gates / calibration on gated subset
        args.calib_on_gated_only = True
        args.ks_on_gated_only = True
        # main quantile default
        if args.tail_q is None:
            args.tail_q = 0.10
        # appendix physical ON
        args.appendix_physical = True
        # auto export package
        args.itsc_package = True

    run_dir = Path(args.run)
    _ensure_dir(args.out_dir)
    _ensure_dir(args.flow_diag_dir)

    # -------------------------
    # Load preprocess state + CSV (ALL splits)
    # -------------------------
    state_path = run_dir / "preprocess_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing preprocess_state.json: {state_path}")
    state = load_preprocess_state(str(state_path))

    header = pd.read_csv(args.csv, nrows=0)
    dtypes = {c: np.float32 for c in header.columns if c != "segment_id"}
    dtypes["segment_id"] = str
    df_all = pd.read_csv(args.csv, dtype=dtypes)

    split_map = _load_split_map(str(run_dir))

    # Subset for evaluation split
    df_split = _filter_df_by_split(df_all.copy(), split_map, args.split)
    if args.max_rows and args.max_rows > 0 and len(df_split) > args.max_rows:
        df_split = df_split.sample(n=int(args.max_rows), random_state=0).copy()

    # TRAIN subset for quantile threshold (always from train split)
    df_train = _filter_df_by_split(df_all.copy(), split_map, "train")

    # Transform
    df_t = transform_dataframe(df_split, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)
    ds = RiskCSVDataset(df_t, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)

    df_train_t = transform_dataframe(df_train, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)

    device = torch.device(args.device)
    gate, flow = _load_models_for_run(run_dir, state, device=device)

    scale = ds.get_scale_tensors(device=device)
    gate_mean, gate_std = scale["gate_mean"], scale["gate_std"]
    expert_mean, expert_std = scale["expert_mean"], scale["expert_std"]

    # -------------------------
    # Build tensors + cond
    # -------------------------
    xg_raw = ds.tensors.x_gate_raw.to(device)
    xe_raw = ds.tensors.x_expert_raw.to(device)

    y_gate = ds.tensors.y_gate.to(device)
    y_expert = ds.tensors.y_expert.to(device)
    expert_mask = ds.tensors.expert_mask.to(device)
    censored_mask = ds.tensors.censored_mask.to(device)

    v_close = ds.tensors.raw_closing_speed_mps.to(device)

    xg = (xg_raw - gate_mean) / (gate_std + 1e-6)
    xe = (xe_raw - expert_mean) / (expert_std + 1e-6)

    cond = _make_flow_cond(
        xe,
        flow,
        flow_x_idx=getattr(state, "flow_x_idx", None),
        flow_c_idx=getattr(state, "flow_c_idx", None),
    )

    gate_logits = gate(xg)
    p_gate = torch.sigmoid(gate_logits).reshape(-1)

    # -------------------------
    # Directories
    # -------------------------
    base_out = Path(args.out_dir) / run_dir.name / args.split
    base_flow = Path(args.flow_diag_dir) / run_dir.name / args.split
    _ensure_dir(str(base_out))
    _ensure_dir(str(base_flow))

    # -------------------------
    # Randomized PIT (density calibration) - tau independent
    # -------------------------
    rng = np.random.default_rng(int(getattr(args, "pit_seed", 0)))
    y_obs_std = y_expert.reshape(-1)

    y_cap = float(args.ttc_cap)
    y_cap_std = (torch.log(torch.tensor(y_cap, device=device) + 1e-9) - float(state.target_std.mu_y)) / (float(state.target_std.sigma_y) + 1e-6)
    y_cap_std = y_cap_std.reshape(1)

    u_obs = flow.cdf(y_obs_std, cond).reshape(-1).detach().cpu().numpy()
    cens_np = (censored_mask.detach().cpu().numpy().reshape(-1) > 0.5)

    if np.any(cens_np):
        y_cap_std_vec = torch.full_like(y_obs_std, float(y_cap_std.item()))
        u_low = flow.cdf(y_cap_std_vec, cond).reshape(-1).detach().cpu().numpy()
        r = rng.random(int(np.sum(cens_np)), dtype=np.float64)
        u_obs[cens_np] = u_low[cens_np] + (1.0 - u_low[cens_np]) * r

    pit_np = np.clip(u_obs, 0.0, 1.0)
    pit_png = str(base_flow / "pit_hist.png")
    pit_stats = _plot_pit(pit_np, pit_png)
    pit_stats.update({
        "pit_mode": "randomized_cdf",
        "ttc_cap": float(args.ttc_cap),
        "n_censored": int(np.sum(cens_np)),
        "censored_ratio": float(np.mean(cens_np)) if cens_np.size else float("nan"),
    })
    (base_flow / "pit_stats.json").write_text(json.dumps(pit_stats, indent=2), encoding="utf-8")

    # -------------------------
    # Strategy A: sigma_log via flow sampling
    # -------------------------
    sigma_log_np = None
    if int(args.uncertainty_samples) > 0:
        print(f"[INFO] sigma_log via flow sampling: S={int(args.uncertainty_samples)}")
        sigma_log_np = infer_flow_sigma_log(
            flow,
            cond,
            target_mu=float(state.target_std.mu_y),
            target_sigma=float(state.target_std.sigma_y),
            num_samples=int(args.uncertainty_samples),
            batch=int(args.uncertainty_batch),
            device=device,
        )
        np.save(str(base_flow / "sigma_log.npy"), sigma_log_np)
        sigma_stats = {
            "n": int(sigma_log_np.size),
            "mean": float(np.nanmean(sigma_log_np)) if sigma_log_np.size else float("nan"),
            "std": float(np.nanstd(sigma_log_np)) if sigma_log_np.size else float("nan"),
            "p50": float(np.nanpercentile(sigma_log_np, 50)) if sigma_log_np.size else float("nan"),
            "p90": float(np.nanpercentile(sigma_log_np, 90)) if sigma_log_np.size else float("nan"),
        }
        (base_flow / "sigma_log_stats.json").write_text(json.dumps(sigma_stats, indent=2), encoding="utf-8")

    # -------------------------
    # Tail threshold (quantile) from TRAIN
    # -------------------------
    if args.main_risk_def == 'quantile_tail' or args.appendix_physical:
        mu = float(state.target_std.mu_y)
        sg = float(state.target_std.sigma_y)

    ttc_q = None
    y_thr_std_q = None
    if args.main_risk_def == 'quantile_tail':
        # Use TRAIN distribution of TTC (in seconds)
        y_train_std = torch.tensor(df_train_t["y_expert"].to_numpy(np.float32))
        expert_mask_train = torch.tensor(df_train_t["expert_mask"].to_numpy(np.float32))
        y_gate_train = torch.tensor(df_train_t["y_gate"].to_numpy(np.float32))

        y_train_log = y_train_std * (sg + 1e-6) + mu
        ttc_train = torch.exp(y_train_log).clamp(min=float(args.ttc_floor), max=float(args.ttc_cap)).numpy()

        # censored mask may not exist in transformed DF (older preprocess).
        # Derive it consistently from (standardized logTTC -> seconds) and ttc_cap.
        if "censored_mask" in df_train_t.columns:
            cens_train_np = (df_train_t["censored_mask"].to_numpy(np.float32) > 0.5)
        else:
            # Since TTC is clamped to ttc_cap, values at cap imply right-censoring in our pipeline.
            cens_train_np = (ttc_train >= (float(args.ttc_cap) - 1e-9))

        if args.tail_q_source == 'train_candidates':
            m = (expert_mask_train.numpy() > 0.5)
        elif args.tail_q_source == 'train_uncensored':
            m = (expert_mask_train.numpy() > 0.5) & (~cens_train_np)
        elif args.tail_q_source == 'train_gatepos':
            m = (y_gate_train.numpy() > 0.5)
        else:
            m = np.ones_like(ttc_train, dtype=bool)

        if m.sum() <= 0:
            raise RuntimeError(f"No samples to compute tail quantile: source={args.tail_q_source}")

        q = float(args.tail_q)
        q = min(max(q, 1e-6), 1.0 - 1e-6)
        ttc_q = float(np.quantile(ttc_train[m], q))
        y_thr_std_q = (np.log(ttc_q + 1e-9) - mu) / (sg + 1e-6)

        (base_out / "tail_event_config.json").write_text(json.dumps({
            "main_risk_def": "quantile_tail",
            "tail_q": float(args.tail_q),
            "tail_q_source": str(args.tail_q_source),
            "ttc_q_seconds": float(ttc_q),
            "y_thr_std_q": float(y_thr_std_q),
            "ttc_floor": float(args.ttc_floor),
            "ttc_cap": float(args.ttc_cap),
        }, indent=2), encoding="utf-8")

    # -------------------------
    # Helper: get subgroup feature arrays
    # -------------------------
    subgroup_features = [s.strip() for s in str(args.subgroup_features).split(",") if s.strip()]
    subgroup_vals = {}
    for feat in subgroup_features:
        try:
            subgroup_vals[feat] = _get_feature_array(ds, feat)
        except KeyError:
            print(f"[WARN] subgroup feature not found: {feat} (skip)")

    # -------------------------
    # Evaluate MAIN risk
    # -------------------------
    rows = []

    def _save_preds_csv(tag: str, y_true_np, p_gate_np, p_event_np, risk_np, gate_thr, gated_mask):
        if not args.save_preds:
            return None
        pred_df = pd.DataFrame({
            "segment_id": ds.tensors.segment_ids,
            "frame_label": ds.tensors.frame_label.cpu().numpy(),
            "y_event": y_true_np,
            "p_gate": p_gate_np,
            "p_event": p_event_np,
            "risk": risk_np,
            "gate_thr": float(gate_thr) if np.isfinite(gate_thr) else float('nan'),
            "gated": gated_mask.astype(int),
            "raw_closing_speed_mps": ds.tensors.raw_closing_speed_mps.cpu().numpy(),
            "censored": (ds.tensors.censored_mask.cpu().numpy() > 0.5).astype(int),
        })
        out_csv = base_out / f"preds_{tag}.csv"
        pred_df.to_csv(out_csv, index=False)
        return str(out_csv)

    # Main: quantile tail
    if args.main_risk_def == 'quantile_tail':
        assert y_thr_std_q is not None
        y_thr_vec = torch.full_like(y_expert.reshape(-1), float(y_thr_std_q))
        p_event = flow.cdf(y_thr_vec, cond).reshape(-1)

        if args.require_edges:
            y_true = ((expert_mask.reshape(-1) > 0.5) & (y_expert.reshape(-1) <= y_thr_vec)).float()
        else:
            y_true = (y_expert.reshape(-1) <= y_thr_vec).float()

        risk = (p_gate * p_event).reshape(-1)

        y_true_np = y_true.detach().cpu().numpy().astype(np.int64)
        p_gate_np = p_gate.detach().cpu().numpy()
        p_event_np = p_event.detach().cpu().numpy()
        risk_np = risk.detach().cpu().numpy()

        pos_rate = float(np.mean(y_true_np)) if y_true_np.size else float('nan')
        pr = pr_auc_average_precision(y_true_np, risk_np)
        roc = roc_auc_trapezoid(y_true_np, risk_np)
        brier = float(np.mean((risk_np - y_true_np) ** 2)) if y_true_np.size else float('nan')
        e_risk = ece(y_true_np, risk_np, n_bins=int(args.calib_bins))

        # Gate ROC/PR against y_true (event detection)
        gate_pr = pr_auc_average_precision(y_true_np, p_gate_np)
        gate_roc = roc_auc_trapezoid(y_true_np, p_gate_np)

        gate_thr = _choose_gate_thr_for_recall(p_gate_np, y_true_np, [], float(args.gate_target_recall))
        gated_mask = (p_gate_np >= gate_thr) if np.isfinite(gate_thr) else np.ones_like(p_gate_np, dtype=bool)

        # Gate budget
        n_all = int(y_true_np.size)
        n_pos_all = int(np.sum(y_true_np == 1))
        n_g = int(np.sum(gated_mask))
        n_pos_g = int(np.sum(y_true_np[gated_mask] == 1)) if n_g > 0 else 0
        pos_rate_all = float(pos_rate)
        pos_rate_gated = float(n_pos_g / max(1, n_g))
        gate_pass_rate = float(n_g / max(1, n_all))
        lift = float(pos_rate_gated / max(1e-12, pos_rate_all)) if np.isfinite(pos_rate_all) else float('nan')
        pos_mask = (y_true_np == 1)
        gate_recall_achieved = float(np.mean(p_gate_np[pos_mask] >= gate_thr)) if (pos_mask.sum() > 0 and np.isfinite(gate_thr)) else float('nan')

        gate_budget = pd.DataFrame([{
            "main_risk_def": "quantile_tail",
            "tail_q": float(args.tail_q),
            "ttc_q_seconds": float(ttc_q) if ttc_q is not None else float('nan'),
            "gate_target_recall": float(args.gate_target_recall),
            "gate_thr": float(gate_thr),
            "gate_recall_achieved": float(gate_recall_achieved),
            "gate_pass_rate": float(gate_pass_rate),
            "n_all": int(n_all),
            "n_gated": int(n_g),
            "pos_rate_all": float(pos_rate_all),
            "pos_rate_gated": float(pos_rate_gated),
            "lift": float(lift),
            "n_pos_all": int(n_pos_all),
            "n_pos_gated": int(n_pos_g),
        }])
        gate_budget_csv = base_out / "gate_budget.csv"
        gate_budget.to_csv(gate_budget_csv, index=False)

        # Calibration curves
        # p_event calibration (conditional) + risk calibration
        if args.calib_on_gated_only:
            y_cal = y_true_np[gated_mask]
            p_cal = p_event_np[gated_mask]
            r_cal = risk_np[gated_mask]
        else:
            y_cal = y_true_np
            p_cal = p_event_np
            r_cal = risk_np

        calib_p_png = base_out / "p_event_calibration.png"
        calib_p_csv = base_out / "p_event_calibration.csv"
        calib_p_stats = _save_calibration_curve(y_cal, p_cal, str(calib_p_png), str(calib_p_csv), n_bins=int(args.calib_bins))

        calib_r_png = base_out / "risk_calibration.png"
        calib_r_csv = base_out / "risk_calibration.csv"
        calib_r_stats = _save_calibration_curve(y_cal, r_cal, str(calib_r_png), str(calib_r_csv), n_bins=int(args.calib_bins))

        # PIT KS
        pit_use = pit_np[gated_mask] if args.ks_on_gated_only else pit_np
        ks_png = base_out / "pit_ks_cdf.png"
        ks_stats = _save_ks_plot_u_vs_uniform(pit_use, str(ks_png))

        out = {
            "run_name": run_dir.name,
            "split": args.split,
            "main_risk_def": "quantile_tail",
            "tail_q": float(args.tail_q),
            "tail_q_source": str(args.tail_q_source),
            "ttc_q_seconds": float(ttc_q) if ttc_q is not None else float('nan'),
            "N": int(n_all),
            "pos_rate": float(pos_rate_all),
            "risk_pr_auc": float(pr),
            "risk_roc_auc": float(roc),
            "risk_brier": float(brier),
            "risk_ece": float(e_risk),
            "gate_pr_auc": float(gate_pr),
            "gate_roc_auc": float(gate_roc),
            "p_gate_mean": float(np.mean(p_gate_np)) if p_gate_np.size else float('nan'),
            "p_event_mean": float(np.mean(p_event_np)) if p_event_np.size else float('nan'),
            "gate_thr_for_target_recall": float(gate_thr),
            "gate_recall_achieved": float(gate_recall_achieved),
            "gate_pass_rate": float(gate_pass_rate),
            "pos_rate_gated": float(pos_rate_gated),
            "gate_lift": float(lift),
            "n_gated": int(n_g),
            "n_pos_gated": int(n_pos_g),
            "p_event_calib_ece": float(calib_p_stats.get('ece', float('nan'))),
            "p_event_calib_brier": float(calib_p_stats.get('brier', float('nan'))),
            "risk_calib_ece": float(calib_r_stats.get('ece', float('nan'))),
            "risk_calib_brier": float(calib_r_stats.get('brier', float('nan'))),
            "pit_ks_u_uniform": float(ks_stats.get('ks', float('nan'))),
            "pit_ks_n": int(ks_stats.get('n', 0)),
            "pit_mean": float(pit_stats.get('u_mean', float('nan'))),
            "pit_std": float(pit_stats.get('u_std', float('nan'))),
        }

        # Strategy A summaries
        if sigma_log_np is not None:
            if args.uncertainty_scope in ("all", "both"):
                out["sigma_log_mean_all"] = float(np.nanmean(sigma_log_np))
                out["sigma_log_p90_all"] = float(np.nanpercentile(sigma_log_np, 90))
            if args.uncertainty_scope in ("gated", "both"):
                sg = sigma_log_np[gated_mask]
                out["sigma_log_mean_gated"] = float(np.nanmean(sg)) if sg.size else float('nan')
                out["sigma_log_p90_gated"] = float(np.nanpercentile(sg, 90)) if sg.size else float('nan')

        # Strategy B: subgroup summaries (now includes p_event/risk too)
        if len(subgroup_vals) > 0:
            subgroup_rows = []
            for feat, vals in subgroup_vals.items():
                for bin_name, bin_mask in _make_bins(vals, args.subgroup_split):
                    m_all = bin_mask
                    n_bin = int(np.sum(m_all))
                    if n_bin == 0:
                        continue
                    row = {"feature": feat, "split": args.subgroup_split, "bin": bin_name, "n_all": n_bin}

                    if args.subgroup_on in ("all", "both"):
                        row.update({
                            "pos_rate_all": float(np.mean(y_true_np[m_all])),
                            "p_gate_mean_all": float(np.mean(p_gate_np[m_all])),
                            "p_event_mean_all": float(np.mean(p_event_np[m_all])),
                            "risk_mean_all": float(np.mean(risk_np[m_all])),
                        })
                        if sigma_log_np is not None:
                            row["sigma_log_mean_all"] = float(np.nanmean(sigma_log_np[m_all]))
                            row["sigma_log_p90_all"] = float(np.nanpercentile(sigma_log_np[m_all], 90))

                    m_g = m_all & gated_mask
                    n_gb = int(np.sum(m_g))
                    row["n_gated"] = n_gb
                    if args.subgroup_on in ("gated", "both"):
                        row.update({
                            "pos_rate_gated": float(np.mean(y_true_np[m_g])) if n_gb else float('nan'),
                            "p_gate_mean_gated": float(np.mean(p_gate_np[m_g])) if n_gb else float('nan'),
                            "p_event_mean_gated": float(np.mean(p_event_np[m_g])) if n_gb else float('nan'),
                            "risk_mean_gated": float(np.mean(risk_np[m_g])) if n_gb else float('nan'),
                        })
                        if sigma_log_np is not None:
                            row["sigma_log_mean_gated"] = float(np.nanmean(sigma_log_np[m_g])) if n_gb else float('nan')
                            row["sigma_log_p90_gated"] = float(np.nanpercentile(sigma_log_np[m_g], 90)) if n_gb else float('nan')

                    subgroup_rows.append(row)

            if subgroup_rows:
                subgroup_df = pd.DataFrame(subgroup_rows)
                subgroup_csv = base_out / "subgroup_summary.csv"
                subgroup_df.to_csv(subgroup_csv, index=False)
                out["subgroup_summary_csv"] = str(subgroup_csv)

        # Density sensitivity CI (sigma_log)
        if (sigma_log_np is not None) and ("x__density" in subgroup_vals):
            dens = np.asarray(subgroup_vals["x__density"], dtype=np.float64).reshape(-1)
            bins_d = dict(_make_bins(dens, args.subgroup_split))
            if ("low" in bins_d) and ("high" in bins_d):
                m_low = np.asarray(bins_d["low"], dtype=bool)
                m_high = np.asarray(bins_d["high"], dtype=bool)
                stat_all = _mean_diff_ci(sigma_log_np[m_high], sigma_log_np[m_low])
                stat_g = _mean_diff_ci(sigma_log_np[m_high & gated_mask], sigma_log_np[m_low & gated_mask])

                dens_df = pd.DataFrame([
                    {"scope": "all", "split": args.subgroup_split,
                     "mean_high": stat_all["mean_a"], "mean_low": stat_all["mean_b"],
                     "diff_high_minus_low": stat_all["diff"], "ci_low": stat_all["ci_low"], "ci_high": stat_all["ci_high"],
                     "n_high": int(np.sum(m_high)), "n_low": int(np.sum(m_low)),
                     "gate_pass_high": float(np.mean(gated_mask[m_high])), "gate_pass_low": float(np.mean(gated_mask[m_low]))},
                    {"scope": "gated", "split": args.subgroup_split,
                     "mean_high": stat_g["mean_a"], "mean_low": stat_g["mean_b"],
                     "diff_high_minus_low": stat_g["diff"], "ci_low": stat_g["ci_low"], "ci_high": stat_g["ci_high"],
                     "n_high": int(np.sum(m_high & gated_mask)), "n_low": int(np.sum(m_low & gated_mask)),
                     "gate_pass_high": float(np.mean(gated_mask[m_high])), "gate_pass_low": float(np.mean(gated_mask[m_low]))},
                ])
                dens_csv = base_out / "density_sensitivity.csv"
                dens_df.to_csv(dens_csv, index=False)
                out["density_sensitivity_csv"] = str(dens_csv)

                try:
                    plt.figure(figsize=(5,4))
                    xs = np.arange(len(dens_df))
                    diffs = dens_df["diff_high_minus_low"].values.astype(np.float64)
                    yerr_low = diffs - dens_df["ci_low"].values.astype(np.float64)
                    yerr_high = dens_df["ci_high"].values.astype(np.float64) - diffs
                    plt.errorbar(xs, diffs, yerr=[yerr_low, yerr_high], fmt='o', capsize=4)
                    plt.axhline(0.0, linestyle='--')
                    plt.xticks(xs, dens_df["scope"].tolist())
                    plt.ylabel("sigma_log(high) - sigma_log(low)")
                    plt.title("Density sensitivity (95% CI)")
                    plt.tight_layout()
                    plt.savefig(str(base_out / "density_sensitivity.png"))
                    plt.close()
                except Exception as e:
                    print(f"[WARN] density plot failed: {e}")

        # ITSC package: partial-effect regression + interactions
        if args.itsc_package and (sigma_log_np is not None):
            # ITSC package: partial-effect regression + interaction terms (statsmodels-free, HC3 robust)
            feats_req = [
                "x__density",
                "x__dist_to_intersection_lane_m",
                "x__dyn_label_count_30m",
                "x__occlusion_low_points_ratio_dyn_30m",
            ]
            feats = [f for f in feats_req if f in subgroup_vals]
            if len(feats) < 2:
                print(f"[WARN] ITSC package skipped: need >=2 features, found {len(feats)}: {feats}")
            else:
                try:
                    from math import erfc, sqrt
                    # Build regression frame
                    df_reg = pd.DataFrame({
                        "sigma_log": np.asarray(sigma_log_np, dtype=np.float64),
                        "p_event": np.asarray(p_event_np, dtype=np.float64),
                        "gated": gated_mask.astype(int),
                    })
                    for f in feats:
                        df_reg[f] = np.asarray(subgroup_vals[f], dtype=np.float64)

                    df_reg = df_reg.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
                    if len(df_reg) < 50:
                        print(f"[WARN] ITSC package skipped: too few rows after NaN drop (n={len(df_reg)})")
                    else:
                        # z-score predictors
                        for f in feats:
                            mu_f = df_reg[f].mean()
                            sd_f = df_reg[f].std() + 1e-12
                            df_reg[f"z_{f}"] = (df_reg[f] - mu_f) / sd_f

                        # Interactions: density with others
                        if "x__density" in feats:
                            for f in feats:
                                if f == "x__density":
                                    continue
                                df_reg[f"z_x__density: z_{f}"] = df_reg["z_x__density"] * df_reg[f"z_{f}"]

                        X_cols = [f"z_{f}" for f in feats] + [c for c in df_reg.columns if c.startswith("z_x__density: ")]
                        # Design matrix with intercept
                        X = np.column_stack([np.ones(len(df_reg), dtype=np.float64)] + [df_reg[c].to_numpy(np.float64) for c in X_cols])

                        def _ols_hc3(Xm: np.ndarray, y: np.ndarray):
                            # beta = (X'X)^-1 X'y
                            XtX = Xm.T @ Xm
                            XtX_inv = np.linalg.pinv(XtX)
                            beta = XtX_inv @ (Xm.T @ y)
                            yhat = Xm @ beta
                            resid = y - yhat
                            # leverage h_ii
                            h = np.sum((Xm @ XtX_inv) * Xm, axis=1)
                            denom = np.clip(1.0 - h, 1e-12, None)
                            w = (resid / denom) ** 2
                            meat = Xm.T @ (Xm * w[:, None])
                            cov = XtX_inv @ meat @ XtX_inv
                            se = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
                            t = beta / se
                            # Normal approx p-value
                            p = np.array([erfc(abs(tt) / sqrt(2.0)) for tt in t], dtype=np.float64)
                            ci_low = beta - 1.96 * se
                            ci_high = beta + 1.96 * se
                            return beta, se, t, p, ci_low, ci_high

                        # (1) sigma_log OLS
                        y1 = df_reg["sigma_log"].to_numpy(np.float64)
                        b1, se1, t1, p1, lo1, hi1 = _ols_hc3(X, y1)
                        terms = ["const"] + X_cols
                        tab1 = pd.DataFrame({
                            "term": terms,
                            "beta": b1,
                            "se": se1,
                            "t": t1,
                            "p": p1,
                            "ci_low": lo1,
                            "ci_high": hi1,
                        })
                        tab1.to_csv(base_out / "reg_sigma_log.csv", index=False)

                        # (2) logit(p_event) OLS
                        pe = np.clip(df_reg["p_event"].to_numpy(np.float64), 1e-6, 1 - 1e-6)
                        y2 = np.log(pe / (1.0 - pe))
                        b2, se2, t2, p2, lo2, hi2 = _ols_hc3(X, y2)
                        tab2 = pd.DataFrame({
                            "term": terms,
                            "beta": b2,
                            "se": se2,
                            "t": t2,
                            "p": p2,
                            "ci_low": lo2,
                            "ci_high": hi2,
                        })
                        tab2.to_csv(base_out / "reg_logit_p_event.csv", index=False)
                        # Export interaction-only table (ITSC convenience): stack sigma_log + logit(p_event)
                        try:
                            _int_rows = []
                            for _out_name, _tab in [("sigma_log", tab1), ("logit_p_event", tab2)]:
                                _mask = _tab["term"].astype(str).str.startswith("z_x__density: ")
                                if bool(_mask.any()):
                                    _tmp = _tab.loc[_mask].copy()
                                    _tmp.insert(0, "outcome", _out_name)
                                    _tmp["partner"] = _tmp["term"].astype(str).str.replace("z_x__density: ", "", regex=False)
                                    _int_rows.append(_tmp)
                            if len(_int_rows) > 0:
                                _itab = pd.concat(_int_rows, ignore_index=True)
                                _itab.to_csv(base_out / "interaction_table.csv", index=False)
                                out["itsc_interaction_table_csv"] = str(base_out / "interaction_table.csv")
                        except Exception as _e:
                            print(f"[WARN] interaction_table export failed: {_e}")

                        # Partial effect curve for density (holding others at 0)
                        if "x__density" in feats:
                            grid = np.linspace(-2.0, 2.0, 101)
                            # build Xp with intercept + cols
                            Xp = np.zeros((grid.size, len(terms)), dtype=np.float64)
                            Xp[:, 0] = 1.0
                            # set z_density
                            if "z_x__density" in X_cols:
                                j = terms.index("z_x__density")
                                Xp[:, j] = grid
                            # interactions are zero because others are held at 0
                            yhat_grid = Xp @ b1
                            df_pe = pd.DataFrame({"z_x__density": grid, "sigma_log_hat": yhat_grid})
                            df_pe.to_csv(base_out / "partial_effect_density_sigma_log.csv", index=False)
                            try:
                                plt.figure()
                                plt.plot(df_pe["z_x__density"], df_pe["sigma_log_hat"])
                                plt.xlabel("z(x__density)")
                                plt.ylabel("E[sigma_log] (partial)")
                                plt.title("Partial effect: density → sigma_log")
                                plt.tight_layout()
                                plt.savefig(base_out / "partial_effect_density_sigma_log.png", dpi=160)
                                plt.close()
                            except Exception as e:
                                print(f"[WARN] partial effect plot failed: {e}")

                        print(f"[ITSC] regression exported: {base_out / 'reg_sigma_log.csv'} (features={feats})")
                        out["itsc_reg_sigma_log_csv"] = str(base_out / "reg_sigma_log.csv")
                        out["itsc_reg_logit_p_event_csv"] = str(base_out / "reg_logit_p_event.csv")
                        out["itsc_partial_effect_density_sigma_log_csv"] = str(base_out / "partial_effect_density_sigma_log.csv")

                except Exception as e:
                    print(f"[WARN] ITSC package regression failed (numpy HC3): {e}")


        out["gate_budget_csv"] = str(gate_budget_csv)
        out["p_event_calibration_csv"] = str(calib_p_csv)
        out["risk_calibration_csv"] = str(calib_r_csv)

        preds_csv = _save_preds_csv("main", y_true_np, p_gate_np, p_event_np, risk_np, gate_thr, gated_mask)
        if preds_csv:
            out["preds_csv"] = preds_csv

        # Save main json
        out_path = base_out / f"diagnose_{run_dir.name}_{args.split}_main.json"
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        rows.append(out)

        print(
            f"[OK] MAIN quantile_tail q={args.tail_q:.2f} t_q={ttc_q:.3f}s "
            f"pos={pos_rate_all:.4f} PR-AUC={pr:.4f} ROC-AUC={roc:.4f} "
            f"gate_pass={gate_pass_rate:.3f} lift={lift:.2f} thr@recall{args.gate_target_recall:.2f}={gate_thr:.4f}"
        )

    # -------------------------
    # Appendix: physical risk (tau sweep)
    # -------------------------
    if args.appendix_physical:
        taus = [float(x) for x in str(args.tau_sweep).split(",") if str(x).strip()]
        app_dir = base_out / "appendix_physical"
        _ensure_dir(str(app_dir))
        app_rows = []

        for tau in taus:
            cfg = RiskConfig(tau=float(tau), a_max=float(args.amax), ttc_floor=float(args.ttc_floor), ttc_cap=float(args.ttc_cap))

            # s*(v)
            v = torch.clamp(v_close, min=0.0)
            s = torch.clamp(cfg.tau + v / max(cfg.a_max, 1e-6), min=cfg.ttc_floor, max=cfg.ttc_cap)
            y_thr = torch.log(s + 1e-9)
            y_thr_std = (y_thr - float(state.target_std.mu_y)) / (float(state.target_std.sigma_y) + 1e-6)
            p_event = flow.cdf(y_thr_std, cond).reshape(-1)
            risk = (p_gate * p_event).reshape(-1)

            # event label = TTC <= s*(v) (uncensored-only)
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
            if args.require_edges:
                y_true = (y_true > 0.5) & (expert_mask.reshape(-1) > 0.5)
                y_true = y_true.float()

            y_true_np = y_true.detach().cpu().numpy().astype(np.int64)
            risk_np = risk.detach().cpu().numpy()
            p_gate_np = p_gate.detach().cpu().numpy()
            p_event_np = p_event.detach().cpu().numpy()

            pos_rate = float(np.mean(y_true_np)) if y_true_np.size else float('nan')
            pr = pr_auc_average_precision(y_true_np, risk_np)
            roc = roc_auc_trapezoid(y_true_np, risk_np)

            gate_thr = _choose_gate_thr_for_recall(p_gate_np, y_true_np, [], float(args.gate_target_recall))
            gated_mask = (p_gate_np >= gate_thr) if np.isfinite(gate_thr) else np.ones_like(p_gate_np, dtype=bool)

            # Save minimal appendix json per tau
            out_tau = {
                "tau": float(tau),
                "pos_rate": float(pos_rate),
                "risk_pr_auc": float(pr),
                "risk_roc_auc": float(roc),
                "p_gate_mean": float(np.mean(p_gate_np)),
                "p_event_mean": float(np.mean(p_event_np)),
                "gate_thr_for_target_recall": float(gate_thr),
                "gate_pass_rate": float(np.mean(gated_mask)),
            }
            (app_dir / f"physical_tau{tau}.json").write_text(json.dumps(out_tau, indent=2), encoding="utf-8")
            app_rows.append(out_tau)

        if app_rows:
            pd.DataFrame(app_rows).to_csv(app_dir / "appendix_physical_summary.csv", index=False)

    # Summary
    if rows:
        pd.DataFrame(rows).to_csv(base_out / "summary_main.csv", index=False)


if __name__ == "__main__":
    main()
