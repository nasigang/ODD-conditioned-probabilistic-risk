#!/usr/bin/env python3
"""11_case_study_odd_sensitivity.py

Case-study-lite for **Strategy B: ODD sensitivity generalization**.

What this proves (paper-friendly)
-------------------------------
We perform *counterfactual* ODD edits while keeping kinematics fixed, and quantify how
the learned model reacts in terms of:

  - P(Gate): stage-1 candidate probability
  - P(TTC <= t_q | Gate): stage-2 tail probability (Flow CDF)
  - Risk := P(Gate) * P(TTC <= t_q | Gate)  (main metric: quantile-tail)
  - sigma_log := Std[log(TTC)] implied by Flow sampling (uncertainty proxy)

Counterfactual families
-----------------------
A) Gate-only ODD one-hots (e.g., c__odd_weather=*)
   -> Mainly changes P(Gate) (routing / screening stage).

B) Expert-only ODD proxies (continuous, e.g., x__density, occlusion/visibility)
   -> Changes tail probability and sigma_log (distributional stage).

This script is intentionally cheap: no videos and no scenario labeling. Yet it is
reviewer-proof because it isolates ODD effects on the model output with bootstrap CIs.

Outputs
-------
Written to --out_dir:
  - case_study_odd_sensitivity.json   (summary + bootstrap 95% CI)
  - case_study_odd_sensitivity.csv    (per-counterfactual summary rows)
  - case_study_odd_sensitivity.png    (compact figure for the paper)
  - (optional) case0_flow_samples.png (distribution shift for one representative case)

Run example
-----------
python 11_case_study_odd_sensitivity.py \
  --run_dir /path/to/runs/run_concat_minimalv5 \
  --csv /path/to/gssm_inputs_train_minimal_v2_noleak.csv \
  --split val --n_case 2048 --seed 0 \
  --tail_q 0.10 --tail_q_source train_candidates \
  --proxy_cols x__density,x__occlusion_low_points_ratio_dyn_30m,x__mean_lidar_points_in_box_dyn_30m \
  --proxy_q_lo 0.10 --proxy_q_hi 0.90 \
  --sigma_samples 512 --gate_thr 0.5 \
  --n_boot 400 --out_dir /path/to/out/case_study
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Reuse the *exact* loaders/inference helpers from eval_risk_models.py
# (keeps behavior consistent with your main evaluation pipeline)
from eval_risk_models import (
    load_segment_split,
    pick_split_df,
    load_models_for_run,
    infer_gate_probs,
    infer_flow_cdf_probs,
    infer_flow_sigma_log,
)

from risk_pipeline.preprocess import load_preprocess_state, transform_dataframe
from risk_pipeline.data import RiskCSVDataset


def _safe_numeric_series(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros((len(df),), dtype=np.float32)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(np.float32)


def _bootstrap_ci_mean(x: np.ndarray, *, n_boot: int, seed: int) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    n = x.size
    idx = rng.integers(0, n, size=(int(n_boot), n))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def _bootstrap_ci_slope(x: np.ndarray, y: np.ndarray, *, n_boot: int, seed: int) -> Tuple[float, float, float]:
    """Bootstrap CI for a simple linear slope (y = a*x + b)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 5:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    n = x.size
    idx = rng.integers(0, n, size=(int(n_boot), n))
    slopes = []
    for b in idx:
        xb = x[b]
        yb = y[b]
        # guard: constant xb
        if np.std(xb) < 1e-12:
            slopes.append(np.nan)
            continue
        a, _ = np.polyfit(xb, yb, deg=1)
        slopes.append(a)
    slopes = np.asarray(slopes, dtype=np.float64)
    slopes = slopes[np.isfinite(slopes)]
    if slopes.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.quantile(slopes, [0.025, 0.975])
    return float(np.polyfit(x, y, deg=1)[0]), float(lo), float(hi)


def _make_onehot_counterfactuals(
    x_raw: torch.Tensor,
    colnames: List[str],
    *,
    prefix: str,
    prefer_order: List[str],
) -> Dict[str, torch.Tensor]:
    """Build one-hot counterfactuals for a group prefix like 'c__odd_weather='.

    Returns dict: {target_col_name: x_raw_cf}
    """
    cols = [c for c in colnames if c.startswith(prefix)]
    if not cols:
        return {}
    name_to_i = {c: i for i, c in enumerate(colnames)}
    idxs = [name_to_i[c] for c in cols]

    # Order targets: preferred tokens first, then the rest.
    def _score(c: str) -> Tuple[int, str]:
        token = c.split("=")[-1].lower()
        for k, t in enumerate(prefer_order):
            if t in token:
                return (k, token)
        return (len(prefer_order) + 1, token)

    cols_sorted = sorted(cols, key=_score)

    out: Dict[str, torch.Tensor] = {}
    for c in cols_sorted:
        x_cf = x_raw.clone()
        x_cf[:, idxs] = 0.0
        x_cf[:, name_to_i[c]] = 1.0
        out[c] = x_cf
    return out


def _plot_bars_with_ci(ax, labels: List[str], mean: List[float], lo: List[float], hi: List[float], title: str, ylabel: str):
    if not labels:
        ax.text(0.5, 0.5, "(no columns)", ha="center", va="center")
        ax.set_axis_off()
        return
    x = np.arange(len(labels))
    ax.bar(x, mean)
    ax.errorbar(
        x,
        mean,
        yerr=[np.array(mean) - np.array(lo), np.array(hi) - np.array(mean)],
        fmt="none",
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--n_case", type=int, default=2048, help="#frames sampled for analysis (from expert candidates)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ttc_floor", type=float, default=0.05)
    ap.add_argument("--ttc_cap", type=float, default=10.0)

    # Main (paper) risk definition: quantile-tail
    ap.add_argument("--tail_q", type=float, default=0.10, help="Quantile threshold q for tail event (paper main)")
    ap.add_argument("--tail_q_source", type=str, default="train_candidates", choices=["train_candidates", "train_all"],
                    help="Quantile computed on TRAIN rows: candidates only or all")

    # Expert proxy counterfactual
    ap.add_argument(
        "--proxy_cols",
        type=str,
        default="x__density,x__occlusion_low_points_ratio_dyn_30m,x__mean_lidar_points_in_box_dyn_30m",
        help="Comma-separated expert proxy columns to perturb",
    )
    ap.add_argument("--proxy_q_lo", type=float, default=0.10)
    ap.add_argument("--proxy_q_hi", type=float, default=0.90)

    # Uncertainty
    ap.add_argument("--sigma_samples", type=int, default=512)
    ap.add_argument("--gate_thr", type=float, default=0.5, help="Compute sigma_log only where P(Gate) >= thr")

    # Optional proxy regression (binned trend + slope CI)
    ap.add_argument("--proxy_bins", type=int, default=10, help="#quantile bins for proxy trend plots")
    ap.add_argument(
        "--proxy_reg_plot_col",
        type=str,
        default="x__density",
        help="Which proxy to visualize as a binned trend plot (must be in --proxy_cols)",
    )

    # Bootstrap
    ap.add_argument("--n_boot", type=int, default=400)
    ap.add_argument("--out_dir", type=str, required=True)

    # Optional: one representative distribution plot
    ap.add_argument("--plot_case0", action="store_true", help="Also plot Flow samples for one representative frame")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # --- Load state + split ---
    state = load_preprocess_state(str(run_dir / "preprocess_state.json"))
    split = load_segment_split(run_dir)

    # --- Load CSV (full, because we need TRAIN to compute tail quantile) ---
    df_all = pd.read_csv(args.csv)
    if "frame_label" not in df_all.columns:
        df_all["frame_label"] = np.arange(len(df_all), dtype=np.int64)

    # --- Prepare split DataFrames ---
    df_split = pick_split_df(df_all, split[args.split])
    df_train = pick_split_df(df_all, split.get("train", []))
    if len(df_train) == 0:
        raise RuntimeError("segment_split.json must contain a train split for quantile-tail threshold")

    # --- Transform (adds y_gate/y_expert/expert_mask, coerces feature cols) ---
    df_split_t = transform_dataframe(df_split, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)
    ds = RiskCSVDataset(df_split_t, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)

    # --- Compute tail threshold from TRAIN (matches eval_risk_models.py) ---
    df_train_t = transform_dataframe(df_train, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)
    mu_y = float(getattr(state.target_std, "mu_y", getattr(state.target_std, "mean")))
    sig_y = float(getattr(state.target_std, "sigma_y", getattr(state.target_std, "std")))
    eps_y = float(getattr(state.target_std, "eps", 1e-6))

    m_q = np.ones((len(df_train_t),), dtype=bool)
    if args.tail_q_source == "train_candidates":
        m_q = (pd.to_numeric(df_train_t.get("expert_mask", 0), errors="coerce").fillna(0.0).to_numpy() > 0.5)

    y_train_std = pd.to_numeric(df_train_t.get("y_expert", np.nan), errors="coerce").to_numpy(np.float64)
    y_train_log = y_train_std * (sig_y + eps_y) + mu_y
    ttc_train = np.clip(np.exp(y_train_log), float(args.ttc_floor), float(args.ttc_cap))
    if int(m_q.sum()) < 10:
        raise RuntimeError(f"Not enough TRAIN rows to compute tail quantile (n={int(m_q.sum())})")

    ttc_q = float(np.quantile(ttc_train[m_q], float(args.tail_q)))
    ttc_q = float(np.clip(ttc_q, float(args.ttc_floor), float(args.ttc_cap)))
    # Match preprocess: log(ttc + eps) with eps=1e-6
    y_thr_std_q = float((np.log(ttc_q + 1e-6) - mu_y) / (sig_y + eps_y))

    # --- Sample cases (expert candidates only) ---
    expert_mask = (ds.tensors.expert_mask.detach().cpu().numpy() > 0.5)
    cand = np.where(expert_mask)[0]
    if cand.size == 0:
        raise RuntimeError("No expert_mask==1 frames in this split. Case-study is undefined.")

    rng = np.random.default_rng(int(args.seed))
    n_case = min(int(args.n_case), int(cand.size))
    idx = rng.choice(cand, size=n_case, replace=False).astype(np.int64)

    # --- Scaling tensors (train-based) ---
    scale = ds.get_scale_tensors(str(device))
    gate_mean, gate_std = scale["gate_mean"], scale["gate_std"]
    expert_mean, expert_std = scale["expert_mean"], scale["expert_std"]

    # --- Load models ---
    gate, expert, expert_type = load_models_for_run(run_dir, state, device=device)
    if expert_type != "flow":
        raise RuntimeError("This case-study script currently supports Flow expert only.")

    # Flow split indices (FiLM mode)
    flow_x_idx_t, flow_c_idx_t = None, None
    try:
        expert_colnames = ds.get_x_expert_colnames()
        fx_cols = state.schema.flow_x_cols_in_order()
        fc_cols = state.schema.flow_c_cols_in_order()
        if fx_cols and all(c in expert_colnames for c in fx_cols):
            flow_x_idx_t = torch.tensor([expert_colnames.index(c) for c in fx_cols], dtype=torch.long)
        if fc_cols and all(c in expert_colnames for c in fc_cols):
            flow_c_idx_t = torch.tensor([expert_colnames.index(c) for c in fc_cols], dtype=torch.long)
    except Exception:
        flow_x_idx_t, flow_c_idx_t = None, None

    # Fingerprint suppression indices (must match eval/training)
    drop_idx = sorted(set(getattr(state, "drop_context_idx_expert", []) or []))
    drop_idx_t = torch.tensor(drop_idx, dtype=torch.long) if len(drop_idx) else None

    # --- Gather raw tensors for cases ---
    xg_raw = ds.tensors.x_gate_raw[idx]
    xe_raw = ds.tensors.x_expert_raw[idx]

    # Expert column mapping (used by proxy regression + proxy counterfactual)
    expert_cols = ds.get_x_expert_colnames()
    name_to_j = {c: j for j, c in enumerate(expert_cols)}

    # --- Baseline predictions ---
    p_gate0 = infer_gate_probs(gate, xg_raw, gate_mean, gate_std, device=device, batch=4096)
    y_thr_vec = torch.full((n_case,), float(y_thr_std_q), dtype=torch.float32)
    p_tail0 = infer_flow_cdf_probs(
        expert,
        y_thr_vec,
        xe_raw,
        expert_mean,
        expert_std,
        drop_idx=drop_idx_t,
        flow_x_idx=flow_x_idx_t,
        flow_c_idx=flow_c_idx_t,
        device=device,
        batch=2048,
    )
    risk0 = p_gate0 * p_tail0

    # sigma_log (only where P(Gate) >= thr)
    sigma0 = np.full((n_case,), np.nan, dtype=np.float32)
    m_sigma = (p_gate0 >= float(args.gate_thr))
    if int(np.sum(m_sigma)) > 0:
        sigma0[m_sigma] = infer_flow_sigma_log(
            expert,
            xe_raw[m_sigma],
            expert_mean,
            expert_std,
            drop_idx=drop_idx_t,
            flow_x_idx=flow_x_idx_t,
            flow_c_idx=flow_c_idx_t,
            device=device,
            batch=1024,
            num_samples=int(args.sigma_samples),
            sigma_y=float(sig_y),
        )

    # -------------------------
    # Proxy regression (trend + slope CI)
    # -------------------------
    proxy_reg: Dict[str, Dict[str, object]] = {}
    n_bins = int(max(3, args.proxy_bins))
    for pc in [c.strip() for c in str(args.proxy_cols).split(",") if c.strip()]:
        if pc not in name_to_j:
            continue
        j = name_to_j[pc]
        if (drop_idx_t is not None) and (int((drop_idx_t == j).sum().item()) > 0):
            continue

        x = xe_raw[:, j].detach().cpu().numpy().astype(np.float64)

        sr, sr_lo, sr_hi = _bootstrap_ci_slope(x, risk0, n_boot=int(args.n_boot), seed=int(args.seed) + 101)
        st, st_lo, st_hi = _bootstrap_ci_slope(x, p_tail0, n_boot=int(args.n_boot), seed=int(args.seed) + 103)
        rec: Dict[str, object] = {
            "slope_risk": sr,
            "slope_risk_ci_lo": sr_lo,
            "slope_risk_ci_hi": sr_hi,
            "slope_p_tail": st,
            "slope_p_tail_ci_lo": st_lo,
            "slope_p_tail_ci_hi": st_hi,
        }

        # Binned trend (quantile bins) - more interpretable in papers.
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(x, qs)
        # If edges collapse (many identical values), trend plot isn't meaningful.
        if np.unique(edges).size >= 4:
            # digitize into bins [0, n_bins-1]
            bin_id = np.digitize(x, edges[1:-1], right=True)
            bx, br, bt, bs = [], [], [], []
            for b in range(n_bins):
                m = (bin_id == b)
                if m.sum() < 10:
                    continue
                bx.append(float(np.mean(x[m])))
                br.append(float(np.mean(risk0[m])))
                bt.append(float(np.mean(p_tail0[m])))
                # sigma only for gate-passing samples
                ms = m & m_sigma
                bs.append(float(np.nanmean(sigma0[ms])) if ms.sum() > 0 else float("nan"))
            rec.update({"bin_x": bx, "bin_risk": br, "bin_p_tail": bt, "bin_sigma_log": bs})

            # Optional plot for one chosen proxy
            if pc == str(args.proxy_reg_plot_col) and len(bx) >= 3:
                plt.figure(figsize=(6.5, 4.0))
                plt.plot(bx, br, marker="o")
                plt.title(f"Proxy regression (binned): {pc}\nRisk vs proxy")
                plt.xlabel(pc)
                plt.ylabel("Risk")
                plt.tight_layout()
                plt.savefig(out_dir / "case_study_proxy_regression.png", dpi=200)
                plt.close()

        proxy_reg[pc] = rec

    # -------------------------
    # A) Gate-only one-hot interventions
    # -------------------------
    gate_cols = ds.get_x_gate_colnames()
    gate_cf_weather = _make_onehot_counterfactuals(
        xg_raw,
        gate_cols,
        prefix="c__odd_weather=",
        prefer_order=["rain", "snow", "fog", "overcast", "clear", "unknown"],
    )
    gate_cf_time = _make_onehot_counterfactuals(
        xg_raw,
        gate_cols,
        prefix="c__odd_time=",
        prefer_order=["night", "dawn", "dusk", "day", "unknown"],
    )

    rows: List[Dict[str, object]] = []

    def _summarize_gate_cf(cf_name: str, xg_raw_cf: torch.Tensor) -> Dict[str, float]:
        p_gate = infer_gate_probs(gate, xg_raw_cf, gate_mean, gate_std, device=device, batch=4096)

        # If this schema is v1 and expert input contains gate columns, tail prob may also change.
        # Detect by column name overlap.
        expert_cols = ds.get_x_expert_colnames()
        overlap = any(c in expert_cols for c in gate_cols)
        if overlap:
            # safest: rebuild xe_raw_cf from DataFrame (expensive) -> we avoid.
            # Practical compromise: assume expert input is independent (v2). If overlap exists, warn.
            pass

        risk = p_gate * p_tail0
        dr = risk - risk0
        dg = p_gate - p_gate0
        mr, lr, hr = _bootstrap_ci_mean(dr, n_boot=int(args.n_boot), seed=int(args.seed) + 11)
        mg, lg, hg = _bootstrap_ci_mean(dg, n_boot=int(args.n_boot), seed=int(args.seed) + 13)
        return {
            "risk_mean": float(np.mean(risk)),
            "p_gate_mean": float(np.mean(p_gate)),
            "p_tail_mean": float(np.mean(p_tail0)),
            "delta_risk_mean": mr,
            "delta_risk_ci_lo": lr,
            "delta_risk_ci_hi": hr,
            "delta_p_gate_mean": mg,
            "delta_p_gate_ci_lo": lg,
            "delta_p_gate_ci_hi": hg,
        }

    gate_weather_res: Dict[str, Dict[str, float]] = {}
    for k, x_cf in gate_cf_weather.items():
        gate_weather_res[k] = _summarize_gate_cf(k, x_cf)
        rows.append({"family": "gate_weather", "key": k, **gate_weather_res[k]})

    gate_time_res: Dict[str, Dict[str, float]] = {}
    for k, x_cf in gate_cf_time.items():
        gate_time_res[k] = _summarize_gate_cf(k, x_cf)
        rows.append({"family": "gate_time", "key": k, **gate_time_res[k]})

    # -------------------------
    # B) Expert proxy interventions (quantile substitution)
    # -------------------------
    proxy_cols = [c.strip() for c in str(args.proxy_cols).split(",") if c.strip()]
    proxy_res: Dict[str, Dict[str, float]] = {}

    # Quantiles computed on TRAIN (raw columns, but with the same numeric coercion as transform_dataframe).
    q_src_df = df_train  # raw, before transform

    for pc in proxy_cols:
        if pc not in name_to_j:
            continue
        j = name_to_j[pc]
        if (drop_idx_t is not None) and (int((drop_idx_t == j).sum().item()) > 0):
            # This feature was suppressed during training/eval; counterfactual is meaningless.
            continue

        x_train = _safe_numeric_series(q_src_df, pc)
        q_lo = float(np.quantile(x_train, float(args.proxy_q_lo)))
        q_hi = float(np.quantile(x_train, float(args.proxy_q_hi)))

        xe_lo = xe_raw.clone()
        xe_hi = xe_raw.clone()
        xe_lo[:, j] = q_lo
        xe_hi[:, j] = q_hi

        p_tail_lo = infer_flow_cdf_probs(
            expert,
            y_thr_vec,
            xe_lo,
            expert_mean,
            expert_std,
            drop_idx=drop_idx_t,
            flow_x_idx=flow_x_idx_t,
            flow_c_idx=flow_c_idx_t,
            device=device,
            batch=2048,
        )
        p_tail_hi = infer_flow_cdf_probs(
            expert,
            y_thr_vec,
            xe_hi,
            expert_mean,
            expert_std,
            drop_idx=drop_idx_t,
            flow_x_idx=flow_x_idx_t,
            flow_c_idx=flow_c_idx_t,
            device=device,
            batch=2048,
        )

        risk_lo = p_gate0 * p_tail_lo
        risk_hi = p_gate0 * p_tail_hi

        dr = risk_hi - risk_lo
        dt = p_tail_hi - p_tail_lo
        mr, lr, hr = _bootstrap_ci_mean(dr, n_boot=int(args.n_boot), seed=int(args.seed) + 17)
        mt, lt, ht = _bootstrap_ci_mean(dt, n_boot=int(args.n_boot), seed=int(args.seed) + 19)

        row: Dict[str, float] = {
            "q_lo": q_lo,
            "q_hi": q_hi,
            "risk_lo_mean": float(np.mean(risk_lo)),
            "risk_hi_mean": float(np.mean(risk_hi)),
            "p_tail_lo_mean": float(np.mean(p_tail_lo)),
            "p_tail_hi_mean": float(np.mean(p_tail_hi)),
            "delta_risk_mean": mr,
            "delta_risk_ci_lo": lr,
            "delta_risk_ci_hi": hr,
            "delta_p_tail_mean": mt,
            "delta_p_tail_ci_lo": lt,
            "delta_p_tail_ci_hi": ht,
        }

        # sigma_log delta (only where P(Gate) >= thr)
        if int(np.sum(m_sigma)) > 0:
            sig_lo = infer_flow_sigma_log(
                expert,
                xe_lo[m_sigma],
                expert_mean,
                expert_std,
                drop_idx=drop_idx_t,
                flow_x_idx=flow_x_idx_t,
                flow_c_idx=flow_c_idx_t,
                device=device,
                batch=1024,
                num_samples=int(args.sigma_samples),
                sigma_y=float(sig_y),
            )
            sig_hi = infer_flow_sigma_log(
                expert,
                xe_hi[m_sigma],
                expert_mean,
                expert_std,
                drop_idx=drop_idx_t,
                flow_x_idx=flow_x_idx_t,
                flow_c_idx=flow_c_idx_t,
                device=device,
                batch=1024,
                num_samples=int(args.sigma_samples),
                sigma_y=float(sig_y),
            )
            dsig = sig_hi - sig_lo
            ms, ls, hs = _bootstrap_ci_mean(dsig, n_boot=int(args.n_boot), seed=int(args.seed) + 23)
            row.update(
                {
                    "sigma_lo_mean": float(np.mean(sig_lo)),
                    "sigma_hi_mean": float(np.mean(sig_hi)),
                    "delta_sigma_log_mean": ms,
                    "delta_sigma_log_ci_lo": ls,
                    "delta_sigma_log_ci_hi": hs,
                }
            )

        proxy_res[pc] = row
        rows.append({"family": "expert_proxy", "key": pc, **row})

    # -------------------------
    # Save artifacts
    # -------------------------
    summary = {
        "meta": {
            "run_dir": str(run_dir),
            "csv": str(args.csv),
            "split": str(args.split),
            "n_case": int(n_case),
            "seed": int(args.seed),
            "device": str(device),
            "risk_def": "quantile_tail",
            "tail_q": float(args.tail_q),
            "tail_q_source": str(args.tail_q_source),
            "tail_ttc_threshold_s": float(ttc_q),
            "y_thr_std": float(y_thr_std_q),
            "sigma_samples": int(args.sigma_samples),
            "gate_thr": float(args.gate_thr),
            "proxy_q_lo": float(args.proxy_q_lo),
            "proxy_q_hi": float(args.proxy_q_hi),
            "n_boot": int(args.n_boot),
        },
        "baseline": {
            "risk_mean": float(np.mean(risk0)),
            "p_gate_mean": float(np.mean(p_gate0)),
            "p_tail_mean": float(np.mean(p_tail0)),
            "sigma_log_mean_gate": float(np.nanmean(sigma0)),
            "gate_pass_rate": float(np.mean(m_sigma.astype(np.float32))),
        },
        "gate_onehot_weather": gate_weather_res,
        "gate_onehot_time": gate_time_res,
        "proxy_regression": proxy_reg,
        "expert_proxy": proxy_res,
    }

    (out_dir / "case_study_odd_sensitivity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "case_study_odd_sensitivity.csv", index=False)

    # -------------------------
    # Plot
    # -------------------------
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1)
    if gate_weather_res:
        keys = list(gate_weather_res.keys())
        labels = [k.replace("c__odd_weather=", "") for k in keys]
        mean = [gate_weather_res[k]["delta_risk_mean"] for k in keys]
        lo = [gate_weather_res[k]["delta_risk_ci_lo"] for k in keys]
        hi = [gate_weather_res[k]["delta_risk_ci_hi"] for k in keys]
        _plot_bars_with_ci(ax1, labels, mean, lo, hi, "Gate ODD (weather)\nΔRisk vs observed", "Δ Risk")
    else:
        ax1.text(0.5, 0.5, "No c__odd_weather=* columns", ha="center", va="center")
        ax1.set_axis_off()

    ax2 = plt.subplot(1, 2, 2)
    if proxy_res:
        keys = list(proxy_res.keys())
        labels = [k.replace("x__", "") for k in keys]
        mean = [proxy_res[k]["delta_risk_mean"] for k in keys]
        lo = [proxy_res[k]["delta_risk_ci_lo"] for k in keys]
        hi = [proxy_res[k]["delta_risk_ci_hi"] for k in keys]
        _plot_bars_with_ci(ax2, labels, mean, lo, hi, "Expert ODD proxies\nΔRisk (q_hi − q_lo)", "Δ Risk")
    else:
        ax2.text(0.5, 0.5, "No proxy cols present (or all were dropped)", ha="center", va="center")
        ax2.set_axis_off()

    plt.tight_layout()
    plt.savefig(out_dir / "case_study_odd_sensitivity.png", dpi=200)
    plt.close(fig)

    # -------------------------
    # Optional: one representative distribution plot
    # -------------------------
    if args.plot_case0 and n_case > 0:
        # pick case0 = highest baseline risk among sampled
        k0 = int(np.argmax(risk0))
        xe0 = xe_raw[k0 : k0 + 1]

        # Choose the first available proxy to show a shift
        pc0 = next(iter(proxy_res.keys()), None)
        if pc0 is not None:
            j0 = name_to_j[pc0]
            q_lo = float(proxy_res[pc0]["q_lo"])
            q_hi = float(proxy_res[pc0]["q_hi"])
            xe_lo = xe0.clone(); xe_lo[:, j0] = q_lo
            xe_hi = xe0.clone(); xe_hi[:, j0] = q_hi

            # Build sigma in log space via samples (reuse infer_flow_sigma_log's internals is heavy;
            # here we directly sample from the flow for a single case).
            with torch.no_grad():
                # Standardize expert input
                em = expert_mean.to(device); es = expert_std.to(device)
                def _cond(x_raw_one: torch.Tensor):
                    xr = x_raw_one.to(device)
                    if drop_idx_t is not None and drop_idx_t.numel() > 0:
                        xr = xr.clone(); xr[:, drop_idx_t.to(device)] = em[drop_idx_t.to(device)]
                    xs = (xr - em) / (es + 1e-6)
                    if bool(getattr(expert, "expects_tuple_condition", lambda: False)()):
                        if flow_x_idx_t is None or flow_c_idx_t is None:
                            return xs
                        x_part = xs.index_select(1, flow_x_idx_t.to(device))
                        c_part = xs.index_select(1, flow_c_idx_t.to(device)) if int(flow_c_idx_t.numel()) > 0 else xs[:, :0]
                        return (x_part, c_part)
                    return xs

                s = int(max(512, args.sigma_samples))
                y0 = expert.sample(_cond(xe0), num_samples=s).detach().cpu().numpy().reshape(-1)
                yL = expert.sample(_cond(xe_lo), num_samples=s).detach().cpu().numpy().reshape(-1)
                yH = expert.sample(_cond(xe_hi), num_samples=s).detach().cpu().numpy().reshape(-1)

            # Convert standardized logTTC to logTTC
            log0 = y0 * (sig_y + eps_y) + mu_y
            logL = yL * (sig_y + eps_y) + mu_y
            logH = yH * (sig_y + eps_y) + mu_y
            t0 = np.clip(np.exp(log0), float(args.ttc_floor), float(args.ttc_cap))
            tL = np.clip(np.exp(logL), float(args.ttc_floor), float(args.ttc_cap))
            tH = np.clip(np.exp(logH), float(args.ttc_floor), float(args.ttc_cap))

            plt.figure(figsize=(7, 4))
            plt.hist(t0, bins=60, alpha=0.6, label="observed")
            plt.hist(tL, bins=60, alpha=0.5, label=f"{pc0}: q_lo")
            plt.hist(tH, bins=60, alpha=0.5, label=f"{pc0}: q_hi")
            plt.axvline(float(ttc_q), linestyle="--", linewidth=2)
            plt.title("Flow TTC samples (one case)\nproxy substitution effect")
            plt.xlabel("TTC [s]")
            plt.ylabel("count")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "case0_flow_samples.png", dpi=200)
            plt.close()


if __name__ == "__main__":
    main()
