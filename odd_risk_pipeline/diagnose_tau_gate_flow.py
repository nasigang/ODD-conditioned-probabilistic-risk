#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostics for:
  1) Tau sweep (threshold sensitivity / calibration)
  2) Gate recall bottleneck w.r.t. y_event
  3) Flow tail diagnostics (PIT hist + conditional TTC PDF low vs high)

Built by reusing the logic/implementations from eval_risk_models_v6.5.py (v6.5).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import torch


def _flow_expects_tuple(flow) -> bool:
    fn = getattr(flow, "expects_tuple_condition", None)
    if callable(fn):
        return bool(fn())
    return bool((flow.expects_tuple_condition() if callable(getattr(flow, 'expects_tuple_condition', None)) else bool(getattr(flow, 'expects_tuple_condition', False))))

def _make_flow_cond(x_expert_std: torch.Tensor, flow: torch.nn.Module, flow_x_idx=None, flow_c_idx=None):
    # Prepare conditioning for both concat and FiLM Flow.
    if _flow_expects_tuple(flow):
        if flow_x_idx is None or flow_c_idx is None or len(flow_x_idx) == 0 or len(flow_c_idx) == 0:
            raise RuntimeError("FiLM Flow requires non-empty flow_x_idx and flow_c_idx (check preprocess_state.json and --flow_feature_split).")
        return (x_expert_std[:, flow_x_idx], x_expert_std[:, flow_c_idx])
    return x_expert_std
import re
from typing import Dict, List, Tuple, Iterable

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


# sklearn optional
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
# Labels / thresholds (same as eval_risk_models_v6.5.py)
# ----------------------------
def compute_ttc(df: pd.DataFrame, eps: float = 1e-6) -> np.ndarray:
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
    ttc = compute_ttc(df, eps=1e-6)
    ttc = np.clip(ttc, ttc_floor, ttc_cap)

    if label_mode == "gate_clean":
        if y_gate_proxy is None:
            y_gate_proxy = (
                pd.to_numeric(df.get("n_edges", 0), errors="coerce")
                .fillna(0.0)
                .to_numpy() > 0.0
            ).astype(np.int64)
        return (y_gate_proxy > 0).astype(np.int64)

    if label_mode == "ttc_fixed":
        y = (ttc <= float(fixed_ttc)).astype(np.int64)
    elif label_mode == "ttc_sstar":
        if sstar_mode == "ego_speed":
            v = pd.to_numeric(df.get("x__ego_speed_mps", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float64)
        else:
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
    if label_mode == "ttc_fixed":
        return np.full((len(df),), float(fixed_ttc), dtype=np.float64)
    if label_mode == "gate_clean":
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
# Run artifacts (same as eval_risk_models_v6.5.py)
# ----------------------------
def load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_id_list(v: Any) -> Optional[List[str]]:
    if isinstance(v, (list, tuple, set)):
        return [str(x) for x in v]
    if isinstance(v, dict):
        for kk in ("segment_ids", "segments", "ids"):
            vv = v.get(kk, None)
            if isinstance(vv, (list, tuple, set)):
                return [str(x) for x in vv]
    return None


def load_segment_split(run_dir: Path) -> Dict[str, List[str]]:
    p = run_dir / "segment_split.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    d = load_json(p)
    out: Dict[str, List[str]] = {}
    for k in ("train", "val", "test"):
        if k in d:
            ids = _as_id_list(d[k])
            if ids is not None:
                out[k] = ids
    if not out and any((k.endswith("_segments") for k in d.keys())):
        for k in ("train", "val", "test"):
            kk = f"{k}_segments"
            if kk in d:
                ids = _as_id_list(d[kk])
                if ids is not None:
                    out[k] = ids
    if not out:
        raise TypeError(f"{p} has no usable split lists. Keys={list(d.keys())[:20]}")
    return out


def pick_split_df(df: pd.DataFrame, segs: List[str]) -> pd.DataFrame:
    segset = set(map(str, segs))
    return df[df["segment_id"].astype(str).isin(segset)].copy()


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in sd.keys()):
        return sd
    return {k[len("module.") :]: v for k, v in sd.items()}


def _torch_load_sd(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location=device)

    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"{path} is not a state_dict (type={type(obj)})")
    obj = _strip_module_prefix(obj)
    return obj


def _infer_mlp_dims_from_sd(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    w2 = [(k, v) for k, v in sd.items() if k.endswith("weight") and getattr(v, "ndim", 0) == 2]
    if not w2:
        raise RuntimeError("Could not infer MLP dims: no 2D weights found.")
    first_w = w2[0][1]
    hidden = int(first_w.shape[0])
    in_dim = int(first_w.shape[1])
    n_linear = len(w2)
    depth = max(1, n_linear - 1)
    return in_dim, hidden, depth


def _infer_flow_hparams_from_sd(flow_sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    lin_keys = [k for k, v in flow_sd.items() if k.startswith("net.") and k.endswith(".weight") and getattr(v, "ndim", 0) == 2]
    if not lin_keys:
        raise RuntimeError("Could not find flow linear weights under net.*.weight")

    def _net_idx(k: str) -> int:
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
    from risk_pipeline.models import GateMLP, ConditionalSpline1DFlow  # type: ignore

    gate_ckpt = run_dir / "gate.pt"
    if not gate_ckpt.exists():
        raise FileNotFoundError(f"Missing {gate_ckpt}")
    gate_sd = _torch_load_sd(gate_ckpt, device=torch.device("cpu"))
    in_dim, hidden, depth = _infer_mlp_dims_from_sd(gate_sd)
    gate = GateMLP(in_dim, hidden=hidden, depth=depth, dropout=0.0).to(device)
    gate.load_state_dict(gate_sd, strict=True)
    gate.eval()

    flow_ckpt = run_dir / "expert_flow.pt"
    if not flow_ckpt.exists():
        legacy = run_dir / "expert.pt"
        if legacy.exists():
            flow_ckpt = legacy

    if flow_ckpt.exists():
        flow_sd = _torch_load_sd(flow_ckpt, device=torch.device("cpu"))
        ctx_dim = len(state.schema.x_expert_cols_in_order())
        f_hidden, f_depth, num_bins = _infer_flow_hparams_from_sd(flow_sd)
        expert = ConditionalSpline1DFlow(cond_dim=ctx_dim, num_bins=num_bins, hidden=f_hidden, depth=f_depth, dropout=0.0).to(device)
        expert.load_state_dict(flow_sd, strict=True)
        expert.eval()
        return gate, expert, "flow"

    raise FileNotFoundError(f"Cannot find expert checkpoint in {run_dir}")


# ----------------------------
# Inference helpers (same as eval_risk_models_v6.5.py)
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
        cdf = flow.cdf(yb, xe_cond).reshape(-1)
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

        mask_u = ~is_c
        if mask_u.any():
            loss_val[mask_u] = -flow.log_prob(yb[mask_u], xb[mask_u])
            pit = flow.cdf(yb[mask_u], xb[mask_u])
            pits.append(pit.detach().cpu())

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
# Diagnostics helpers (new)
# ----------------------------
def _parse_float_list(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return []
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out


def _parse_tau_grid(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(":")]
    if len(parts) != 3:
        raise ValueError("--tau_grid must be 'start:stop:step'")
    a, b, step = map(float, parts)
    if step <= 0:
        raise ValueError("--tau_grid step must be > 0")
    vals = []
    x = a
    for _ in range(10000):
        if x > b + 1e-12:
            break
        vals.append(float(x))
        x += step
    return vals


def gate_event_threshold_curve(y_event: np.ndarray, p_gate: np.ndarray, thresholds: List[float]) -> List[dict]:
    y = y_event.astype(np.int64)
    P = int(np.sum(y))
    out = []
    for thr in thresholds:
        pred = (p_gate >= float(thr))
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum((~pred) & (y == 1)))
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        out.append(
            {
                "thr": float(thr),
                "P": P,
                "pred_pos": int(tp + fp),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": prec,
                "recall": rec,
                "pass_rate": float(np.mean(pred)),
                "blocked_pos_rate": float(fn / P) if P > 0 else float("nan"),
            }
        )
    return out


def threshold_for_target_recall(curve: List[dict], target_recall: float) -> Optional[float]:
    best = None
    for row in curve:
        r = row.get("recall", float("nan"))
        if not (isinstance(r, float) and np.isfinite(r)):
            continue
        if r >= target_recall:
            t = float(row["thr"])
            if best is None or t > best:
                best = t
    return best


def flow_tail_diagnostics(
    *,
    flow,
    ds,
    df_t: pd.DataFrame,
    state,
    expert_mean: torch.Tensor,
    expert_std: torch.Tensor,
    drop_idx_t: Optional[torch.Tensor],
    device: torch.device,
    batch: int,
    out_dir: Path,
    ttc_floor: float,
    ttc_cap: float,
    ttc_cut: float,
    ttc_hi_cut: float,
    max_samples: int,
    grid_n: int,
) -> Dict[str, Any]:
    """
    Saves:
      - pit_hist.png
      - ttc_pdf_low_vs_high.png
      - pit_stats.json

    Returns a dict with summary stats (also useful for JSON aggregation).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        return {"skipped": True, "reason": f"matplotlib not available: {e}"}

    # --- PIT collection (uncensored only, expert_mask==1) ---
    pits = []
    n = len(ds)
    flow.eval()

    expert_mean_d = expert_mean.to(device)
    expert_std_d = expert_std.to(device)
    y_std_d = ds.tensors.y_expert.to(device)
    mask_d = ds.tensors.expert_mask.to(device)
    cens_d = ds.tensors.censored_mask.to(device)
    drop_d = drop_idx_t.to(device) if (drop_idx_t is not None and drop_idx_t.numel() > 0) else None

    for i0 in range(0, n, batch):
        i1 = min(n, i0 + batch)
        m = (mask_d[i0:i1] > 0.5) & (cens_d[i0:i1] < 0.5)
        if int(m.sum().item()) < 2:
            continue
        xe_raw = ds.tensors.x_expert_raw[i0:i1].to(device)
        if drop_d is not None:
            xe_raw = xe_raw.clone()
            xe_raw[:, drop_d] = expert_mean_d[drop_d]
        xe = (xe_raw - expert_mean_d) / (expert_std_d + 1e-6)
        yb = y_std_d[i0:i1][m]
        xb = xe[m]
        pit = flow.cdf(yb, xb_cond).reshape(-1)
        pits.append(pit.detach().cpu().numpy())

    pit_all = np.concatenate(pits, axis=0) if pits else np.zeros((0,), dtype=np.float32)

    if pit_all.size > 0:
        plt.figure()
        plt.hist(pit_all, bins=20)
        plt.title("PIT histogram (uncensored only)")
        plt.xlabel("PIT = CDF(y_true | x)")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / "pit_hist.png", dpi=160)
        plt.close()

        pit_sorted = np.sort(pit_all)
        u = (np.arange(1, pit_sorted.size + 1) - 0.5) / pit_sorted.size
        ks = float(np.max(np.abs(pit_sorted - u)))

        (out_dir / "pit_stats.json").write_text(
            json.dumps(
                {
                    "N_pit": int(pit_all.size),
                    "pit_mean": float(np.mean(pit_all)),
                    "pit_std": float(np.std(pit_all)),
                    "ks_uniform": ks,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        (out_dir / "pit_stats.json").write_text(json.dumps({"N_pit": 0, "note": "no uncensored expert samples"}, indent=2), encoding="utf-8")

    # --- Conditional TTC PDF: low vs high TTC ---
    ttc_raw = compute_ttc(df_t, eps=1e-6)
    ttc_clip = np.clip(ttc_raw, ttc_floor, ttc_cap)

    base = (ds.tensors.expert_mask.detach().cpu().numpy() > 0.5) & (ds.tensors.censored_mask.detach().cpu().numpy() < 0.5)
    m_low = base & (ttc_clip <= float(ttc_cut))
    m_hi  = base & (ttc_clip >= float(ttc_hi_cut))

    rng = np.random.default_rng(0)
    idx_low = np.where(m_low)[0]
    idx_hi  = np.where(m_hi)[0]
    if idx_low.size > max_samples:
        idx_low = rng.choice(idx_low, size=max_samples, replace=False)
    if idx_hi.size > max_samples:
        idx_hi = rng.choice(idx_hi, size=max_samples, replace=False)

    t_grid = np.linspace(ttc_floor, ttc_cap, int(grid_n))
    y_grid = np.log(t_grid + 1e-9)

    mu_y = float(state.target_std.mu_y)
    sig_y = float(state.target_std.sigma_y)
    eps_y = float(getattr(state.target_std, "eps", 1e-6))
    y_grid_std = (y_grid - mu_y) / (sig_y + eps_y)

    y_grid_std_t = torch.tensor(y_grid_std, dtype=torch.float32, device=device)

    def _avg_pdf(idxs: np.ndarray) -> Optional[np.ndarray]:
        if idxs.size == 0:
            return None
        xe_raw = ds.tensors.x_expert_raw[idxs].to(device)
        if drop_d is not None:
            xe_raw = xe_raw.clone()
            xe_raw[:, drop_d] = expert_mean_d[drop_d]
        xe = (xe_raw - expert_mean_d) / (expert_std_d + 1e-6)  # [B,C]
        B = xe.shape[0]
        G = y_grid_std_t.shape[0]

        xe_rep = xe[:, None, :].expand(B, G, xe.shape[1]).reshape(B * G, xe.shape[1])
        y_rep  = y_grid_std_t[None, :].expand(B, G).reshape(B * G)

        logp_y_std = flow.log_prob(y_rep, xe_rep).reshape(B, G)  # density w.r.t y_std

        # Convert to TTC density: p(t) = p(y_std) * (1/sigma) * (1/t)
        p_t = torch.exp(logp_y_std) / (sig_y + eps_y)
        p_t = p_t / torch.tensor(t_grid, dtype=torch.float32, device=device)[None, :]

        return p_t.mean(dim=0).detach().cpu().numpy()

    pdf_low = _avg_pdf(idx_low)
    pdf_hi = _avg_pdf(idx_hi)

    plt.figure()
    if pdf_low is not None:
        plt.plot(t_grid, pdf_low, label=f"TTC <= {ttc_cut}s (n={int(idx_low.size)})")
    if pdf_hi is not None:
        plt.plot(t_grid, pdf_hi, label=f"TTC >= {ttc_hi_cut}s (n={int(idx_hi.size)})")
    plt.title("Conditional predicted TTC PDF (flow)")
    plt.xlabel("TTC (s)")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "ttc_pdf_low_vs_high.png", dpi=160)
    plt.close()

    return {
        "skipped": False,
        "N_pit": int(pit_all.size),
        "N_low": int(idx_low.size),
        "N_high": int(idx_hi.size),
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--run", action="append", required=True, help="Run directory (repeatable)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])

    ap.add_argument("--ttc_floor", type=float, default=0.05)
    ap.add_argument("--ttc_cap", type=float, default=8.0)

    ap.add_argument("--x_to_c_regex", type=str, default="",
                    help="Comma-separated regex. Matching x__* columns will be renamed to c__* before preprocessing.")
    ap.add_argument("--expert_drop_feature_regex", type=str, default="",
                    help="Comma-separated regex. Matching expert feature columns will be hard-dropped (set to train-mean) for flow, in diagnostics.")

    ap.add_argument("--label_mode", default="ttc_sstar", choices=["ttc_sstar", "ttc_fixed", "gate_clean"])
    ap.add_argument("--sstar_mode", default="closing_speed", choices=["closing_speed", "ego_speed"])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--amax", type=float, default=6.0)
    ap.add_argument("--fixed_ttc", type=float, default=2.0)

    ap.add_argument("--require_edges", action="store_true", default=True)
    ap.add_argument("--no_require_edges", dest="require_edges", action="store_false")

    # ---- new: tau sweep ----
    ap.add_argument("--tau_sweep", default=None, help="Comma-separated tau list (e.g., 0.1,0.3,0.5,1.0)")
    ap.add_argument("--tau_grid", default=None, help="Grid start:stop:step (e.g., 0.1:1.0:0.1)")

    # ---- new: gate recall bottleneck ----
    ap.add_argument("--gate_thresholds", default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5")
    ap.add_argument("--gate_target_recall", type=float, default=0.90)

    # TTC-bin diagnostic
    ap.add_argument("--ttc_bins", type=int, default=6)

    # performance/output
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--out_dir", default=None, help="If set, save under <out_dir>/<run_name>/ ...")
    ap.add_argument("--save_preds", action="store_true")
    ap.add_argument("--no_save", action="store_true")

    # ---- new: flow tail diagnostics ----
    ap.add_argument("--flow_diag_dir", default=None)
    ap.add_argument("--flow_ttc_cut", type=float, default=3.0)
    ap.add_argument("--flow_ttc_hi_cut", type=float, default=5.0)
    ap.add_argument("--flow_diag_max_samples", type=int, default=512)
    ap.add_argument("--flow_diag_grid_n", type=int, default=200)

    args = ap.parse_args()

    df_all = pd.read_csv(args.csv)
    assert "segment_id" in df_all.columns
    assert ("y_soft" in df_all.columns) or ("min_ttc_est" in df_all.columns)

    device = torch.device(args.device)

    from risk_pipeline.schema import load_schema
    from risk_pipeline.preprocess import load_preprocess_state, transform_dataframe
    from risk_pipeline.data import RiskCSVDataset

    # tau list
    if args.tau_sweep:
        tau_list = _parse_float_list(args.tau_sweep)
    elif args.tau_grid:
        tau_list = _parse_tau_grid(args.tau_grid)
    else:
        tau_list = [float(args.tau)]
    if not tau_list:
        tau_list = [float(args.tau)]
    sweep_mode = len(tau_list) > 1

    gate_thr_list = _parse_float_list(args.gate_thresholds)
    if not gate_thr_list:
        gate_thr_list = [0.1, 0.2, 0.3, 0.5]

    all_results: Dict[str, dict] = {}
    summary_rows: List[dict] = []

    for run in args.run:
        run_dir = Path(run)
        split = load_segment_split(run_dir)
        segs = split[args.split]
        df_split = pick_split_df(df_all, segs)

        if "frame_label" not in df_split.columns:
            df_split["frame_label"] = np.arange(len(df_split), dtype=np.int64)

        _ = load_schema(str(run_dir / "feature_schema.json"))
        state = load_preprocess_state(str(run_dir / "preprocess_state.json"))
        df_t = transform_dataframe(df_split, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)

        ds = RiskCSVDataset(df_t, state, ttc_floor=args.ttc_floor, ttc_cap=args.ttc_cap)

        scale = ds.get_scale_tensors("cpu")
        gate_mean, gate_std = scale["gate_mean"], scale["gate_std"]
        expert_mean, expert_std = scale["expert_mean"], scale["expert_std"]

        drop_idx = sorted(set(getattr(state, "drop_context_idx_expert", []) or []))
        extra_drop_patterns = _parse_csv_list(args.expert_drop_feature_regex)
        if extra_drop_patterns:
            expert_colnames = ds.get_x_expert_colnames()
            extra_cols = _match_cols_by_regex(expert_colnames, extra_drop_patterns)
            if extra_cols:
                extra_idx = [expert_colnames.index(c) for c in extra_cols]
                print(f"[expert_drop_feature_regex] hard-drop {len(extra_idx)} expert cols. Example: {extra_cols[:10]}")
                drop_idx = sorted(set(drop_idx + extra_idx))
        drop_idx_t = torch.tensor(drop_idx, dtype=torch.long) if len(drop_idx) else None

        gate, expert, expert_type = load_models_for_run(run_dir, state, device=device)
        if expert_type != "flow":
            raise RuntimeError("This diagnostics script currently expects flow expert.")

        # gate inference (tau-independent)
        p_gate = infer_gate_probs(gate, ds.tensors.x_gate_raw, gate_mean, gate_std, device=device, batch=args.batch)

        # expert diagnostics (tau-independent)
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

        # optional flow plots
        if args.flow_diag_dir:
            flow_out = Path(args.flow_diag_dir) / run_dir.name / args.split
            diag_stats = flow_tail_diagnostics(
                flow=expert,
                ds=ds,
                df_t=df_t,
                state=state,
                expert_mean=expert_mean,
                expert_std=expert_std,
                drop_idx_t=drop_idx_t,
                device=device,
                batch=args.batch,
                out_dir=flow_out,
                ttc_floor=args.ttc_floor,
                ttc_cap=args.ttc_cap,
                ttc_cut=args.flow_ttc_cut,
                ttc_hi_cut=args.flow_ttc_hi_cut,
                max_samples=args.flow_diag_max_samples,
                grid_n=args.flow_diag_grid_n,
            )
        else:
            diag_stats = {"skipped": True}

        # output base
        if args.out_dir:
            out_base = Path(args.out_dir) / run_dir.name
        else:
            out_base = run_dir
        out_base.mkdir(parents=True, exist_ok=True)

        # sweep results table
        sweep_table = []

        y_gate_true = ds.tensors.y_gate.detach().cpu().numpy().astype(np.int64)

        for tau_val in tau_list:
            # y_event depends on tau
            y_event = build_event_label(
                df_t,
                ttc_floor=args.ttc_floor,
                ttc_cap=args.ttc_cap,
                label_mode=args.label_mode,
                sstar_mode=args.sstar_mode,
                tau=float(tau_val),
                amax=args.amax,
                fixed_ttc=args.fixed_ttc,
                require_edges=args.require_edges,
                y_gate_proxy=y_gate_true,
            )

            # threshold -> standardized log TTC
            thr_ttc = build_event_threshold_ttc(
                df_t,
                ttc_cap=args.ttc_cap,
                label_mode=args.label_mode,
                sstar_mode=args.sstar_mode,
                tau=float(tau_val),
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

            # p_event via exact CDF
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

            risk_prob = np.clip(p_gate * p_event, 0.0, 1.0)

            gate_curve = gate_event_threshold_curve(y_event, p_gate, gate_thr_list)
            thr_for_target = threshold_for_target_recall(gate_curve, float(args.gate_target_recall))

            # TTC-bin diag
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

            res = {
                "run": str(run_dir),
                "run_name": run_dir.name,
                "split": args.split,
                "tau": float(tau_val),
                "amax": float(args.amax),
                "N": int(len(df_t)),
                "pos_rate": float(np.mean(y_event)),

                # gate metrics (training label)
                "gate_pr_auc": _safe_prauc(y_gate_true, p_gate),
                "gate_roc_auc": _safe_auc(y_gate_true, p_gate),
                "gate_brier": brier(y_gate_true, p_gate),
                "gate_ece": ece(y_gate_true, p_gate),

                # risk metrics (event label)
                "risk_label_mode": args.label_mode,
                "risk_pr_auc": _safe_prauc(y_event, risk_prob),
                "risk_roc_auc": _safe_auc(y_event, risk_prob),
                "risk_brier": brier(y_event, risk_prob),
                "risk_ece": ece(y_event, risk_prob),

                # expert diagnostics (tau-independent)
                "expert_type": "flow",
                "expert_nll": float(expert_nll),
                "pit_mean": float(pit_mean),
                "pit_std": float(pit_std),

                # summary stats
                "risk_mean": float(np.mean(risk_prob)),
                "risk_p95": float(np.quantile(risk_prob, 0.95)),
                "p_gate_mean": float(np.mean(p_gate)),
                "p_event_mean": float(np.mean(p_event)),

                # new diags
                "gate_event_threshold_curve": gate_curve,
                "gate_thr_for_target_recall": thr_for_target,
                "ttc_bin_diag": bin_rows,
                "flow_diag": diag_stats,
            }

            # save per tau
            if not args.no_save:
                tau_tag = f"_tau{tau_val:g}" if sweep_mode else ""
                out_json = out_base / f"diag_{args.split}_{args.label_mode}{tau_tag}.json"
                out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")

                if args.save_preds:
                    seg_id = df_t["segment_id"].astype(str).to_numpy()
                    frame = pd.to_numeric(df_t["frame_label"], errors="coerce").fillna(-1).to_numpy(np.int64)
                    out_npz = out_base / f"preds_{args.split}_{args.label_mode}{tau_tag}.npz"
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

            key = f"{str(run_dir)}::tau={tau_val:g}" if sweep_mode else str(run_dir)
            all_results[key] = res

            sweep_table.append(
                {
                    "run_name": run_dir.name,
                    "split": args.split,
                    "tau": float(tau_val),
                    "pos_rate": float(np.mean(y_event)),
                    "risk_pr_auc": res["risk_pr_auc"],
                    "risk_brier": res["risk_brier"],
                    "risk_ece": res["risk_ece"],
                    "p_gate_mean": res["p_gate_mean"],
                    "p_event_mean": res["p_event_mean"],
                    "gate_thr_for_target_recall": res["gate_thr_for_target_recall"],
                }
            )

            thr_s = f"{thr_for_target:.3f}" if (thr_for_target is not None and np.isfinite(thr_for_target)) else "NA"
            print(
                f"[OK] {run_dir.name} split={args.split} tau={tau_val:g} "
                f"pos={res['pos_rate']:.4f} PR-AUC={res['risk_pr_auc']:.4f} "
                f"Brier={res['risk_brier']:.4f} p_gate_mean={res['p_gate_mean']:.3f} "
                f"p_event_mean={res['p_event_mean']:.3f} gate_thr@recall{args.gate_target_recall:.2f}={thr_s}"
            )

        # save sweep csv
        if not args.no_save:
            csv_path = out_base / f"tau_sweep_{args.split}_{args.label_mode}.csv"
            pd.DataFrame(sweep_table).to_csv(csv_path, index=False)

        summary_rows.extend(sweep_table)

    print("\n===== SUMMARY (per run,tau) =====")
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        print(df_sum.sort_values(["run_name", "tau"]).to_string(index=False))


if __name__ == "__main__":
    main()
