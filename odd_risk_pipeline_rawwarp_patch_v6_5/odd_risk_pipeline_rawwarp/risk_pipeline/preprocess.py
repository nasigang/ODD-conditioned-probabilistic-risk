from __future__ import annotations

"""Preprocessing utilities for the ODD-conditioned risk pipeline.

Design goals
------------
1) **Raw-space warp compatibility**
   - We must keep *raw physical units* in the Dataset so that
     `warp -> scaler.transform` ordering is possible.
   - Therefore, `transform_dataframe()` only creates numeric columns + targets,
     and does **NOT** apply z-score scaling.

2) **Safety Pins**
   - Gate label cleaning (distance/time horizon): `y_gate` uses
     (n_edges > 0) AND (TTC valid) AND (TTC < ttc_cap)
   - Expert target stabilization: TTC floor/cap + log + train-standardization
   - Scaler persistence: Train mean/std are saved and reused.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd

from .schema import FeatureSchema


def _is_binary_series(s: pd.Series) -> bool:
    v = s.dropna().unique()
    if len(v) == 0:
        return False
    if len(v) <= 2 and set(v).issubset({0, 1, 0.0, 1.0, True, False}):
        return True
    return False


@dataclass
class VarianceFilterState:
    kept_columns: List[str]
    dropped_constant: List[str]


@dataclass
class ScalerState:
    """Z-score parameters for *continuous* columns only."""

    means: Dict[str, float]
    stds: Dict[str, float]
    eps: float = 1e-6


@dataclass
class TargetStandardizerState:
    """Standardization parameters for the Expert target y_expert."""

    mu_y: float
    sigma_y: float
    eps: float = 1e-6


@dataclass
class PreprocessState:
    schema: FeatureSchema
    var_filter: VarianceFilterState
    scaler: ScalerState
    target_std: TargetStandardizerState
    continuous_cols: List[str]
    binary_cols: List[str]
    onehot_cols: List[str]
    expert_extra_cols: List[str]
    # Expert-only fingerprint suppression.
    # Indices are with respect to schema.x_expert_cols_in_order().
    drop_context_cols_expert: List[str] = None
    drop_context_idx_expert: List[int] = None

    def to_json(self) -> str:
        d = asdict(self)
        d["schema"] = json.loads(self.schema.to_json())
        return json.dumps(d, indent=2)

    @staticmethod
    def from_json(s: str) -> "PreprocessState":
        d = json.loads(s)
        schema = FeatureSchema(**d["schema"])
        d["schema"] = schema
        d["var_filter"] = VarianceFilterState(**d["var_filter"])
        d["scaler"] = ScalerState(**d["scaler"])
        d["target_std"] = TargetStandardizerState(**d["target_std"])
        # Backward compatibility: older saved states may not include these.
        d.setdefault("drop_context_cols_expert", [])
        d.setdefault("drop_context_idx_expert", [])
        return PreprocessState(**d)


def compute_ttc_from_y_soft(y_soft: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert TTCI (=1/TTC) to TTC."""

    return 1.0 / (y_soft + eps)


def clip_ttc(ttc: np.ndarray, ttc_floor: float, ttc_cap: float) -> np.ndarray:
    return np.clip(ttc, ttc_floor, ttc_cap)


def log_ttc(ttc: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return np.log(ttc + eps)


def standardize_target(y: np.ndarray, ts: TargetStandardizerState) -> np.ndarray:
    return (y - ts.mu_y) / (ts.sigma_y + ts.eps)


def _apply_count_transforms_inplace(df: pd.DataFrame) -> None:
    """Apply monotonic transforms for heavy-tailed count features."""

    for c in ["n_edges", "n_flagged"]:
        if c in df.columns:
            df[c] = np.log1p(pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0))


def _compute_gate_label(df: pd.DataFrame, *, ttc_floor: float, ttc_cap: float, eps: float) -> np.ndarray:
    """Gate label with a physically meaningful horizon.

    y_gate = (n_edges > 0) AND (TTC valid) AND (TTC < ttc_cap)

    Notes
    -----
    - `n_edges` may already be log1p-transformed; the condition ( > 0 ) is stable.
    - If `min_ttc_est` exists, we prefer it over y_soft.
    """

    if "n_edges" not in df.columns:
        return np.zeros((len(df),), dtype=np.int64)

    edge = pd.to_numeric(df["n_edges"], errors="coerce").fillna(0.0).to_numpy(np.float64) > 0.0

    if "min_ttc_est" in df.columns:
        ttc_raw = pd.to_numeric(df["min_ttc_est"], errors="coerce").to_numpy(np.float64)
        ttc_valid = np.isfinite(ttc_raw) & (ttc_raw > 0.0)
        ttc_for_cut = ttc_raw
    elif "y_soft" in df.columns:
        y_soft = pd.to_numeric(df["y_soft"], errors="coerce").to_numpy(np.float64)
        ttc_valid = np.isfinite(y_soft) & (y_soft > 0.0)
        ttc_for_cut = compute_ttc_from_y_soft(np.where(ttc_valid, y_soft, 0.0), eps=eps)
    else:
        # No TTC proxy available -> Gate cannot be cleaned
        ttc_valid = np.zeros((len(df),), dtype=bool)
        ttc_for_cut = np.full((len(df),), np.inf, dtype=np.float64)

    # IMPORTANT: compare against *raw* TTC (no clipping for the cut)
    y_gate = (edge & ttc_valid & (ttc_for_cut < float(ttc_cap))).astype(np.int64)
    return y_gate


def fit_variance_filter(df_train: pd.DataFrame, feature_cols: List[str], var_eps: float = 1e-10) -> VarianceFilterState:
    dropped, kept = [], []
    for c in feature_cols:
        if c not in df_train.columns:
            continue
        x = pd.to_numeric(df_train[c], errors="coerce").to_numpy(np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            dropped.append(c)
            continue
        v = float(np.var(x))
        if v < var_eps:
            dropped.append(c)
        else:
            kept.append(c)
    return VarianceFilterState(kept_columns=kept, dropped_constant=dropped)


def fit_scaler(df_train: pd.DataFrame, continuous_cols: List[str], eps: float = 1e-6) -> ScalerState:
    means, stds = {}, {}
    for c in continuous_cols:
        x = pd.to_numeric(df_train[c], errors="coerce").to_numpy(np.float64)
        m = float(np.nanmean(x))
        sd = float(np.nanstd(x))
        if not np.isfinite(sd) or sd < eps:
            sd = 1.0
        means[c] = m
        stds[c] = sd
    return ScalerState(means=means, stds=stds, eps=eps)


def fit_target_standardizer(
    df_train: pd.DataFrame,
    *,
    ttc_floor: float,
    ttc_cap: float,
    eps: float = 1e-6,
) -> TargetStandardizerState:
    """Fit standardization for y_expert = log(TTC) on positive (expert_mask==1) frames."""

    if "y_gate" not in df_train.columns:
        raise ValueError("df_train must include y_gate before fitting target standardizer.")

    gate = pd.to_numeric(df_train["y_gate"], errors="coerce").fillna(0).to_numpy(np.int64)

    if "min_ttc_est" in df_train.columns:
        ttc_raw = pd.to_numeric(df_train["min_ttc_est"], errors="coerce").to_numpy(np.float64)
        mask = (gate == 1) & np.isfinite(ttc_raw) & (ttc_raw > 0)
        if mask.sum() < 10:
            mask = np.isfinite(ttc_raw) & (ttc_raw > 0)
        ttc = clip_ttc(ttc_raw[mask], ttc_floor, ttc_cap)
    else:
        y_soft = pd.to_numeric(df_train["y_soft"], errors="coerce").to_numpy(np.float64)
        mask = (gate == 1) & np.isfinite(y_soft) & (y_soft > 0)
        if mask.sum() < 10:
            mask = np.isfinite(y_soft) & (y_soft > 0)
        ttc = compute_ttc_from_y_soft(y_soft[mask], eps=eps)
        ttc = clip_ttc(ttc, ttc_floor, ttc_cap)

    y = log_ttc(ttc)
    mu = float(np.mean(y))
    sd = float(np.std(y))
    if not np.isfinite(sd) or sd < 1e-6:
        sd = 1.0
    return TargetStandardizerState(mu_y=mu, sigma_y=sd, eps=1e-6)


def _suggest_expert_context_drop(
    df_train: pd.DataFrame,
    schema: FeatureSchema,
    *,
    uniq_ratio_thr: float = 0.95,
    nan_ratio_thr: float = 0.5,
) -> Tuple[List[str], List[int]]:
    """Identify context (c__*) columns that act as a near-unique segment fingerprint.

    We only target *continuous* c__ columns (no '=' in the name) because:
      - one-hot ODD features are essential to the paper's claim
      - many per-segment continuous stats (map counts, speed stats, etc.)
        are effectively segment_id in disguise under segment-level split.

    Returns:
      (drop_cols, drop_idx) where drop_idx are indices into
      schema.x_expert_cols_in_order().
    """

    cols = schema.x_expert_cols_in_order()
    cand = [
        c
        for c in cols
        if c.startswith("c__")
        and ("=" not in c)
        and (not c.startswith("c__has_"))
        and (c in df_train.columns)
    ]
    if not cand:
        return [], []

    # One row per segment (c__ are constant per segment by construction)
    seg = df_train.groupby("segment_id", sort=False)[cand].first()
    nseg = len(seg)
    if nseg == 0:
        return [], []

    drop_cols: List[str] = []
    for c in cand:
        s = seg[c]
        nanr = float(s.isna().mean())
        uniq = int(s.nunique(dropna=True))
        uniq_ratio = uniq / max(1, nseg)
        if nanr > nan_ratio_thr or uniq_ratio > uniq_ratio_thr:
            drop_cols.append(c)

    drop_idx = [cols.index(c) for c in drop_cols]
    return drop_cols, drop_idx


def build_preprocess_state(
    df_train: pd.DataFrame,
    schema: FeatureSchema,
    *,
    ttc_floor: float = 0.05,
    ttc_cap: float = 10.0,
    eps: float = 1e-6,
) -> PreprocessState:
    """Fit variance-filter, scaler, and target standardizer on the *train split*.

    This function also produces a **pruned FeatureSchema** so that downstream
    Dataset/tensors always use the same kept column set.
    """

    if "segment_id" not in df_train.columns:
        raise ValueError("CSV must include segment_id")
    if "n_edges" not in df_train.columns:
        raise ValueError("Need n_edges in CSV to define Gate label")
    if ("y_soft" not in df_train.columns) and ("min_ttc_est" not in df_train.columns):
        raise ValueError("Need y_soft (TTCI) or min_ttc_est to build labels/targets")

    df_train = df_train.copy()

    # Count transforms must be applied BEFORE variance/scaler fitting.
    _apply_count_transforms_inplace(df_train)

    # Gate label
    df_train["y_gate"] = _compute_gate_label(df_train, ttc_floor=ttc_floor, ttc_cap=ttc_cap, eps=eps)

    # Variance filter on all candidate features (Gate + Expert extras)
    candidate_cols = schema.x_gate_cont + schema.x_gate_bin + schema.x_gate_onehot + schema.x_expert_extra_cont
    vf = fit_variance_filter(df_train, candidate_cols)
    kept = set(vf.kept_columns)

    # Prune schema to kept cols
    gate_onehot = [c for c in schema.x_gate_onehot if c in kept]
    gate_bin = [c for c in schema.x_gate_bin if c in kept]
    gate_cont = [c for c in schema.x_gate_cont if (c in kept and c not in gate_onehot and c not in gate_bin)]
    expert_extra = [c for c in schema.x_expert_extra_cont if c in kept]

    # Train-based binary detection (move from cont->bin for Gate only)
    gate_cont_final: List[str] = []
    gate_bin_final: List[str] = list(gate_bin)
    for c in gate_cont:
        if c in df_train.columns and _is_binary_series(df_train[c]):
            gate_bin_final.append(c)
        else:
            gate_cont_final.append(c)

    # Binary expert-extra columns should NOT be z-scored
    expert_bin = [c for c in expert_extra if (c in df_train.columns and _is_binary_series(df_train[c]))]

    onehot_cols = sorted(set(gate_onehot))
    binary_cols = sorted(set(gate_bin_final + expert_bin))
    continuous_cols = sorted(
        [c for c in (gate_cont_final + expert_extra) if (c not in onehot_cols and c not in binary_cols)]
    )

    scaler = fit_scaler(df_train, continuous_cols, eps=eps)
    tgt_std = fit_target_standardizer(df_train, ttc_floor=ttc_floor, ttc_cap=ttc_cap, eps=eps)

    new_schema = FeatureSchema(
        x_gate_cont=sorted(set(gate_cont_final)),
        x_gate_bin=sorted(set(gate_bin_final)),
        x_gate_onehot=sorted(set(gate_onehot)),
        x_expert_extra_cont=sorted(set(expert_extra)),
        id_cols=list(schema.id_cols),
        target_cols=list(schema.target_cols),
    )

    # Expert-only: auto-drop context continuous columns that effectively act as a
    # segment fingerprint ...
    drop_cols_expert, drop_idx_expert = _suggest_expert_context_drop(df_train, new_schema)

    return PreprocessState(
        schema=new_schema,
        var_filter=vf,
        scaler=scaler,
        target_std=tgt_std,
        continuous_cols=continuous_cols,
        binary_cols=binary_cols,
        onehot_cols=onehot_cols,
        expert_extra_cols=sorted(set(expert_extra)),
        drop_context_cols_expert=drop_cols_expert,
        drop_context_idx_expert=drop_idx_expert,
    )


def transform_dataframe(
    df: pd.DataFrame,
    state: PreprocessState,
    *,
    ttc_floor: float = 0.05,
    ttc_cap: float = 10.0,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Create model-ready numeric columns + targets.

    IMPORTANT
    ---------
    - Keeps **raw** feature values (no z-score scaling).
    - Adds/overwrites the following columns:
        y_gate (0/1)
        y_expert (standardized log TTC)
        expert_mask (0/1)
    """

    out = df.copy()

    # Consistent transforms
    _apply_count_transforms_inplace(out)

    # Gate label
    out["y_gate"] = _compute_gate_label(out, ttc_floor=ttc_floor, ttc_cap=ttc_cap, eps=eps).astype(np.float32)

    # TTC -> log TTC -> standardized target
    if "min_ttc_est" in out.columns:
        ttc_raw = pd.to_numeric(out["min_ttc_est"], errors="coerce").to_numpy(np.float64)
        ttc_raw = np.where(np.isfinite(ttc_raw) & (ttc_raw > 0), ttc_raw, np.inf)
    else:
        y_soft = pd.to_numeric(out["y_soft"], errors="coerce").to_numpy(np.float64)
        valid = np.isfinite(y_soft) & (y_soft > 0)
        ttc_raw = np.where(valid, compute_ttc_from_y_soft(y_soft, eps=eps), np.inf)

    ttc = clip_ttc(ttc_raw, ttc_floor, ttc_cap)
    y_log = log_ttc(ttc)
    out["y_expert"] = standardize_target(y_log, state.target_std).astype(np.float32)
    out["expert_mask"] = out["y_gate"].astype(np.float32)

    # Ensure all feature columns exist and are numeric
    feat_cols = state.schema.x_expert_cols_in_order()
    for c in feat_cols:
        if c not in out.columns:
            out[c] = 0.0
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out[feat_cols] = out[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Keep identifiers as-is
    if "segment_id" in out.columns:
        out["segment_id"] = out["segment_id"].astype(str)
    if "frame_label" in out.columns:
        out["frame_label"] = pd.to_numeric(out["frame_label"], errors="coerce").fillna(-1).astype(int)

    return out


def save_preprocess_state(state: PreprocessState, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(state.to_json())


def load_preprocess_state(path: str) -> PreprocessState:
    with open(path, "r", encoding="utf-8") as f:
        return PreprocessState.from_json(f.read())
