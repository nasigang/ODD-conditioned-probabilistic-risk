from __future__ import annotations

"""Preprocessing utilities.

This module does **three** things:
1) Builds a *PreprocessState* from TRAIN split only:
   - FeatureSchema (which columns are inputs)
   - Variance filter (drop near-constant columns)
   - Per-feature scaler (z-score for continuous features only)
   - Target standardization for y_expert (log-TTC)
   - Gate-label config used consistently in train/eval

2) Transforms any dataframe (train/val/test) into model-ready columns:
   - derived columns (e.g., max_closing_speed_side)
   - y_gate (Stage-1 candidate label)
   - y_expert (standardized log-TTC target)
   - expert_mask (which rows are allowed to train/eval Expert)

3) Provides small, explicit guards against leakage / fingerprinting:
   - never use identifiers/targets as inputs
   - optional auto-suggested drop list for per-segment constant c__ columns (legacy)

Important
---------
"y_gate" is NOT the final event label. It is only the Stage-1 *candidate* label.
The final event metric labels (e.g., TTC <= s*(v)) are computed in eval scripts.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import json

from .schema import FeatureSchema


# -------------------------------
# Small stats helpers
# -------------------------------


def _is_binary_series(s: pd.Series) -> bool:
    v = pd.to_numeric(s, errors="coerce").dropna().unique()
    if len(v) == 0:
        return True
    if len(v) > 3:
        return False
    vset = set(float(x) for x in v)
    return vset.issubset({0.0, 1.0})


def _safe_to_numpy(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full((len(df),), default, dtype=np.float64)
    a = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)
    a = np.where(np.isfinite(a), a, default)
    return a


# -------------------------------
# Variance filter + scaler
# -------------------------------


@dataclass
class VarianceFilterState:
    kept_columns: List[str]


def fit_variance_filter(df_train: pd.DataFrame, cols: Sequence[str], *, min_var: float = 1e-12) -> VarianceFilterState:
    kept: List[str] = []
    for c in cols:
        if c not in df_train.columns:
            continue
        x = pd.to_numeric(df_train[c], errors="coerce").to_numpy(np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        if float(np.nanvar(x)) <= float(min_var):
            continue
        kept.append(c)
    return VarianceFilterState(kept_columns=kept)


@dataclass
class ScalerState:
    means: Dict[str, float]
    stds: Dict[str, float]


def fit_scaler(df_train: pd.DataFrame, cols: Sequence[str], *, eps: float = 1e-6) -> ScalerState:
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for c in cols:
        if c not in df_train.columns:
            continue
        x = pd.to_numeric(df_train[c], errors="coerce").to_numpy(np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            means[c] = 0.0
            stds[c] = 1.0
            continue
        m = float(np.mean(x))
        s = float(np.std(x))
        means[c] = m
        stds[c] = max(s, eps)
    return ScalerState(means=means, stds=stds)


@dataclass
@dataclass
class TargetStandardizer:
    mean: float
    std: float

    # Backward-compatible aliases
    @property
    def mu_y(self) -> float:
        return float(self.mean)

    @property
    def sigma_y(self) -> float:
        return float(self.std)



def fit_target_standardizer(df_train: pd.DataFrame, *, ttc_floor: float, ttc_cap: float, eps: float = 1e-6) -> TargetStandardizer:
    """Fit standardizer for log(TTC).

    Supported training targets (in priority order):
    1) min_ttc_est: TTC in seconds (preferred if provided)
    2) y_soft: inverse-TTC proxy (~ 1/TTC). TTC = 1 / (y_soft + eps)
    3) fallback: derive min_ttc_est from (range, closing_speed) features, taking min over directions.
    """
    _ensure_derived_columns(df_train)

    # If neither explicit TTC nor inverse proxy exists, derive TTC estimate from interaction features.
    if ("min_ttc_est" not in df_train.columns) and ("y_soft" not in df_train.columns):
        _maybe_derive_min_ttc_est(df_train, eps=eps)

    if "min_ttc_est" in df_train.columns:
        ttc = pd.to_numeric(df_train["min_ttc_est"], errors="coerce").to_numpy(np.float64)
        ttc = np.where(np.isfinite(ttc) & (ttc > 0.0), ttc, np.inf)
    elif "y_soft" in df_train.columns:
        y_soft = pd.to_numeric(df_train["y_soft"], errors="coerce").to_numpy(np.float64)
        valid = np.isfinite(y_soft) & (y_soft > 0.0)
        ttc = np.where(valid, 1.0 / (y_soft + eps), np.inf)
    else:
        raise ValueError("Need either min_ttc_est (TTC) or y_soft (1/TTC) to fit target standardizer.")

    ttc = np.clip(ttc, ttc_floor, ttc_cap)
    y = np.log(ttc + eps)
    y = y[np.isfinite(y)]
    m = float(np.mean(y)) if y.size else 0.0
    s = float(np.std(y)) if y.size else 1.0
    return TargetStandardizer(mean=m, std=max(s, eps))
    ttc = np.where(np.isfinite(ttc) & (ttc > 0), ttc, np.inf)
    ttc = np.clip(ttc, ttc_floor, ttc_cap)
    y = np.log(ttc + eps)
    m = float(np.mean(y[np.isfinite(y)]))
    s = float(np.std(y[np.isfinite(y)]))
    return TargetStandardizer(mean=m, std=max(s, eps))


# -------------------------------
# Gate-label config + compute
# -------------------------------


@dataclass
class GateLabelConfig:
    candidate_range_m: float = 50.0
    closing_thr_mps: float = 0.5
    ttc_max_s: float = 8.0
    # if True: use per-direction OR (front/rear/side/any). Recommended.
    use_directional_or: bool = True


def _ensure_derived_columns(df: pd.DataFrame) -> None:
    """Create derived columns in-place (safe if already exists)."""
    # tolerate minor prefix typos from export (x_appr_cnt_* -> x__appr_cnt_*)
    for base in ["any","front","left","right","rear"]:
        a=f"x_appr_cnt_{base}"
        b=f"x__appr_cnt_{base}"
        if (b not in df.columns) and (a in df.columns):
            df[b] = pd.to_numeric(df[a], errors="coerce").fillna(0.0).astype(np.float32)
    # side closing speed
    if "x__max_closing_speed_side_mps" not in df.columns:
        if ("x__max_closing_speed_left_mps" in df.columns) or ("x__max_closing_speed_right_mps" in df.columns):
            left = _safe_to_numpy(df, "x__max_closing_speed_left_mps", default=0.0)
            right = _safe_to_numpy(df, "x__max_closing_speed_right_mps", default=0.0)
            df["x__max_closing_speed_side_mps"] = np.maximum(left, right).astype(np.float32)


def _compute_gate_label_v2(df: pd.DataFrame, cfg: GateLabelConfig, *, eps: float = 1e-6) -> np.ndarray:
    """Stage-1 candidate label (NOT the final event label).

    Core idea
    ---------
    Gate should trigger when there exists at least one *plausible* interaction candidate
    within range and with positive closing speed.

    We avoid `segment_id/frame_label` and also avoid using `y_soft/label/x__best_ttci`.

    If required columns are missing, we fall back to weaker proxies.
    """
    n = len(df)

    # directional OR (preferred): detects front/rear/side candidates
    if cfg.use_directional_or:
        have_front = ("x__min_range_front_m" in df.columns) and ("x__max_closing_speed_front_mps" in df.columns)
        have_rear = ("x__min_range_rear_m" in df.columns) and ("x__max_closing_speed_rear_mps" in df.columns)
        have_left = ("x__min_range_left_m" in df.columns) and ("x__max_closing_speed_left_mps" in df.columns)
        have_right = ("x__min_range_right_m" in df.columns) and ("x__max_closing_speed_right_mps" in df.columns)

        if have_front or have_rear or have_left or have_right:
            cand = np.zeros((n,), dtype=bool)

            def _dir(range_col: str, closing_col: str) -> np.ndarray:
                rng = _safe_to_numpy(df, range_col, default=np.inf)
                clo = _safe_to_numpy(df, closing_col, default=0.0)
                # require positive closing (approaching)
                clo_pos = np.maximum(clo, 0.0)
                ttc = rng / (clo_pos + eps)
                return (rng < cfg.candidate_range_m) & (clo_pos > cfg.closing_thr_mps) & (ttc < cfg.ttc_max_s)

            if have_front:
                cand |= _dir("x__min_range_front_m", "x__max_closing_speed_front_mps")
            if have_rear:
                cand |= _dir("x__min_range_rear_m", "x__max_closing_speed_rear_mps")
            if have_left:
                cand |= _dir("x__min_range_left_m", "x__max_closing_speed_left_mps")
            if have_right:
                cand |= _dir("x__min_range_right_m", "x__max_closing_speed_right_mps")

            # also consider the aggregated any if present
            if ("x__min_range_any_m" in df.columns) and ("x__max_closing_speed_any_mps" in df.columns):
                cand |= _dir("x__min_range_any_m", "x__max_closing_speed_any_mps")

            return cand.astype(np.float32)

    # fallback: any aggregated
    if ("x__min_range_any_m" in df.columns) and ("x__max_closing_speed_any_mps" in df.columns):
        rng = _safe_to_numpy(df, "x__min_range_any_m", default=np.inf)
        clo = _safe_to_numpy(df, "x__max_closing_speed_any_mps", default=0.0)
        clo_pos = np.maximum(clo, 0.0)
        ttc = rng / (clo_pos + eps)
        cand = (rng < cfg.candidate_range_m) & (clo_pos > cfg.closing_thr_mps) & (ttc < cfg.ttc_max_s)
        return cand.astype(np.float32)

    # fallback: dyn label count (weak proxy). NOTE: easy to over-trigger in dense traffic.
    if "x__dyn_label_count_30m" in df.columns:
        cnt = _safe_to_numpy(df, "x__dyn_label_count_30m", default=0.0)
        return (cnt > 0).astype(np.float32)

    # legacy fallback
    if "n_edges" in df.columns:
        ne = _safe_to_numpy(df, "n_edges", default=0.0)
        return (ne > 0).astype(np.float32)

    return np.zeros((n,), dtype=np.float32)


# -------------------------------
# Expert context-drop suggestion (legacy)
# -------------------------------


def _suggest_expert_context_drop(df_train: pd.DataFrame, schema: FeatureSchema) -> Tuple[List[str], List[int]]:
    """Suggest c__ continuous columns that look like segment fingerprints.

    In minimal_v2, Expert should *not* see segment-level c__ stats at all, so this
    typically returns empty.
    """
    if "segment_id" not in df_train.columns:
        return [], []

    expert_cols = schema.x_expert_cols_in_order()
    bad_cols: List[str] = []

    for c in expert_cols:
        if not c.startswith("c__"):
            continue
        if "=" in c or c.startswith("c__has_"):
            # one-hot and simple map presence flags are less fingerprinty
            continue
        # if almost constant within segment, it's essentially a segment-id proxy
        g = df_train.groupby("segment_id")[c].nunique(dropna=False)
        if float(np.mean(g.to_numpy(np.float64) <= 1.0)) > 0.95:
            bad_cols.append(c)

    idx = [expert_cols.index(c) for c in bad_cols if c in expert_cols]
    return bad_cols, idx


# -------------------------------
# Count transforms (optional)
# -------------------------------


def _apply_count_transforms_inplace(df: pd.DataFrame) -> None:
    """Apply mild transforms to heavy-tailed count features.

    Current minimal_v2 uses approach counts and dyn_label_count. log1p helps.
    """
    for c in [
        "x__appr_cnt_any",
        "x__appr_cnt_front",
        "x__appr_cnt_left",
        "x__appr_cnt_right",
        "x__appr_cnt_rear",
        "x__dyn_label_count_30m",
    ]:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)
            x = np.where(np.isfinite(x) & (x > 0), x, 0.0)
            df[c] = np.log1p(x).astype(np.float32)


# -------------------------------
# Preprocess state
# -------------------------------


@dataclass
class PreprocessState:
    schema: FeatureSchema
    var_filter: VarianceFilterState
    scaler: ScalerState
    target_std: TargetStandardizer

    # Column groups (post variance-filter)
    continuous_cols: List[str]
    binary_cols: List[str]
    onehot_cols: List[str]

    # Optional: suggested expert context drop (legacy)
    drop_context_cols_expert: List[str] = field(default_factory=list)
    drop_context_idx_expert: List[int] = field(default_factory=list)

    # Cached flow split indices
    flow_x_cols: List[str] = field(default_factory=list)
    flow_c_cols: List[str] = field(default_factory=list)
    flow_x_idx: List[int] = field(default_factory=list)
    flow_c_idx: List[int] = field(default_factory=list)

    # Gate-label config
    gate_label_cfg: GateLabelConfig = field(default_factory=GateLabelConfig)

    def to_json(self) -> str:
        d = asdict(self)
        # dataclasses -> dict for GateLabelConfig
        d["gate_label_cfg"] = asdict(self.gate_label_cfg)
        return json.dumps(d, indent=2)

    @staticmethod
    def from_json(s: str) -> "PreprocessState":
        import json

        d = json.loads(s)
        if not isinstance(d, dict):
            raise TypeError("PreprocessState JSON must decode to dict")

        schema = FeatureSchema.from_json(json.dumps(d["schema"]))
        vf = VarianceFilterState(**d["var_filter"])
        scaler = ScalerState(**d["scaler"])
        tgt = TargetStandardizer(**d["target_std"])
        gate_cfg = GateLabelConfig(**d.get("gate_label_cfg", {}))

        return PreprocessState(
            schema=schema,
            var_filter=vf,
            scaler=scaler,
            target_std=tgt,
            continuous_cols=d.get("continuous_cols", []),
            binary_cols=d.get("binary_cols", []),
            onehot_cols=d.get("onehot_cols", []),
            drop_context_cols_expert=d.get("drop_context_cols_expert", []),
            drop_context_idx_expert=d.get("drop_context_idx_expert", []),
            flow_x_cols=d.get("flow_x_cols", []),
            flow_c_cols=d.get("flow_c_cols", []),
            flow_x_idx=d.get("flow_x_idx", []),
            flow_c_idx=d.get("flow_c_idx", []),
            gate_label_cfg=gate_cfg,
        )


# -------------------------------
# Build state from TRAIN split
# -------------------------------


def build_preprocess_state(
    df_train: pd.DataFrame,
    schema: FeatureSchema,
    *,
    ttc_floor: float = 0.05,
    ttc_cap: float = 10.0,
    eps: float = 1e-6,
    gate_label_cfg: Optional[GateLabelConfig] = None,
) -> PreprocessState:
    """Fit preprocess state using TRAIN split only."""
    df_train = df_train.copy()

    _ensure_derived_columns(df_train)
    _apply_count_transforms_inplace(df_train)

    gate_label_cfg = gate_label_cfg or GateLabelConfig()

    # Candidate columns for variance filter: all inputs (gate + expert)
    cand = schema.x_gate_cols_in_order() + schema.x_expert_cols_in_order()
    cand = [c for c in cand if c in df_train.columns]

    vf = fit_variance_filter(df_train, cand)
    kept = set(vf.kept_columns)

    # Gate groups
    gate_onehot = [c for c in schema.x_gate_onehot if c in kept]
    gate_bin = [c for c in schema.x_gate_bin if c in kept]
    gate_cont = [c for c in schema.x_gate_cont if c in kept and c not in gate_onehot and c not in gate_bin]

    # (optional) promote binary-looking cont columns to bin
    gate_cont_final: List[str] = []
    gate_bin_final: List[str] = list(gate_bin)
    for c in gate_cont:
        if _is_binary_series(df_train[c]):
            gate_bin_final.append(c)
        else:
            gate_cont_final.append(c)

    # Expert groups
    expert_cols = [c for c in schema.x_expert_cols_in_order() if c in kept]
    expert_bin = [c for c in expert_cols if (c in df_train.columns and _is_binary_series(df_train[c]))]

    onehot_cols = sorted(set(gate_onehot))
    binary_cols = sorted(set(gate_bin_final + expert_bin))

    # Continuous = everything else among (gate_cont + expert_cols)
    continuous_cols = sorted(
        [c for c in (gate_cont_final + expert_cols) if (c not in onehot_cols and c not in binary_cols)]
    )

    scaler = fit_scaler(df_train, continuous_cols, eps=eps)
    tgt_std = fit_target_standardizer(df_train, ttc_floor=ttc_floor, ttc_cap=ttc_cap, eps=eps)

    # Build a new schema reflecting variance-filtered columns
    new_schema = FeatureSchema(
        x_gate_cont=sorted(set(gate_cont_final)),
        x_gate_bin=sorted(set(gate_bin_final)),
        x_gate_onehot=sorted(set(gate_onehot)),
        x_expert_x_cont=list(schema.x_expert_x_cont) if schema.x_expert_x_cont is not None else None,
        x_expert_c_cont=list(schema.x_expert_c_cont) if schema.x_expert_c_cont is not None else None,
        x_expert_extra_cont=list(schema.x_expert_extra_cont or []),
        id_cols=list(schema.id_cols or []),
        target_cols=list(schema.target_cols or []),
        flow_x_cols=list(schema.flow_x_cols or []),
        flow_c_cols=list(schema.flow_c_cols or []),
    )

    # Suggested expert context drop (legacy)
    drop_cols_expert, drop_idx_expert = _suggest_expert_context_drop(df_train, new_schema)

    # Flow split cache (only meaningful for FiLM, but safe to store)
    expert_cols_order = new_schema.x_expert_cols_in_order()
    flow_x_cols = new_schema.flow_x_cols_in_order()
    flow_c_cols = new_schema.flow_c_cols_in_order()
    flow_x_idx = [expert_cols_order.index(c) for c in flow_x_cols if c in expert_cols_order]
    flow_c_idx = [expert_cols_order.index(c) for c in flow_c_cols if c in expert_cols_order]

    return PreprocessState(
        schema=new_schema,
        var_filter=vf,
        scaler=scaler,
        target_std=tgt_std,
        continuous_cols=continuous_cols,
        binary_cols=binary_cols,
        onehot_cols=onehot_cols,
        drop_context_cols_expert=drop_cols_expert,
        drop_context_idx_expert=drop_idx_expert,
        flow_x_cols=flow_x_cols,
        flow_c_cols=flow_c_cols,
        flow_x_idx=flow_x_idx,
        flow_c_idx=flow_c_idx,
        gate_label_cfg=gate_label_cfg,
    )


# -------------------------------
# Transform dataframe
# -------------------------------


def transform_dataframe(
    df: pd.DataFrame,
    state: PreprocessState,
    *,
    ttc_floor: float = 0.05,
    ttc_cap: float = 10.0,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Apply transforms using a fitted PreprocessState."""
    out = df.copy()

    _ensure_derived_columns(out)
    _apply_count_transforms_inplace(out)

    # y_gate: candidate label used for Stage-1 gate + expert mask
    out["y_gate"] = _compute_gate_label_v2(out, state.gate_label_cfg, eps=eps).astype(np.float32)

    # y_expert target: standardized log(TTC)
    # ------------------------------
    # Expert target (y_expert): standardized log(TTC)
    # Priority: min_ttc_est (TTC sec) > y_soft (1/TTC) > fallback derived TTC
    # ------------------------------
    if ("min_ttc_est" not in out.columns) and ("y_soft" not in out.columns):
        _maybe_derive_min_ttc_est(out, eps=eps)
    if "min_ttc_est" in out.columns:
        ttc_raw = pd.to_numeric(out["min_ttc_est"], errors="coerce").to_numpy(np.float64)
        ttc_raw = np.where(np.isfinite(ttc_raw) & (ttc_raw > 0.0), ttc_raw, np.inf)
    elif "y_soft" in out.columns:
        y_soft = pd.to_numeric(out["y_soft"], errors="coerce").to_numpy(np.float64)
        valid = np.isfinite(y_soft) & (y_soft > 0.0)
        ttc_raw = np.where(valid, 1.0 / (y_soft + eps), np.inf)
    else:
        raise ValueError("Need y_soft or min_ttc_est to compute y_expert")
    ttc = np.clip(ttc_raw, ttc_floor, ttc_cap)
    y = np.log(ttc + eps)
    y_std = (y - state.target_std.mean) / (state.target_std.std + eps)
    out["y_expert"] = y_std.astype(np.float32)

    # Expert mask: only train/eval expert where gate candidate is true
    out["expert_mask"] = out["y_gate"].astype(np.float32)

    # Ensure schema columns exist (missing -> 0). This is critical for stable ordering.
    need = set(state.schema.x_gate_cols_in_order() + state.schema.x_expert_cols_in_order())
    for c in need:
        if c not in out.columns:
            out[c] = 0.0

    # Apply variance filter: drop columns that were filtered out (set to 0)
    kept = set(state.var_filter.kept_columns)
    for c in need:
        if c not in kept:
            out[c] = 0.0

    return out



# -------------------------------
# I/O helpers
# -------------------------------

def save_preprocess_state(state: PreprocessState, path: str) -> None:
    """Save PreprocessState as JSON to `path`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(state.to_json())

def load_preprocess_state(path: str) -> PreprocessState:
    """Load PreprocessState from JSON at `path`."""
    p = Path(path)
    s = p.read_text()
    return PreprocessState.from_json(s)