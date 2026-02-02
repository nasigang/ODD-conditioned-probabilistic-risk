from __future__ import annotations

"""Raw-space warp for Gate training.

Why this exists
---------------
The dataset has extreme positive scarcity. Gate recall becomes the bottleneck in any 2-stage
pipeline: if Gate blocks a candidate, Expert never sees it.

This warp augments *negative* gate samples by making them "more risky" in a physically-plausible way.

Key design choice (Strategy C)
-----------------------------
We warp **one direction only** per sample.

- Not "always front" — we choose the most risky direction among {front,rear,left,right,side,any}
  given the available features.
- This avoids creating unrealistic samples where *every* direction simultaneously becomes riskier.

Two guards are implemented:

A) Consistency warping
   - If ego speed is warped, derived relative-risk proxies (closing speeds) are warped consistently.
   - If ego speed isn't available in Gate input (minimal_v2 setup), we directly warp closing speeds.

B) Physics-gated labeling
   - Warped samples are **NOT** blindly set to y_gate=1.
   - We recompute y_gate from warped (range, closing speed) using the same thresholds as preprocess.
   - This produces useful hard negatives.

This file intentionally only touches Gate inputs/labels. Expert training is unchanged.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch


@dataclass
class RawWarpConfig:
    # probability to warp each negative sample
    p_warp: float = 0.35

    # how much to increase closing speed (m/s)
    closing_add_min: float = 0.5
    closing_add_max: float = 6.0

    # optionally scale ego speed if present
    speed_scale_min: float = 1.00
    speed_scale_max: float = 1.30

    # physical caps
    speed_cap_mps: float = 60.0
    closing_cap_mps: float = 80.0

    # gate-label thresholds (must match preprocess.GateLabelConfig)
    candidate_range_m: float = 50.0
    closing_thr_mps: float = 0.5
    ttc_max_s: float = 8.0


def _rand_uniform(shape, low, high, device):
    return (high - low) * torch.rand(shape, device=device) + low


def _get_col(x: torch.Tensor, feature_index: Dict[str, int], name: str, default: Optional[float] = None) -> torch.Tensor:
    if name not in feature_index:
        if default is None:
            raise KeyError(name)
        return torch.full((x.shape[0],), float(default), device=x.device, dtype=x.dtype)
    return x[:, feature_index[name]]


def _set_col(x: torch.Tensor, feature_index: Dict[str, int], name: str, value: torch.Tensor) -> None:
    if name not in feature_index:
        return
    x[:, feature_index[name]] = value


def _compute_gate_label_from_tensor(
    x: torch.Tensor,
    feature_index: Dict[str, int],
    *,
    candidate_range_m: float,
    closing_thr_mps: float,
    ttc_max_s: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Directional OR gate-label, using whatever columns exist.

    Notes
    -----
    In minimal feature setups, Gate input may not include per-direction ranges
    (e.g. x__min_range_front_m). In that case we fall back to x__min_range_any_m
    as an approximate range, so we can still do a directional OR based on
    (range_any, closing_dir).
    """

    def _dir(range_col: str, closing_col: str, *, fallback_range: str = "x__min_range_any_m") -> torch.Tensor:
        if closing_col not in feature_index:
            return torch.zeros((x.shape[0],), device=x.device, dtype=torch.bool)
        if range_col in feature_index:
            rng = x[:, feature_index[range_col]].clamp(min=0.0)
        elif fallback_range in feature_index:
            rng = x[:, feature_index[fallback_range]].clamp(min=0.0)
        else:
            return torch.zeros((x.shape[0],), device=x.device, dtype=torch.bool)

        # Positive closing speed == approaching. Do NOT flip negative values.
        clo = x[:, feature_index[closing_col]]
        clo = torch.clamp(clo, min=0.0)
        ttc = rng / (clo + eps)
        return (rng < candidate_range_m) & (clo > closing_thr_mps) & (ttc < ttc_max_s)

    cand = torch.zeros((x.shape[0],), device=x.device, dtype=torch.bool)

    # front/rear/left/right (if ranges missing, fallback to range_any)
    cand |= _dir("x__min_range_front_m", "x__max_closing_speed_front_mps")
    cand |= _dir("x__min_range_rear_m", "x__max_closing_speed_rear_mps")
    cand |= _dir("x__min_range_left_m", "x__max_closing_speed_left_mps")
    cand |= _dir("x__min_range_right_m", "x__max_closing_speed_right_mps")
    # side aggregate (if present)
    cand |= _dir("x__min_range_any_m", "x__max_closing_speed_side_mps", fallback_range="x__min_range_any_m")
    # any aggregate
    cand |= _dir("x__min_range_any_m", "x__max_closing_speed_any_mps", fallback_range="x__min_range_any_m")

    return cand.to(dtype=torch.float32)


def kinematic_warp_raw_x_gate(
    x_gate_raw: torch.Tensor,
    y_gate: torch.Tensor,
    feature_index: Dict[str, int],
    cfg: RawWarpConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Augment (x_gate_raw, y_gate) with warped negatives.

    Returns
    -------
    x_aug : [B + Bw, D]
    y_aug : [B + Bw]
    is_warp : bool mask [B + Bw] (True for warped rows)
    """

    device = x_gate_raw.device
    B, _ = x_gate_raw.shape

    neg_mask = (y_gate < 0.5)
    if int(neg_mask.sum().item()) == 0:
        return x_gate_raw, y_gate, torch.zeros(B, device=device, dtype=torch.bool)

    warp_mask = neg_mask & (torch.rand((B,), device=device) < float(cfg.p_warp))
    idx = torch.where(warp_mask)[0]
    if idx.numel() == 0:
        return x_gate_raw, y_gate, torch.zeros(B, device=device, dtype=torch.bool)

    xw = x_gate_raw[idx].clone()

    # -------------------------------------------------
    # 1) Warp ego speed (if present) and propagate delta
    # -------------------------------------------------
    delta_v = None
    if "x__ego_speed_mps" in feature_index and (cfg.speed_scale_max > 1.0):
        j = feature_index["x__ego_speed_mps"]
        old_v = xw[:, j]
        scale = _rand_uniform((idx.numel(),), cfg.speed_scale_min, cfg.speed_scale_max, device)
        new_v = torch.clamp(old_v * scale, 0.0, cfg.speed_cap_mps)
        xw[:, j] = new_v
        delta_v = (new_v - old_v).clamp(min=0.0)

    # -------------------------------------------------
    # 2) Warp closing speeds (directly or via delta_v)
    # -------------------------------------------------
    if delta_v is None:
        delta_v = _rand_uniform((idx.numel(),), cfg.closing_add_min, cfg.closing_add_max, device)

    # Strategy C: warp ONE "most risky" direction per sample (not a fixed direction).
    # We choose among available direction-specific closing speeds using a simple score.
    # If per-direction ranges don't exist in Gate input, we fall back to range_any.

    range_any = _get_col(xw, feature_index, "x__min_range_any_m", default=float("inf")).clamp(min=0.0)

    def _dir_score(closing_col: str, appr_col: Optional[str] = None, range_col: Optional[str] = None) -> torch.Tensor:
        if closing_col not in feature_index:
            return torch.full((xw.shape[0],), -float("inf"), device=device)
        clo = _get_col(xw, feature_index, closing_col, default=0.0)
        clo = torch.clamp(clo, min=0.0)
        # validity: prefer directions with observed approaching objects (appr_cnt > 0)
        if appr_col is not None and appr_col in feature_index:
            valid = _get_col(xw, feature_index, appr_col, default=0.0) > 0.0
        else:
            valid = clo > 0.0
        rng = range_any
        if range_col is not None and range_col in feature_index:
            rng = _get_col(xw, feature_index, range_col, default=float("inf")).clamp(min=0.0)
        # score ~ 1/TTC proxy (bigger is more risky)
        score = clo / (rng + 1e-3)
        score = torch.where(valid, score, torch.full_like(score, -float("inf")))
        return score

    # Candidate direction set (use whatever exists)
    dir_defs = [
        ("front", "x__max_closing_speed_front_mps", "x__appr_cnt_front", "x__min_range_front_m"),
        ("rear", "x__max_closing_speed_rear_mps", "x__appr_cnt_rear", "x__min_range_rear_m"),
        ("left", "x__max_closing_speed_left_mps", "x__appr_cnt_left", "x__min_range_left_m"),
        ("right", "x__max_closing_speed_right_mps", "x__appr_cnt_right", "x__min_range_right_m"),
        ("side", "x__max_closing_speed_side_mps", "x__appr_cnt_any", None),
    ]
    scores = []
    cols = []
    for _name, c_col, a_col, r_col in dir_defs:
        if c_col in feature_index:
            scores.append(_dir_score(c_col, a_col, r_col))
            cols.append(c_col)

    if len(cols) == 0:
        # No direction-specific closing speeds exist. Fallback: warp any aggregate if present.
        if "x__max_closing_speed_any_mps" in feature_index:
            j_any = feature_index["x__max_closing_speed_any_mps"]
            xw[:, j_any] = torch.clamp(xw[:, j_any] + delta_v, 0.0, cfg.closing_cap_mps)
    else:
        S = torch.stack(scores, dim=1)  # [Nw, K]
        best_k = torch.argmax(S, dim=1)  # [Nw]

        # Apply delta to ONLY the selected direction per sample.
        for k, c_col in enumerate(cols):
            j = feature_index[c_col]
            m = best_k == k
            if torch.any(m):
                xw[m, j] = xw[m, j] + delta_v[m]
                xw[m, j] = torch.clamp(xw[m, j], 0.0, cfg.closing_cap_mps)

        # Recompute side (if left/right exist) and any aggregate (max across directions)
        if "x__max_closing_speed_side_mps" in feature_index and (
            ("x__max_closing_speed_left_mps" in feature_index) and ("x__max_closing_speed_right_mps" in feature_index)
        ):
            jl = feature_index["x__max_closing_speed_left_mps"]
            jr = feature_index["x__max_closing_speed_right_mps"]
            js = feature_index["x__max_closing_speed_side_mps"]
            xw[:, js] = torch.maximum(xw[:, jl], xw[:, jr])

        # any = max of available direction closings (do not include any itself)
        if "x__max_closing_speed_any_mps" in feature_index:
            pool = []
            for c in [
                "x__max_closing_speed_front_mps",
                "x__max_closing_speed_rear_mps",
                "x__max_closing_speed_left_mps",
                "x__max_closing_speed_right_mps",
                "x__max_closing_speed_side_mps",
            ]:
                if c in feature_index:
                    pool.append(_get_col(xw, feature_index, c, default=0.0).clamp(min=0.0))
            if len(pool) > 0:
                new_any = torch.stack(pool, dim=1).max(dim=1).values
                j_any = feature_index["x__max_closing_speed_any_mps"]
                xw[:, j_any] = torch.clamp(new_any, 0.0, cfg.closing_cap_mps)

    # -------------------------------------------------
    # 3) Physics-gated labeling (hard negatives allowed)
    # -------------------------------------------------
    y_warp = _compute_gate_label_from_tensor(
        xw,
        feature_index,
        candidate_range_m=float(cfg.candidate_range_m),
        closing_thr_mps=float(cfg.closing_thr_mps),
        ttc_max_s=float(cfg.ttc_max_s),
    )

    x_aug = torch.cat([x_gate_raw, xw], dim=0)
    y_aug = torch.cat([y_gate, y_warp], dim=0)
    is_warp = torch.cat(
        [
            torch.zeros(B, device=device, dtype=torch.bool),
            torch.ones(idx.numel(), device=device, dtype=torch.bool),
        ],
        dim=0,
    )
    return x_aug, y_aug, is_warp
