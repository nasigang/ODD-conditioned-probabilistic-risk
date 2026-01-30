from __future__ import annotations

"""Raw-space warp for Gate training.

Why this exists
---------------
The dataset has extreme positive scarcity. Gate recall becomes the bottleneck in any 2-stage
pipeline: if Gate blocks a candidate, Expert never sees it.

This warp augments *negative* gate samples by making them "more risky" in a physically-plausible way.
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
    """Directional OR gate-label, using whatever columns exist."""

    def _dir(range_col: str, closing_col: str) -> torch.Tensor:
        if range_col not in feature_index or closing_col not in feature_index:
            return torch.zeros((x.shape[0],), device=x.device, dtype=torch.bool)
        rng = x[:, feature_index[range_col]].clamp(min=0.0)
        clo = x[:, feature_index[closing_col]].clamp(min=0.0)
        ttc = rng / (clo + eps)
        return (rng < candidate_range_m) & (clo > closing_thr_mps) & (ttc < ttc_max_s)

    cand = torch.zeros((x.shape[0],), device=x.device, dtype=torch.bool)

    # front/rear/left/right
    cand |= _dir("x__min_range_front_m", "x__max_closing_speed_front_mps")
    cand |= _dir("x__min_range_rear_m", "x__max_closing_speed_rear_mps")
    cand |= _dir("x__min_range_left_m", "x__max_closing_speed_left_mps")
    cand |= _dir("x__min_range_right_m", "x__max_closing_speed_right_mps")

    # any aggregate
    cand |= _dir("x__min_range_any_m", "x__max_closing_speed_any_mps")

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

    closing_cols = [
        "x__max_closing_speed_any_mps",
        "x__max_closing_speed_front_mps",
        "x__max_closing_speed_rear_mps",
        "x__max_closing_speed_left_mps",
        "x__max_closing_speed_right_mps",
        "x__max_closing_speed_side_mps",
    ]

    for c in closing_cols:
        if c in feature_index:
            j = feature_index[c]
            # increase only if already non-negative
            xw[:, j] = torch.clamp(xw[:, j] + delta_v, 0.0, cfg.closing_cap_mps)

    # If side speed is derived and not explicitly present, we can recompute it from left/right.
    if "x__max_closing_speed_side_mps" in feature_index:
        if ("x__max_closing_speed_left_mps" in feature_index) and ("x__max_closing_speed_right_mps" in feature_index):
            jl = feature_index["x__max_closing_speed_left_mps"]
            jr = feature_index["x__max_closing_speed_right_mps"]
            js = feature_index["x__max_closing_speed_side_mps"]
            xw[:, js] = torch.maximum(xw[:, jl], xw[:, jr])

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
