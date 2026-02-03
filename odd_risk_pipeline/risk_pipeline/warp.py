from __future__ import annotations

import math

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


def _safe_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=1e-12))


def _get_optional_idx(feature_index: Dict[str, int], name: str) -> Optional[int]:
    return feature_index.get(name, None)


def _recompute_any_and_side_closing(
    xw: torch.Tensor,
    feature_index: Dict[str, int],
) -> None:
    """Recompute any/side closing speed aggregates if present."""
    i_f = _get_optional_idx(feature_index, "x__max_closing_speed_front_mps")
    i_r = _get_optional_idx(feature_index, "x__max_closing_speed_rear_mps")
    i_l = _get_optional_idx(feature_index, "x__max_closing_speed_left_mps")
    i_rt = _get_optional_idx(feature_index, "x__max_closing_speed_right_mps")
    i_side = _get_optional_idx(feature_index, "x__max_closing_speed_side_mps")
    i_any = _get_optional_idx(feature_index, "x__max_closing_speed_any_mps")

    vals = []
    for i in (i_f, i_r, i_l, i_rt):
        if i is not None:
            vals.append(xw[:, i])
    if len(vals) == 0:
        return
    stack = torch.stack(vals, dim=0)  # [K,B]

    if i_side is not None and (i_l is not None or i_rt is not None):
        if i_l is None:
            side = xw[:, i_rt]
        elif i_rt is None:
            side = xw[:, i_l]
        else:
            side = torch.maximum(xw[:, i_l], xw[:, i_rt])
        xw[:, i_side] = side

    if i_any is not None:
        xw[:, i_any] = torch.amax(stack, dim=0)


def _ttc_proxy_from_ranges_and_closing(
    xw: torch.Tensor,
    feature_index: Dict[str, int],
    closing_thr_mps: float,
    ttc_cap_s: float,
) -> torch.Tensor:
    """Conservative TTC proxy from directional range/closing. min over valid dirs."""
    eps = 1e-3
    r_cols = {
        "front": _get_optional_idx(feature_index, "x__min_range_front_m"),
        "rear": _get_optional_idx(feature_index, "x__min_range_rear_m"),
        "left": _get_optional_idx(feature_index, "x__min_range_left_m"),
        "right": _get_optional_idx(feature_index, "x__min_range_right_m"),
    }
    c_cols = {
        "front": _get_optional_idx(feature_index, "x__max_closing_speed_front_mps"),
        "rear": _get_optional_idx(feature_index, "x__max_closing_speed_rear_mps"),
        "left": _get_optional_idx(feature_index, "x__max_closing_speed_left_mps"),
        "right": _get_optional_idx(feature_index, "x__max_closing_speed_right_mps"),
    }

    ttcs = []
    valids = []
    for k in ("front", "rear", "left", "right"):
        ir, ic = r_cols[k], c_cols[k]
        if ir is None or ic is None:
            continue
        closing = xw[:, ic]
        rng = torch.clamp(xw[:, ir], min=0.0)
        valid = closing > float(closing_thr_mps)
        ttc = rng / torch.clamp(closing, min=eps)
        ttcs.append(ttc)
        valids.append(valid)

    if len(ttcs) == 0:
        return torch.full((xw.shape[0],), float(ttc_cap_s), device=xw.device, dtype=xw.dtype)

    ttc_stack = torch.stack(ttcs, dim=0)      # [K,B]
    valid_stack = torch.stack(valids, dim=0)  # [K,B]

    inf = torch.full_like(ttc_stack, float("inf"))
    ttc_masked = torch.where(valid_stack, ttc_stack, inf)
    ttc_min = torch.amin(ttc_masked, dim=0)
    ttc_min = torch.where(torch.isfinite(ttc_min), ttc_min, torch.full_like(ttc_min, float(ttc_cap_s)))
    return torch.clamp(ttc_min, min=0.0, max=float(ttc_cap_s))


def kinematic_warp_raw_x_expert(
    x_expert_raw: torch.Tensor,
    y_expert: torch.Tensor,
    censored_mask: torch.Tensor,
    feature_index: Dict[str, int],
    cfg: RawWarpConfig,
    *,
    ttc_floor_s: float,
    ttc_cap_s: float,
    target_mu_y: float,
    target_sigma_y: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append label-consistent warped samples into Expert batch.

    핵심(첨부 PDF 요구사항 반영)
    ---------------------------
    1) Expert warp는 feature만 바꾸면 끝이 아니라, **y_expert(=표준화 logTTC)** 를 반드시 재생성해야 한다.
    2) Warp는 {front, rear, side(max(left,right))} 중 **가장 위험한(=양수 closing이 가장 큰) 한 방향만** 적용한다.
    3) Warp 후에는 any/side aggregate closing을 재정합(recompute)한다.
    4) y_expert 재생성은 선택된 동일 방향의 (range, closing)으로
         TTC' = range_dir / (max(closing_dir, 0) + eps)
       를 계산한 뒤 log/clip/standardize 한다.
    5) TTC'가 ttc_cap에 걸리면 strict right-censoring으로 처리한다.

    Returns
    -------
    x_aug : [B + Bw, D]
    y_aug : [B + Bw]
    cens_aug : [B + Bw]
    is_warp : bool mask [B + Bw] (True for appended warped rows)
    """
    device = x_expert_raw.device
    B, _ = x_expert_raw.shape

    if B == 0 or float(cfg.p_warp) <= 0.0:
        is_warp = torch.zeros((B,), device=device, dtype=torch.bool)
        return x_expert_raw, y_expert, censored_mask, is_warp

    # Candidate filter: avoid warping clearly irrelevant "no-approach" rows
    i_rng_any = _get_optional_idx(feature_index, "x__min_range_any_m")
    i_cls_any = _get_optional_idx(feature_index, "x__max_closing_speed_any_mps")
    if i_rng_any is not None and i_cls_any is not None:
        cand = (x_expert_raw[:, i_rng_any] < float(cfg.candidate_range_m)) & (x_expert_raw[:, i_cls_any] > float(cfg.closing_thr_mps))
    else:
        cand = torch.ones((B,), device=device, dtype=torch.bool)

    warp_mask = cand & (torch.rand((B,), device=device) < float(cfg.p_warp))
    idx = torch.where(warp_mask)[0]
    if idx.numel() == 0:
        is_warp = torch.zeros((B,), device=device, dtype=torch.bool)
        return x_expert_raw, y_expert, censored_mask, is_warp

    # Work only on warped rows
    xw = x_expert_raw[idx].clone()  # [Bw, D]
    Bw = xw.shape[0]

    # Column indices (optional)
    i_f = _get_optional_idx(feature_index, "x__max_closing_speed_front_mps")
    i_r = _get_optional_idx(feature_index, "x__max_closing_speed_rear_mps")
    i_l = _get_optional_idx(feature_index, "x__max_closing_speed_left_mps")
    i_rt = _get_optional_idx(feature_index, "x__max_closing_speed_right_mps")
    i_speed = _get_optional_idx(feature_index, "x__ego_speed_mps")

    i_rf = _get_optional_idx(feature_index, "x__min_range_front_m")
    i_rr = _get_optional_idx(feature_index, "x__min_range_rear_m")
    i_rl = _get_optional_idx(feature_index, "x__min_range_left_m")
    i_rrt = _get_optional_idx(feature_index, "x__min_range_right_m")

    def _g(i):
        if i is None:
            return torch.zeros((Bw,), device=device, dtype=xw.dtype)
        return torch.clamp(xw[:, i], min=0.0)

    c_front = _g(i_f)
    c_rear = _g(i_r)
    c_left = _g(i_l)
    c_right = _g(i_rt)
    side_is_left = (c_left >= c_right)
    c_side = torch.maximum(c_left, c_right)

    # dominant direction among {front, rear, side}
    stack = torch.stack([c_front, c_rear, c_side], dim=0)  # [3,Bw]
    dom = torch.argmax(stack, dim=0)  # 0 front, 1 rear, 2 side

    # Sample warp magnitude per warped row
    dv = _rand_uniform((Bw,), float(cfg.closing_add_min), float(cfg.closing_add_max), device=device)

    # --- Apply one-direction warp (closing + optional speed for front only) ---
    if i_f is not None:
        m = (dom == 0)
        if m.any():
            xw[m, i_f] = torch.clamp(xw[m, i_f] + dv[m], min=0.0, max=float(cfg.closing_cap_mps))
            if i_speed is not None and float(cfg.speed_scale_max) > 1.0:
                s = _rand_uniform((int(m.sum().item()),), float(cfg.speed_scale_min), float(cfg.speed_scale_max), device=device)
                xw[m, i_speed] = torch.clamp(xw[m, i_speed] * s, min=0.0, max=float(cfg.speed_cap_mps))

    if i_r is not None:
        m = (dom == 1)
        if m.any():
            # rear: do NOT scale ego speed (directional sign ambiguity)
            xw[m, i_r] = torch.clamp(xw[m, i_r] + dv[m], min=0.0, max=float(cfg.closing_cap_mps))

    # side: boost the currently larger side direction only
    if (i_l is not None) or (i_rt is not None):
        m_side = (dom == 2)
        if m_side.any():
            if i_l is not None:
                mL = m_side & side_is_left
                if mL.any():
                    xw[mL, i_l] = torch.clamp(xw[mL, i_l] + dv[mL], min=0.0, max=float(cfg.closing_cap_mps))
            if i_rt is not None:
                mR = m_side & (~side_is_left)
                if mR.any():
                    xw[mR, i_rt] = torch.clamp(xw[mR, i_rt] + dv[mR], min=0.0, max=float(cfg.closing_cap_mps))

    # Guard: never allow negative closing after warp
    for i in (i_f, i_r, i_l, i_rt):
        if i is not None:
            xw[:, i] = torch.clamp(xw[:, i], min=0.0, max=float(cfg.closing_cap_mps))

    # Recompute aggregate closing speeds if present (any/side)
    _recompute_any_and_side_closing(xw, feature_index)

    # --- Label recompute: TTC' from the SAME chosen direction ---
    eps = 1e-3
    ttc = torch.full((Bw,), float(ttc_cap_s), device=device, dtype=xw.dtype)

    # front TTC
    if i_rf is not None and i_f is not None:
        m = (dom == 0)
        if m.any():
            rng = torch.clamp(xw[m, i_rf], min=0.0)
            clo = torch.clamp(xw[m, i_f], min=0.0)
            ttc[m] = rng / (clo + eps)

    # rear TTC
    if i_rr is not None and i_r is not None:
        m = (dom == 1)
        if m.any():
            rng = torch.clamp(xw[m, i_rr], min=0.0)
            clo = torch.clamp(xw[m, i_r], min=0.0)
            ttc[m] = rng / (clo + eps)

    # side TTC uses the boosted side (left vs right) choice
    if dom.eq(2).any():
        if i_l is not None and i_rl is not None:
            mL = (dom == 2) & side_is_left
            if mL.any():
                rng = torch.clamp(xw[mL, i_rl], min=0.0)
                clo = torch.clamp(xw[mL, i_l], min=0.0)
                ttc[mL] = rng / (clo + eps)
        if i_rt is not None and i_rrt is not None:
            mR = (dom == 2) & (~side_is_left)
            if mR.any():
                rng = torch.clamp(xw[mR, i_rrt], min=0.0)
                clo = torch.clamp(xw[mR, i_rt], min=0.0)
                ttc[mR] = rng / (clo + eps)

    # Clip + strict censoring at cap
    ttc = torch.clamp(ttc, min=float(ttc_floor_s), max=float(ttc_cap_s))
    log_ttc = _safe_log(ttc)

    y_c = (math.log(float(ttc_cap_s)) - float(target_mu_y)) / float(target_sigma_y)
    y_new = (log_ttc - float(target_mu_y)) / float(target_sigma_y)

    is_cens_new = (ttc >= float(ttc_cap_s) - 1e-6)
    y_new = torch.where(is_cens_new, torch.full_like(y_new, float(y_c)), y_new)
    cens_new = is_cens_new.to(dtype=censored_mask.dtype)

    x_aug = torch.cat([x_expert_raw, xw], dim=0)
    y_aug = torch.cat([y_expert, y_new], dim=0)
    cens_aug = torch.cat([censored_mask, cens_new], dim=0)

    is_warp = torch.cat(
        [
            torch.zeros((B,), device=device, dtype=torch.bool),
            torch.ones((Bw,), device=device, dtype=torch.bool),
        ],
        dim=0,
    )
    return x_aug, y_aug, cens_aug, is_warp

