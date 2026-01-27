from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Union, Tuple, List

import torch
import torch.nn as nn

from .models import GateMLP, focal_loss_with_logits, ConditionalSpline1DFlow
from .warp import RawWarpConfig, kinematic_warp_raw_x_gate


@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gate_lr: float = 2e-4
    expert_lr: float = 2e-4
    weight_decay: float = 1e-4
    gate_epochs: int = 5
    expert_epochs: int = 10
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    grad_clip: float = 5.0
    input_noise: float = 0.0


def _scale(x_raw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x_raw - mean) / (std + 1e-6)


def _make_flow_cond(
    x_scaled_full: torch.Tensor,
    flow: ConditionalSpline1DFlow,
    *,
    flow_x_idx: Optional[Union[torch.Tensor, List[int]]] = None,
    flow_c_idx: Optional[Union[torch.Tensor, List[int]]] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Return condition input for the flow.

    - concat mode: returns x_scaled_full
    - FiLM mode: returns (x_part, c_part) by slicing the full expert vector

    IMPORTANT
    ---------
    For FiLM mode, you MUST pass indices from preprocess_state.flow_x_idx / flow_c_idx.
    """
    if not hasattr(flow, "expects_tuple_condition") or not flow.expects_tuple_condition():
        return x_scaled_full

    if flow_x_idx is None or flow_c_idx is None:
        raise ValueError(
            "Flow is in FiLM mode but flow_x_idx/flow_c_idx were not provided. "
            "Pass indices from preprocess_state.flow_x_idx / flow_c_idx."
        )

    if isinstance(flow_x_idx, list):
        flow_x_idx = torch.tensor(flow_x_idx, dtype=torch.long, device=x_scaled_full.device)
    if isinstance(flow_c_idx, list):
        flow_c_idx = torch.tensor(flow_c_idx, dtype=torch.long, device=x_scaled_full.device)

    x_part = x_scaled_full.index_select(1, flow_x_idx)
    c_part = x_scaled_full.index_select(1, flow_c_idx)
    return (x_part, c_part)


def train_gate_one_epoch_raw(
    model: GateMLP,
    loader,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    warp_cfg: RawWarpConfig,
    feature_index: Dict[str, int],
    gate_mean: torch.Tensor,
    gate_std: torch.Tensor,
) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        x_raw = batch["x_gate_raw"].to(cfg.device)
        y = batch["y_gate"].to(cfg.device)

        x_aug_raw, y_aug, _ = kinematic_warp_raw_x_gate(x_raw, y, feature_index, warp_cfg)
        x_aug = _scale(x_aug_raw, gate_mean, gate_std)

        if cfg.input_noise > 0:
            x_aug = x_aug + torch.randn_like(x_aug) * cfg.input_noise

        logits = model(x_aug)
        loss = focal_loss_with_logits(logits, y_aug, alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_gate_raw(
    model: GateMLP,
    loader,
    device: str,
    gate_mean: torch.Tensor,
    gate_std: torch.Tensor,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x_raw = batch["x_gate_raw"].to(device)
        y = batch["y_gate"].to(device)
        x = _scale(x_raw, gate_mean, gate_std)
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="mean")
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


# Helper: slice a condition (tensor or tuple) by a boolean mask

def _cond_index(cond_in, mask: torch.Tensor):
    if isinstance(cond_in, tuple):
        return (cond_in[0][mask], cond_in[1][mask])
    return cond_in[mask]


def train_expert_one_epoch_raw(
    flow: ConditionalSpline1DFlow,
    loader,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    expert_mean: torch.Tensor,
    expert_std: torch.Tensor,
    *,
    drop_idx: Optional[torch.Tensor] = None,
    ctx_all_idx: Optional[torch.Tensor] = None,
    ctx_block_drop_prob: float = 0.0,
    flow_x_idx: Optional[Union[torch.Tensor, List[int]]] = None,
    flow_c_idx: Optional[Union[torch.Tensor, List[int]]] = None,
) -> float:
    flow.train()
    total, n = 0.0, 0
    for batch in loader:
        if batch["x_expert_raw"].numel() == 0:
            continue
        x_raw = batch["x_expert_raw"].to(cfg.device)
        y = batch["y_expert"].to(cfg.device)
        m = batch["expert_mask"].to(cfg.device)

        # Hard feature dropping for Expert (e.g., remove segment-fingerprint c__ continuous stats)
        if drop_idx is not None and drop_idx.numel() > 0:
            x_raw = x_raw.clone()
            x_raw[:, drop_idx] = expert_mean[drop_idx]

        # Structured context dropout: randomly neutralize *all* context features (c__ block)
        if ctx_block_drop_prob > 0 and ctx_all_idx is not None and ctx_all_idx.numel() > 0:
            drop_mask = (torch.rand((x_raw.shape[0],), device=cfg.device) < ctx_block_drop_prob)
            if drop_mask.any():
                x_raw = x_raw.clone()
                rows = drop_mask.nonzero(as_tuple=True)[0]
                x_raw[rows.unsqueeze(1), ctx_all_idx.unsqueeze(0)] = expert_mean[ctx_all_idx].unsqueeze(0)

        keep = (m > 0.5)
        if keep.sum() < 2:
            continue

        x_scaled_full = (x_raw[keep] - expert_mean) / (expert_std + 1e-6)
        if cfg.input_noise > 0:
            x_scaled_full = x_scaled_full + torch.randn_like(x_scaled_full) * cfg.input_noise

        cond = _make_flow_cond(x_scaled_full, flow, flow_x_idx=flow_x_idx, flow_c_idx=flow_c_idx)

        is_censored = (batch["censored_mask"].to(cfg.device)[keep] > 0.5)
        loss_val = torch.zeros_like(y[keep])
        y_k = y[keep]

        mask_u = ~is_censored
        if mask_u.any():
            loss_val[mask_u] = -flow.log_prob(y_k[mask_u], _cond_index(cond, mask_u))

        mask_c = is_censored
        if mask_c.any():
            u, _ = flow.y_to_u(y_k[mask_c], _cond_index(cond, mask_c))
            surv = 0.5 * torch.erfc(u / 1.41421356)
            surv = torch.clamp(surv, min=1e-6)
            loss_val[mask_c] = -torch.log(surv)

        loss = loss_val.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(flow.parameters(), cfg.grad_clip)
        optimizer.step()

        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_expert_raw(
    flow: ConditionalSpline1DFlow,
    loader,
    device: str,
    expert_mean: torch.Tensor,
    expert_std: torch.Tensor,
    *,
    drop_idx: Optional[torch.Tensor] = None,
    flow_x_idx: Optional[Union[torch.Tensor, List[int]]] = None,
    flow_c_idx: Optional[Union[torch.Tensor, List[int]]] = None,
) -> float:
    flow.eval()
    total, n = 0.0, 0
    for batch in loader:
        x_raw = batch["x_expert_raw"].to(device)
        y = batch["y_expert"].to(device)
        m = batch["expert_mask"].to(device)

        if drop_idx is not None and drop_idx.numel() > 0:
            x_raw = x_raw.clone()
            x_raw[:, drop_idx] = expert_mean[drop_idx]

        keep = (m > 0.5)
        if keep.sum() < 2:
            continue

        x_scaled_full = (x_raw[keep] - expert_mean) / (expert_std + 1e-6)
        cond = _make_flow_cond(x_scaled_full, flow, flow_x_idx=flow_x_idx, flow_c_idx=flow_c_idx)

        is_censored = (batch["censored_mask"].to(device)[keep] > 0.5)
        loss_val = torch.zeros_like(y[keep])
        y_k = y[keep]

        mask_u = ~is_censored
        if mask_u.any():
            loss_val[mask_u] = -flow.log_prob(y_k[mask_u], _cond_index(cond, mask_u))

        mask_c = is_censored
        if mask_c.any():
            u, _ = flow.y_to_u(y_k[mask_c], _cond_index(cond, mask_c))
            surv = 0.5 * torch.erfc(u / 1.41421356)
            surv = torch.clamp(surv, min=1e-6)
            loss_val[mask_c] = -torch.log(surv)

        loss = loss_val.mean()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)
