from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW

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

def train_gate_one_epoch_raw(model: GateMLP, loader, optimizer: torch.optim.Optimizer, cfg: TrainConfig,
                             warp_cfg: RawWarpConfig, feature_index: Dict[str, int],
                             gate_mean: torch.Tensor, gate_std: torch.Tensor) -> float:
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

        total += float(loss.item()); n += 1
    return total / max(n, 1)

@torch.no_grad()
def eval_gate_raw(model: GateMLP, loader, device: str,
                  gate_mean: torch.Tensor, gate_std: torch.Tensor) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x_raw = batch["x_gate_raw"].to(device)
        y = batch["y_gate"].to(device)
        x = _scale(x_raw, gate_mean, gate_std)
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="mean")
        total += float(loss.item()); n += 1
    return total / max(n, 1)

def train_expert_one_epoch_raw(flow: ConditionalSpline1DFlow, loader, optimizer: torch.optim.Optimizer, cfg: TrainConfig,
                               expert_mean: torch.Tensor, expert_std: torch.Tensor,
                               *, drop_idx: Optional[torch.Tensor] = None,
                               ctx_all_idx: Optional[torch.Tensor] = None,
                               ctx_block_drop_prob: float = 0.0) -> float:
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
        # This prevents memorizing per-segment fingerprints while still allowing ODD conditioning.
        if ctx_block_drop_prob > 0 and ctx_all_idx is not None and ctx_all_idx.numel() > 0:
            if x_raw.numel() > 0:
                drop_mask = (torch.rand((x_raw.shape[0],), device=cfg.device) < ctx_block_drop_prob)
                if drop_mask.any():
                    # NOTE: avoid chained indexing (creates a copy). Use advanced indexing once.
                    x_raw = x_raw.clone()
                    rows = drop_mask.nonzero(as_tuple=True)[0]
                    x_raw[rows.unsqueeze(1), ctx_all_idx.unsqueeze(0)] = expert_mean[ctx_all_idx].unsqueeze(0)
        keep = (m > 0.5)
        if keep.sum() < 2:
            continue
        x = (x_raw[keep] - expert_mean) / (expert_std + 1e-6)
        
        if cfg.input_noise > 0:
            x = x + torch.randn_like(x) * cfg.input_noise
            
        # Censored loss logic
        # If censored (TTC >= cap), maximize Survival = (1 - CDF(y)).
        # Loss = -log(1 - CDF(y))
        # Else, maximize PDF. Loss = -log_prob(y)
        
        is_censored = (batch["censored_mask"].to(cfg.device)[keep] > 0.5)
        
        # We need to compute both because python branching on tensor is slow? 
        # Actually we can mask.
        
        loss_val = torch.zeros_like(y[keep])
        
        x_k = x
        y_k = y[keep]
        
        # 1. Uncensored
        mask_u = ~is_censored
        if mask_u.any():
            loss_val[mask_u] = -flow.log_prob(y_k[mask_u], x_k[mask_u])
            
        # 2. Censored
        mask_c = is_censored
        if mask_c.any():
            # For numerical stability: 1 - CDF = 0.5 * erfc(u / sqrt(2))
            # But flow.cdf returns standard CDF.
            # We can use flow.y_to_u to get u, then use log_erfc derivative or similar?
            # Standard: -log(1 - CDF). 
            # If CDF -> 1, this explodes. Cap at eps.
            
            # Re-use flow.cdf() which calls y_to_u + standard_normal_cdf
            # But standard_normal_cdf uses torch.erf
            # 1 - 0.5(1 + erf) = 0.5(1 - erf) = 0.5 erfc
            # log(0.5 erfc) = -log 2 + log(erfc)
            
            u, _ = flow.y_to_u(y_k[mask_c], x_k[mask_c])
            # standard normal cdf
            # cdf = 0.5 * (1.0 + torch.erf(u / math.sqrt(2.0)))
            # 1 - cdf = 0.5 * (1.0 - torch.erf(u / math.sqrt(2.0))) = 0.5 * torch.erfc(u / math.sqrt(2.0))
            
            surv = 0.5 * torch.erfc(u / 1.41421356)
            surv = torch.clamp(surv, min=1e-6)
            loss_val[mask_c] = -torch.log(surv)

        loss = loss_val.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(flow.parameters(), cfg.grad_clip)
        optimizer.step()

        total += float(loss.item()); n += 1
    return total / max(n, 1)

@torch.no_grad()
def eval_expert_raw(flow: ConditionalSpline1DFlow, loader, device: str,
                    expert_mean: torch.Tensor, expert_std: torch.Tensor,
                    *, drop_idx: Optional[torch.Tensor] = None) -> float:
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
        x = (x_raw[keep] - expert_mean) / (expert_std + 1e-6)
        
        is_censored = (batch["censored_mask"].to(device)[keep] > 0.5)
        loss_val = torch.zeros_like(y[keep])
        x_k = x
        y_k = y[keep]
        
        mask_u = ~is_censored
        if mask_u.any():
            loss_val[mask_u] = -flow.log_prob(y_k[mask_u], x_k[mask_u])
            
        mask_c = is_censored
        if mask_c.any():
            u, _ = flow.y_to_u(y_k[mask_c], x_k[mask_c])
            surv = 0.5 * torch.erfc(u / 1.41421356)
            surv = torch.clamp(surv, min=1e-6)
            loss_val[mask_c] = -torch.log(surv)
            
        loss = loss_val.mean()
        total += float(loss.item()); n += 1
    return total / max(n, 1)
