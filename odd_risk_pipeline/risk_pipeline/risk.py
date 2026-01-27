from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union
import torch

from .preprocess import TargetStandardizerState


@dataclass
class RiskConfig:
    tau: float = 0.7
    a_max: float = 6.0
    ttc_floor: float = 0.05
    ttc_cap: float = 10.0


def s_star(v_close_mps: torch.Tensor, cfg: RiskConfig) -> torch.Tensor:
    s = cfg.tau + v_close_mps / max(cfg.a_max, 1e-6)
    return torch.clamp(s, cfg.ttc_floor, cfg.ttc_cap)


CondInput = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def compute_risk(
    gate_logits: torch.Tensor,
    expert_flow,
    x_expert_scaled: CondInput,
    raw_closing_speed_mps: torch.Tensor,
    target_std: TargetStandardizerState,
    cfg: RiskConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (risk, p_gate).

    Supports:
      - x_expert_scaled as a single tensor (concat conditioning)
      - x_expert_scaled as (x_part, c_part) tuple (FiLM conditioning)
    """
    p_gate = torch.sigmoid(gate_logits)

    v = torch.clamp(raw_closing_speed_mps, min=0.0)
    s = s_star(v, cfg)

    y = torch.log(s + 1e-9)
    y_std = (y - target_std.mu_y) / (target_std.sigma_y + target_std.eps)

    p_tail = expert_flow.cdf(y_std, x_expert_scaled)
    return p_gate * p_tail, p_gate
