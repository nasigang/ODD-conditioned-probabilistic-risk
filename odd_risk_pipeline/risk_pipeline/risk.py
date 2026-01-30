from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union, Any
import torch


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


def _get_target_std_params(target_std: Any, *, eps_default: float = 1e-6) -> Tuple[float, float, float]:
    """Compatibility shim across preprocess versions.

    New preprocess uses TargetStandardizer(mean, std).
    Legacy used TargetStandardizerState(mu_y, sigma_y, eps).
    """
    if hasattr(target_std, "mean") and hasattr(target_std, "std"):
        mu = float(getattr(target_std, "mean"))
        sig = float(getattr(target_std, "std"))
        eps = float(getattr(target_std, "eps", eps_default))
        return mu, sig, eps
    if hasattr(target_std, "mu_y") and hasattr(target_std, "sigma_y"):
        mu = float(getattr(target_std, "mu_y"))
        sig = float(getattr(target_std, "sigma_y"))
        eps = float(getattr(target_std, "eps", eps_default))
        return mu, sig, eps
    # fallback
    return 0.0, 1.0, eps_default


def compute_risk(
    gate_logits: torch.Tensor,
    expert_flow,
    x_expert_scaled: CondInput,
    raw_closing_speed_mps: torch.Tensor,
    target_std: Any,
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

    mu, sig, eps = _get_target_std_params(target_std)
    y = torch.log(s + 1e-9)
    y_std = (y - mu) / (sig + eps)

    p_tail = expert_flow.cdf(y_std, x_expert_scaled)
    return p_gate * p_tail, p_gate
