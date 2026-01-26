from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import torch

from .preprocess import TargetStandardizerState


@dataclass
class RiskConfig:
    """Risk scoring config.

    We compute a *context-aware* hazardous-time threshold s*(·) and then evaluate
    the learned TTC distribution:
        Risk_t = P_gate(X_t) * P(TTC <= s*(v_close) | X_t, gate=1)

    tau: reaction / planning latency (seconds)
    a_max: maximum achievable deceleration (m/s^2)
    ttc_floor/cap: numeric stabilization for TTC
    """

    tau: float = 0.7
    a_max: float = 6.0
    ttc_floor: float = 0.05
    ttc_cap: float = 10.0


def s_star(v_close_mps: torch.Tensor, cfg: RiskConfig) -> torch.Tensor:
    """Speed-dependent TTC threshold.

    We use *closing speed* (line-of-sight relative speed) instead of ego speed.
    This is more physically consistent for collision imminence.

    s*(v_close) = tau + v_close / a_max
    """
    s = cfg.tau + v_close_mps / max(cfg.a_max, 1e-6)
    return torch.clamp(s, cfg.ttc_floor, cfg.ttc_cap)


def compute_risk(
    gate_logits: torch.Tensor,
    expert_flow,
    x_expert_scaled: torch.Tensor,
    raw_closing_speed_mps: torch.Tensor,
    target_std: TargetStandardizerState,
    cfg: RiskConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (risk, p_gate).

    expert_flow.cdf expects the *standardized* y (target) and the scaled x.
    """
    p_gate = torch.sigmoid(gate_logits)

    v = torch.clamp(raw_closing_speed_mps, min=0.0)
    s = s_star(v, cfg)

    y = torch.log(s + 1e-9)
    y_std = (y - target_std.mu_y) / (target_std.sigma_y + target_std.eps)

    p_tail = expert_flow.cdf(y_std, x_expert_scaled)
    return p_gate * p_tail, p_gate
