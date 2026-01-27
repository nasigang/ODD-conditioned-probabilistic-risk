from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import torch

@dataclass
class RawWarpConfig:
    p_warp: float = 0.35

    # speed scale (>=1 to increase speed)
    speed_scale_min: float = 1.05
    speed_scale_max: float = 1.25

    # accel / jerk magnitude boost (m/s^2, m/s^3)
    accel_mag_add_min: float = 0.5
    accel_mag_add_max: float = 2.0
    jerk_mag_add_min: float = 2.0
    jerk_mag_add_max: float = 8.0

    # yaw rate magnitude boost (rad/s)
    yaw_mag_add_min: float = 0.0
    yaw_mag_add_max: float = 0.5

    # physical caps
    speed_cap_mps: float = 60.0
    accel_cap_mps2: float = 10.0
    jerk_cap_mps3: float = 30.0
    yaw_cap_rps: float = 3.0

def _rand_uniform(shape, low, high, device):
    return (high - low) * torch.rand(shape, device=device) + low

def _boost_magnitude(x: torch.Tensor, add: torch.Tensor) -> torch.Tensor:
    # increase |x| by add in the same sign direction; if x≈0, random sign
    sign = torch.sign(x)
    zero = (sign == 0)
    if zero.any():
        sign = torch.where(zero, torch.where(torch.rand_like(sign) < 0.5, -torch.ones_like(sign), torch.ones_like(sign)), sign)
    return x + sign * add

def kinematic_warp_raw_x_gate(
    x_gate_raw: torch.Tensor,
    y_gate: torch.Tensor,
    feature_index: Dict[str, int],
    cfg: RawWarpConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Raw-space physical warp (Gate-only).
    Input x_gate_raw must be raw physical units for dynamics features.
    Output concatenates original + warped subset; warped labels are set to 1.
    '''
    device = x_gate_raw.device
    B, _ = x_gate_raw.shape
    neg_mask = (y_gate < 0.5)
    if neg_mask.sum() == 0:
        return x_gate_raw, y_gate, torch.zeros(B, device=device, dtype=torch.bool)

    warp_mask = neg_mask & (torch.rand(B, device=device) < cfg.p_warp)
    idx = torch.where(warp_mask)[0]
    if idx.numel() == 0:
        return x_gate_raw, y_gate, torch.zeros(B, device=device, dtype=torch.bool)

    xw = x_gate_raw[idx].clone()

    def has(name): return name in feature_index

    # Speed: scale up
    if has("x__ego_speed_mps"):
        j = feature_index["x__ego_speed_mps"]
        scale = _rand_uniform((idx.numel(), 1), cfg.speed_scale_min, cfg.speed_scale_max, device)
        xw[:, j:j+1] = torch.clamp(xw[:, j:j+1] * scale, 0.0, cfg.speed_cap_mps)

    # Accel: boost magnitude
    if has("x__ego_accel_mps2"):
        j = feature_index["x__ego_accel_mps2"]
        add = _rand_uniform((idx.numel(), 1), cfg.accel_mag_add_min, cfg.accel_mag_add_max, device)
        xw[:, j:j+1] = torch.clamp(_boost_magnitude(xw[:, j:j+1], add), -cfg.accel_cap_mps2, cfg.accel_cap_mps2)

    # Jerk: boost magnitude
    if has("x__ego_jerk_mps3"):
        j = feature_index["x__ego_jerk_mps3"]
        add = _rand_uniform((idx.numel(), 1), cfg.jerk_mag_add_min, cfg.jerk_mag_add_max, device)
        xw[:, j:j+1] = torch.clamp(_boost_magnitude(xw[:, j:j+1], add), -cfg.jerk_cap_mps3, cfg.jerk_cap_mps3)

    # Yaw rate: boost magnitude
    if has("x__ego_yaw_rate_rps"):
        j = feature_index["x__ego_yaw_rate_rps"]
        add = _rand_uniform((idx.numel(), 1), cfg.yaw_mag_add_min, cfg.yaw_mag_add_max, device)
        xw[:, j:j+1] = torch.clamp(_boost_magnitude(xw[:, j:j+1], add), -cfg.yaw_cap_rps, cfg.yaw_cap_rps)

    # Lags: keep rough temporal consistency
    for lag in ["x__ego_speed_lag1","x__ego_speed_lag2"]:
        if has(lag):
            j = feature_index[lag]
            scale = _rand_uniform((idx.numel(), 1), cfg.speed_scale_min, cfg.speed_scale_max, device)
            xw[:, j:j+1] = torch.clamp(xw[:, j:j+1] * scale, 0.0, cfg.speed_cap_mps)

    for lag in ["x__ego_accel_lag1","x__ego_accel_lag2"]:
        if has(lag):
            j = feature_index[lag]
            add = _rand_uniform((idx.numel(), 1), cfg.accel_mag_add_min, cfg.accel_mag_add_max, device)
            xw[:, j:j+1] = torch.clamp(_boost_magnitude(xw[:, j:j+1], add), -cfg.accel_cap_mps2, cfg.accel_cap_mps2)

    x_aug = torch.cat([x_gate_raw, xw], dim=0)
    y_aug = torch.cat([y_gate, torch.ones(idx.numel(), device=device)], dim=0)
    is_warp = torch.cat([
        torch.zeros(B, device=device, dtype=torch.bool),
        torch.ones(idx.numel(), device=device, dtype=torch.bool),
    ], dim=0)
    return x_aug, y_aug, is_warp
