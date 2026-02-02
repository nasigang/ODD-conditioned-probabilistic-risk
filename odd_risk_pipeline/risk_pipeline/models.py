from __future__ import annotations
import math
from typing import Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gather_1d(params: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather along last dim with broadcasting.

    params: [B, K]
    idx:    [B]
    returns [B]
    """
    return params.gather(-1, idx.unsqueeze(-1)).squeeze(-1)


def rqs_inverse(
    z: torch.Tensor,
    params: torch.Tensor,
    y_bound: float,
    *,
    n_bins: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Invert monotone rational quadratic spline.

    Forward (implemented by rqs) maps y -> z.
    This function maps z -> y.

    This implementation follows the standard Durkan et al. (Neural Spline Flows)
    equations, but is kept minimal for 1D.

    Parameters
    ----------
    z : [...]
        Value in spline output space, assumed to be within [-y_bound, y_bound]
        (outside: we clamp and linearly map).
    params : [B, 3*n_bins+1]
        Unconstrained parameters from the conditioning network.
    y_bound : float
        Boundary used in training.
    n_bins : int
        Number of bins.

    Returns
    -------
    y : [...]
        Inverted value.
    """

    # Flatten for convenience
    orig_shape = z.shape
    z = z.reshape(-1)
    B = z.shape[0]

    widths_u = params[:, :n_bins]
    heights_u = params[:, n_bins : 2 * n_bins]
    derivatives_u = params[:, 2 * n_bins :]

    # same parameterization as rqs()
    widths = F.softmax(widths_u, dim=-1)
    heights = F.softmax(heights_u, dim=-1)
    derivatives = F.softplus(derivatives_u) + 1e-3

    cumwidths = torch.cumsum(widths, dim=-1)
    cumheights = torch.cumsum(heights, dim=-1)

    cumwidths = F.pad(cumwidths, (1, 0), value=0.0)
    cumheights = F.pad(cumheights, (1, 0), value=0.0)

    # map to [-y_bound, y_bound]
    cumwidths = 2 * y_bound * cumwidths - y_bound
    cumheights = 2 * y_bound * cumheights - y_bound

    widths = cumwidths[:, 1:] - cumwidths[:, :-1]
    heights = cumheights[:, 1:] - cumheights[:, :-1]
    deltas = heights / widths

    # Clamp z to spline range; outside -> linear identity
    z_clamped = torch.clamp(z, -y_bound + eps, y_bound - eps)

    # Identify bin index using cumheights (since z is in "height" space)
    # searchsorted expects 1D for each row; emulate by comparing
    # bin_idx in [0, n_bins-1]
    # For stability we use torch.searchsorted on CPU-like behavior
    bin_idx = torch.sum(z_clamped.unsqueeze(1) >= cumheights[:, 1:], dim=1)
    bin_idx = torch.clamp(bin_idx, 0, n_bins - 1)

    z0 = _gather_1d(cumheights[:, :-1], bin_idx)
    h = _gather_1d(heights, bin_idx)
    x0 = _gather_1d(cumwidths[:, :-1], bin_idx)
    w = _gather_1d(widths, bin_idx)
    delta = _gather_1d(deltas, bin_idx)
    d0 = _gather_1d(derivatives[:, :-1], bin_idx)
    d1 = _gather_1d(derivatives[:, 1:], bin_idx)

    # Normalized z within the bin
    y = (z_clamped - z0) / h
    y = torch.clamp(y, 0.0 + eps, 1.0 - eps)

    # Invert RQS: solve quadratic for theta in [0,1]
    # Based on the closed-form inverse used in common implementations.
    # Let a, b, c define: a*theta^2 + b*theta + c = 0
    # where theta is within [0,1].
    a = (y * (d0 + d1 - 2 * delta) + (delta - d0))
    b = (y * (2 * delta - d0 - d1) + d0)
    c = -delta * y

    discriminant = b * b - 4 * a * c
    discriminant = torch.clamp(discriminant, min=0.0)
    sqrt_disc = torch.sqrt(discriminant)

    # Choose root that yields theta in [0,1]
    denom = 2 * a
    # Avoid division by zero (a can be ~0 when derivatives match delta)
    denom_safe = torch.where(torch.abs(denom) < 1e-12, torch.full_like(denom, 1e-12), denom)
    theta = (-b + sqrt_disc) / denom_safe
    theta_alt = (-b - sqrt_disc) / denom_safe
    theta = torch.where((theta >= 0.0) & (theta <= 1.0), theta, theta_alt)
    theta = torch.clamp(theta, 0.0, 1.0)

    x = x0 + w * theta
    return x.reshape(orig_shape)


class GateMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    w = alpha * targets + (1 - alpha) * (1 - targets)
    loss = w * (1 - pt).pow(gamma) * bce
    return loss.mean()


def _standard_normal_logprob(z: torch.Tensor) -> torch.Tensor:
    return -0.5 * (math.log(2 * math.pi) + z.pow(2))


def _standard_normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def _unconstrained_to_spline_params(
    un_w: torch.Tensor,
    un_h: torch.Tensor,
    un_d: torch.Tensor,
    left: float,
    right: float,
    bottom: float,
    top: float,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
):
    widths = F.softmax(un_w, dim=-1)
    heights = F.softmax(un_h, dim=-1)
    K = widths.shape[-1]
    widths = min_bin_width + (right - left - K * min_bin_width) * widths
    heights = min_bin_height + (top - bottom - K * min_bin_height) * heights
    cumwidths = torch.cumsum(widths, dim=-1)
    cumheights = torch.cumsum(heights, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), value=0.0) + left
    cumheights = F.pad(cumheights, (1, 0), value=0.0) + bottom
    derivatives = min_derivative + F.softplus(un_d)
    return widths, heights, derivatives, cumwidths, cumheights


def rqs_forward(
    x: torch.Tensor,
    un_w: torch.Tensor,
    un_h: torch.Tensor,
    un_d: torch.Tensor,
    bound: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    left, right, bottom, top = -bound, bound, -bound, bound
    widths, heights, deriv, cumw, cumh = _unconstrained_to_spline_params(
        un_w, un_h, un_d, left, right, bottom, top
    )

    x_clamped = torch.clamp(x, left, right)
    bin_idx = torch.sum(x_clamped[..., None] >= cumw[..., 1:], dim=-1)
    bin_idx = torch.clamp(bin_idx, 0, widths.shape[-1] - 1)

    x0 = cumw.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    w = widths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    y0 = cumh.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    h = heights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

    delta = h / w
    d0 = deriv.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    d1 = deriv.gather(-1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)

    theta = (x_clamped - x0) / w
    omt = 1 - theta
    num = h * (delta * theta.pow(2) + d0 * theta * omt)
    den = delta + (d0 + d1 - 2 * delta) * theta * omt
    y = y0 + num / den

    der_num = delta.pow(2) * (d1 * theta.pow(2) + 2 * delta * theta * omt + d0 * omt.pow(2))
    der_den = den.pow(2)
    der_num = torch.clamp(der_num, min=1e-12)
    der_den = torch.clamp(der_den, min=1e-12)
    logdet = torch.log(der_num) - torch.log(der_den)

    below = x < left
    above = x > right
    y = torch.where(below | above, x, y)
    logdet = torch.where(below | above, torch.zeros_like(logdet), logdet)
    return y, logdet


CondInput = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class ConditionalSpline1DFlow(nn.Module):
    """Conditional 1D Spline Flow with optional FiLM conditioning.

    Backward compatible:
      - cond_mode="concat": condition is a single tensor (B, cond_dim)

    New:
      - cond_mode="film": condition is (x_part, c_part) tuple
        * x_part enters trunk
        * c_part modulates trunk via FiLM (scale/shift)
    """

    def __init__(
        self,
        cond_dim: Optional[int] = None,
        *,
        x_dim: Optional[int] = None,
        c_dim: Optional[int] = None,
        cond_mode: str = "concat",
        num_bins: int = 16,
        hidden: int = 256,
        depth: int = 2,
        tail_bound: float = 5.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_bins = int(num_bins)
        self.tail_bound = float(tail_bound)
        self.cond_mode = str(cond_mode).lower().strip()

        if self.cond_mode not in {"concat", "film"}:
            raise ValueError(f"cond_mode must be 'concat' or 'film', got {cond_mode!r}")

        out_dim = 2 * self.num_bins + (self.num_bins + 1)

        if self.cond_mode == "concat":
            if cond_dim is None:
                if x_dim is not None and c_dim is not None:
                    cond_dim = int(x_dim + c_dim)
                else:
                    raise ValueError("concat mode requires cond_dim (or x_dim+c_dim).")
            self.cond_dim = int(cond_dim)

            layers = []
            d = self.cond_dim
            for _ in range(depth):
                layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
                d = hidden
            layers.append(nn.Linear(d, out_dim))
            self.net = nn.Sequential(*layers)

            self.x_dim = None
            self.c_dim = None
            self.trunk = None
            self.film = None
            self.head = None
            self.depth = int(depth)
            self.hidden = int(hidden)
            self.dropout = float(dropout)

        else:
            if x_dim is None or c_dim is None:
                raise ValueError("film mode requires x_dim and c_dim (split dims).")

            self.x_dim = int(x_dim)
            self.c_dim = int(c_dim)
            self.depth = int(depth)
            self.hidden = int(hidden)
            self.dropout = float(dropout)

            self.trunk = nn.ModuleList()
            self.film = nn.ModuleList()
            d = self.x_dim
            for _ in range(self.depth):
                self.trunk.append(nn.Linear(d, self.hidden))
                self.film.append(nn.Linear(self.c_dim, 2 * self.hidden))
                d = self.hidden
            self.head = nn.Linear(self.hidden, out_dim)

            self.net = None
            self.cond_dim = None

    def expects_tuple_condition(self) -> bool:
        return self.cond_mode == "film"

    def _forward_concat(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _forward_film(self, x_part: torch.Tensor, c_part: torch.Tensor) -> torch.Tensor:
        h = x_part
        for lin, film_lin in zip(self.trunk, self.film):
            h = lin(h)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

            if c_part.numel() > 0:
                ss = film_lin(c_part)
                scale, shift = ss.chunk(2, dim=-1)
                h = h * (1.0 + torch.tanh(scale)) + shift
        return self.head(h)

    def _params(self, cond: CondInput):
        if self.cond_mode == "concat":
            if isinstance(cond, (tuple, list)):
                x_part, c_part = cond
                x = torch.cat([x_part, c_part], dim=-1)
            else:
                x = cond
            p = self._forward_concat(x)
        else:
            if not isinstance(cond, (tuple, list)) or len(cond) != 2:
                raise ValueError("FiLM mode expects condition as (x_part, c_part) tuple.")
            x_part, c_part = cond
            p = self._forward_film(x_part, c_part)

        K = self.num_bins
        return p[..., :K], p[..., K : 2 * K], p[..., 2 * K :]

    def y_to_u(self, y: torch.Tensor, cond: CondInput) -> Tuple[torch.Tensor, torch.Tensor]:
        un_w, un_h, un_d = self._params(cond)
        u, logdet = rqs_forward(y, un_w, un_h, un_d, bound=self.tail_bound)
        return u, logdet

    def log_prob(self, y: torch.Tensor, cond: CondInput) -> torch.Tensor:
        u, logdet = self.y_to_u(y, cond)
        return _standard_normal_logprob(u) + logdet

    def nll(self, y: torch.Tensor, cond: CondInput) -> torch.Tensor:
        return -self.log_prob(y, cond).mean()

    def cdf(self, y: torch.Tensor, cond: CondInput) -> torch.Tensor:
        u, _ = self.y_to_u(y, cond)
        return _standard_normal_cdf(u)

    @torch.no_grad()
    def sample(self, cond: CondInput, num_samples: int = 1) -> torch.Tensor:
        """Sample y (in the model's y-space: standardized logTTC).

        This draws u ~ N(0,1) and applies the inverse spline to obtain y.

        Returns
        -------
        y_s : [B, num_samples]
        """
        if isinstance(cond, (tuple, list)):
            B = cond[0].shape[0]
            device = cond[0].device
        else:
            B = cond.shape[0]
            device = cond.device

        un_w, un_h, un_d = self._params(cond)
        params = torch.cat([un_w, un_h, un_d], dim=-1)

        S = int(num_samples)
        z = torch.randn((B, S), device=device)

        # flatten + repeat params
        zf = z.reshape(-1)
        pf = params.repeat_interleave(S, dim=0)
        yf = rqs_inverse(zf, pf, self.tail_bound, n_bins=self.num_bins)
        return yf.reshape(B, S)
