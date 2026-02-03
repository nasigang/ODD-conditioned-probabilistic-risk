from __future__ import annotations
import math
from typing import Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    # Expand spline params to match x's extra (sample) dims.
    # widths/heights/cumw/cumh/deriv are per-row (batch) parameters.
    def _expand_param(p: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        q = p
        for _ in range(ref.dim() - 1):
            q = q.unsqueeze(1)
        return q.expand(*ref.shape, p.shape[-1])

    widths_e = _expand_param(widths, x_clamped)
    heights_e = _expand_param(heights, x_clamped)
    deriv_e = _expand_param(deriv, x_clamped)
    cumw_e = _expand_param(cumw, x_clamped)
    cumh_e = _expand_param(cumh, x_clamped)

    bin_idx = torch.sum(x_clamped[..., None] >= cumw_e[..., 1:], dim=-1)
    bin_idx = torch.clamp(bin_idx, 0, widths.shape[-1] - 1)

    x0 = cumw_e.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    w = widths_e.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    y0 = cumh_e.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    h = heights_e.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

    delta = h / w
    d0 = deriv_e.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    d1 = deriv_e.gather(-1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)

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


def rqs_inverse(
    y: torch.Tensor,
    un_w: torch.Tensor,
    un_h: torch.Tensor,
    un_d: torch.Tensor,
    bound: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of rqs_forward (monotone rational-quadratic spline).

    Notes
    -----
    - This implements the analytic inverse used in spline flows (Durkan et al.).
    - Outside [-bound, bound], the transform is identity (same as rqs_forward).

    Returns
    -------
    x : torch.Tensor
        Inverse-mapped value.
    logdet : torch.Tensor
        log|dx/dy| (inverse log-det Jacobian). Useful if you later need u->y log-prob.
    """
    left, right, bottom, top = -bound, bound, -bound, bound
    widths, heights, deriv, cumw, cumh = _unconstrained_to_spline_params(
        un_w, un_h, un_d, left, right, bottom, top
    )

    y_clamped = torch.clamp(y, bottom, top)

    # Expand spline params to match y's extra (sample) dims.
    def _expand_param(p: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        q = p
        for _ in range(ref.dim() - 1):
            q = q.unsqueeze(1)
        return q.expand(*ref.shape, p.shape[-1])

    widths_e = _expand_param(widths, y_clamped)
    heights_e = _expand_param(heights, y_clamped)
    deriv_e = _expand_param(deriv, y_clamped)
    cumw_e = _expand_param(cumw, y_clamped)
    cumh_e = _expand_param(cumh, y_clamped)

    # choose bin by y (cumheights)
    bin_idx = torch.sum(y_clamped[..., None] >= cumh_e[..., 1:], dim=-1)
    bin_idx = torch.clamp(bin_idx, 0, widths.shape[-1] - 1)

    y0 = cumh_e.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    h = heights_e.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    x0 = cumw_e.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    w = widths_e.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

    delta = h / w
    d0 = deriv_e.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    d1 = deriv_e.gather(-1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)

    # a in [0,1]
    a = (y_clamped - y0) / torch.clamp(h, min=1e-12)
    A = d0 + d1 - 2.0 * delta

    # Solve quadratic for theta (see nflows / NSF-RQS inverse).
    c2 = a * A + delta - d0
    c1 = d0 - a * A
    c0 = -a * delta

    disc = torch.clamp(c1.pow(2) - 4.0 * c2 * c0, min=1e-12)
    sqrt_disc = torch.sqrt(disc)

    # Numerically stable root in [0,1]
    denom = (-c1 - sqrt_disc)
    denom = torch.where(denom.abs() < 1e-12, denom.sign() * 1e-12, denom)
    theta = (2.0 * c0) / denom
    theta = torch.clamp(theta, 0.0, 1.0)

    x = x0 + theta * w

    # Inverse logdet: log|dx/dy| = -log|dy/dx|
    omt = 1.0 - theta
    den = delta + A * theta * omt
    der_num = delta.pow(2) * (d1 * theta.pow(2) + 2.0 * delta * theta * omt + d0 * omt.pow(2))
    der_den = torch.clamp(den.pow(2), min=1e-12)
    der_num = torch.clamp(der_num, min=1e-12)
    logdet_fwd = torch.log(der_num) - torch.log(der_den)
    logdet_inv = -logdet_fwd

    below = y < bottom
    above = y > top
    x = torch.where(below | above, y, x)
    logdet_inv = torch.where(below | above, torch.zeros_like(logdet_inv), logdet_inv)
    return x, logdet_inv


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

    def u_to_y(self, u: torch.Tensor, cond: CondInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse map (latent u -> standardized y).

        This is the missing piece needed for Monte-Carlo sampling / uncertainty diagnostics.
        """
        un_w, un_h, un_d = self._params(cond)
        y, logdet = rqs_inverse(u, un_w, un_h, un_d, bound=self.tail_bound)
        return y, logdet

    @torch.no_grad()
    def sample(self, cond: CondInput, *, num_samples: int = 1) -> torch.Tensor:
        """Sample standardized y from the conditional flow.

        Parameters
        ----------
        cond : Tensor or (x_part, c_part)
            Condition vector(s). Shape is [B, D] (concat) or ([B, x_dim], [B, c_dim]) (FiLM).
        num_samples : int
            Number of samples per condition (S).

        Returns
        -------
        y : torch.Tensor
            Samples in standardized y-space with shape [B, S].
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")

        # Determine batch size + device/dtype from cond
        if isinstance(cond, (tuple, list)):
            B = int(cond[0].shape[0])
            device = cond[0].device
            dtype = cond[0].dtype
        else:
            B = int(cond.shape[0])
            device = cond.device
            dtype = cond.dtype

        u = torch.randn((B, int(num_samples)), device=device, dtype=dtype)
        y, _ = self.u_to_y(u, cond)
        return y

    def log_prob(self, y: torch.Tensor, cond: CondInput) -> torch.Tensor:
        u, logdet = self.y_to_u(y, cond)
        return _standard_normal_logprob(u) + logdet

    def nll(self, y: torch.Tensor, cond: CondInput) -> torch.Tensor:
        return -self.log_prob(y, cond).mean()

    def cdf(self, y: torch.Tensor, cond: CondInput) -> torch.Tensor:
        u, _ = self.y_to_u(y, cond)
        return _standard_normal_cdf(u)
