from __future__ import annotations
import math
from typing import Tuple
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

def focal_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
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
    un_w: torch.Tensor, un_h: torch.Tensor, un_d: torch.Tensor,
    left: float, right: float, bottom: float, top: float,
    min_bin_width: float = 1e-3, min_bin_height: float = 1e-3, min_derivative: float = 1e-3
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

def rqs_forward(x: torch.Tensor, un_w: torch.Tensor, un_h: torch.Tensor, un_d: torch.Tensor,
                bound: float = 5.0) -> Tuple[torch.Tensor, torch.Tensor]:
    left, right, bottom, top = -bound, bound, -bound, bound
    widths, heights, deriv, cumw, cumh = _unconstrained_to_spline_params(un_w, un_h, un_d, left, right, bottom, top)

    x_clamped = torch.clamp(x, left, right)
    bin_idx = torch.sum(x_clamped[..., None] >= cumw[..., 1:], dim=-1)
    bin_idx = torch.clamp(bin_idx, 0, widths.shape[-1] - 1)

    x0 = cumw.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    w  = widths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    y0 = cumh.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    h  = heights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

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

class ConditionalSpline1DFlow(nn.Module):
    '''
    Conditional 1D Spline Flow.
    y is a standardized 1D target (e.g., (log TTC - mu)/sigma).
    u = f(y; x), log p(y|x)=log N(u)+log|du/dy|, CDF(y|x)=Phi(u).
    '''
    def __init__(self, cond_dim: int, num_bins: int = 16, hidden: int = 256, depth: int = 2, tail_bound: float = 5.0, dropout: float = 0.0):
        super().__init__()
        self.num_bins = num_bins
        self.tail_bound = float(tail_bound)
        out_dim = 2 * num_bins + (num_bins + 1)
        layers = []
        d = cond_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def _params(self, x: torch.Tensor):
        p = self.net(x)
        K = self.num_bins
        return p[..., :K], p[..., K:2*K], p[..., 2*K:]

    def y_to_u(self, y: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        un_w, un_h, un_d = self._params(x)
        u, logdet = rqs_forward(y, un_w, un_h, un_d, bound=self.tail_bound)
        return u, logdet

    def log_prob(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        u, logdet = self.y_to_u(y, x)
        return _standard_normal_logprob(u) + logdet

    def nll(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(y, x).mean()

    def cdf(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        u, _ = self.y_to_u(y, x)
        return _standard_normal_cdf(u)
