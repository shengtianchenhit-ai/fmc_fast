"""Phase-1b reconstruction network: a compact 2D U-Net over the (tx, rx) plane with
in-band frequencies as channels, plus zero-parameter hard constraints (reciprocity
+ data consistency) so the net only ever predicts the unobserved block.

Torch-only; kept out of fmcfast/__init__ so the numpy core stays torch-free.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """3-level U-Net. cin = 2F+1 (Re,Im of observed M_f + mask), cout = 2F."""

    def __init__(self, cin: int, cout: int, base: int = 64):
        super().__init__()
        self.stem = nn.Conv2d(cin, base, 3, padding=1)
        self.e1 = ConvBlock(base, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.AvgPool2d(2)
        self.bott = ConvBlock(base * 4, base * 4)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.d3 = ConvBlock(base * 4 + base * 4, base * 2)
        self.d2 = ConvBlock(base * 2 + base * 2, base)
        self.d1 = ConvBlock(base + base, base)
        self.head = nn.Conv2d(base, cout, 1)

    def forward(self, x):
        x = self.stem(x)
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        b = self.bott(self.pool(s3))
        d = self.d3(torch.cat([self.up(b), s3], 1))
        d = self.d2(torch.cat([self.up(d), s2], 1))
        d = self.d1(torch.cat([self.up(d), s1], 1))
        return self.head(d)


def apply_constraints(raw: torch.Tensor, mask: torch.Tensor, x_obs: torch.Tensor,
                      F: int) -> torch.Tensor:
    """Hard reciprocity (M = M^T per bin) then data consistency on observed entries.

    raw, x_obs: (B, 2F, N, N) split as [Re(F), Im(F)]. mask: (B, 1, N, N) or (B, N, N).
    x_obs holds the (normalized) observed M_f; observed entries are passed through
    exactly, so the net only predicts the unobserved block.
    """
    re, im = raw[:, :F], raw[:, F:]
    re = 0.5 * (re + re.transpose(-1, -2))
    im = 0.5 * (im + im.transpose(-1, -2))
    out = torch.cat([re, im], 1)
    m = mask.unsqueeze(1) if mask.dim() == 3 else mask
    return m * x_obs + (1.0 - m) * out


def cube_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Phase-sensitive relative L2 on the UNOBSERVED block (= training-time band_nrmse).

    Real+imag channels jointly → phase-sensitive. Observed entries excluded (pinned
    exact by data consistency), so the gradient targets only what the net predicts.
    """
    m = mask.unsqueeze(1) if mask.dim() == 3 else mask
    U = 1.0 - m
    num = ((pred - target) ** 2 * U).sum()
    den = (target ** 2 * U).sum() + 1e-8
    return torch.sqrt(num / den)
