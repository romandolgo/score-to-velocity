from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of t in [0, 1].

    t is multiplied by `scale` so that the usual DDPM frequency range (built for
    integer steps 0..1000) applies to continuous time.
    """

    def __init__(
        self, dim: int = 64, scale: float = 1000.0, max_period: float = 10_000.0
    ):
        super().__init__()
        if dim % 2:
            raise ValueError("dim must be even")
        half = dim // 2
        freqs = scale * torch.exp(-math.log(max_period) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: Tensor) -> Tensor:
        args = t[:, None] * self.freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)


class MLP(nn.Module):
    """Time-conditioned vector field (x, t) -> R^out_dim.

    t may be a float, a 0-d tensor, (B,) or (B, 1), so the model can be passed
    directly as a drift to an integrator that calls drift(x, t) with scalar t.
    """

    def __init__(
        self,
        dim: int = 2,
        hidden: int = 256,
        depth: int = 4,
        t_dim: int = 64,
        out_dim: int | None = None,
    ):
        super().__init__()
        self.time = TimeEmbedding(t_dim)
        layers: list[nn.Module] = []
        width = dim + t_dim
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        layers.append(nn.Linear(width, out_dim or dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor | float) -> Tensor:
        t = (
            torch.as_tensor(t, dtype=x.dtype, device=x.device)
            .reshape(-1)
            .expand(x.shape[0])
        )
        return self.net(torch.cat([x, self.time(t)], dim=-1))
