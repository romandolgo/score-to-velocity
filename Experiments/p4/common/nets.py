"""MLP с эмбеддингом времени (временная версия; общий модуль ведёт пара C)."""
import math
import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """Синусоидальный эмбеддинг t in [0, 1] -> R^dim (как позиционное кодирование)."""

    def __init__(self, dim=32, max_freq=1000.0):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(torch.linspace(0, math.log(max_freq), half))
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        a = t.reshape(-1, 1) * self.freqs[None]
        return torch.cat([torch.sin(a), torch.cos(a)], 1)


class MLP(nn.Module):
    def __init__(self, d=2, hidden=256, layers=4, t_dim=32):
        super().__init__()
        self.emb = TimeEmbedding(t_dim)
        dims = [d + t_dim] + [hidden] * layers
        mods = []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Linear(a, b), nn.SiLU()]
        mods += [nn.Linear(hidden, d)]
        self.net = nn.Sequential(*mods)

    def forward(self, x, t):
        t = torch.as_tensor(t, dtype=x.dtype).expand(x.shape[0]) if torch.as_tensor(t).dim() == 0 else t
        return self.net(torch.cat([x, self.emb(t)], 1))


class ScoreModel(nn.Module):
    """Сеть предсказывает шум eps; score = -eps / std(t).

    Это эквивалентно лоссу DSM с весом lambda(t) = std(t)^2:
        std^2 * || s + z/std ||^2 = || std * s + z ||^2 = || eps_hat - z ||^2.
    """

    def __init__(self, sde, **kw):
        super().__init__()
        self.sde = sde
        self.net = MLP(**kw)

    def eps(self, x, t):
        return self.net(x, t)

    def score(self, x, t):
        t = torch.as_tensor(t, dtype=x.dtype)
        std = self.sde.std(t)
        std = std.reshape(-1, 1) if std.dim() > 0 else std
        return -self.net(x, t) / std
