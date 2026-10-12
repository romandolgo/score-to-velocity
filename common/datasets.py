from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor


def _as_scale(v, x: Tensor) -> Tensor:
    """Scalar, (B,) or (B, 1) -> tensor broadcastable against (B, K, d, d)."""
    v = torch.as_tensor(v, dtype=x.dtype, device=x.device).reshape(-1)
    return v.expand(x.shape[0])[:, None, None, None]


class GaussianMixture:
    """Mixture sum_k w_k N(mu_k, Sigma_k) in R^d, doubling as an exact oracle.

    If x ~ p and x_t = alpha x + sigma eps with eps ~ N(0, I), then x_t is again
    a mixture with centers alpha mu_k and covariances alpha^2 Sigma_k + sigma^2 I.
    `log_prob` and `score` take (alpha, sigma) and return log p_t and grad log p_t
    of that noised mixture in closed form; the defaults alpha = 1, sigma = 0 give
    the clean density. alpha and sigma may be floats or per-sample tensors of
    shape (B,) or (B, 1), so a training batch with random t can be scored at once.
    """

    def __init__(self, weights, means, covs=None, std: float = 1.0):
        self.means = torch.as_tensor(means, dtype=torch.float64)
        k, d = self.means.shape
        w = torch.as_tensor(weights, dtype=torch.float64)
        self.weights = w / w.sum()
        if covs is None:
            covs = std**2 * torch.eye(d, dtype=torch.float64).expand(k, d, d)
        self.covs = torch.as_tensor(covs, dtype=torch.float64).expand(k, d, d).clone()

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    def noised(self, alpha: float, sigma: float) -> GaussianMixture:
        """Distribution of alpha x + sigma eps as a new mixture."""
        eye = torch.eye(self.dim, dtype=torch.float64)
        return GaussianMixture(
            self.weights, alpha * self.means, alpha**2 * self.covs + sigma**2 * eye
        )

    def sample(
        self,
        n: int,
        gen: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
        return_labels: bool = False,
    ):
        labels = torch.multinomial(self.weights, n, replacement=True, generator=gen)
        eps = torch.randn(n, self.dim, generator=gen, dtype=torch.float64)
        chol = torch.linalg.cholesky(self.covs)[labels]
        x = self.means[labels] + (chol @ eps[..., None])[..., 0]
        x = x.to(dtype)
        return (x, labels) if return_labels else x

    def _parts(self, x: Tensor, alpha, sigma) -> tuple[Tensor, Tensor]:
        """Per-component log w_k N(x; alpha mu_k, C_k) and C_k^{-1}(x - alpha mu_k)."""
        means = self.means.to(x)
        alpha, sigma = _as_scale(alpha, x), _as_scale(sigma, x)
        eye = torch.eye(self.dim, dtype=x.dtype, device=x.device)
        cov = alpha**2 * self.covs.to(x) + sigma**2 * eye  # (B, K, d, d)
        diff = x[:, None, :] - alpha[..., 0] * means  # (B, K, d)
        chol = torch.linalg.cholesky(cov)
        prec_diff = torch.cholesky_solve(diff[..., None], chol)[..., 0]
        maha = (diff * prec_diff).sum(-1)
        logdet = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        log_comp = (
            self.weights.to(x).log()
            - 0.5 * (maha + logdet + self.dim * math.log(2 * math.pi))
        )
        return log_comp, prec_diff

    def log_prob(self, x: Tensor, alpha=1.0, sigma=0.0) -> Tensor:
        """log p_t(x), shape (B,)."""
        log_comp, _ = self._parts(x, alpha, sigma)
        return log_comp.logsumexp(-1)

    def score(self, x: Tensor, alpha=1.0, sigma=0.0) -> Tensor:
        """grad_x log p_t(x) = -sum_k r_k(x) C_k^{-1}(x - alpha mu_k), shape (B, d)."""
        log_comp, prec_diff = self._parts(x, alpha, sigma)
        resp = log_comp.softmax(-1)
        return -(resp[..., None] * prec_diff).sum(1)

    def posterior(self, x: Tensor, alpha=1.0, sigma=0.0) -> Tensor:
        """Responsibilities p(k | x_t), shape (B, K)."""
        log_comp, _ = self._parts(x, alpha, sigma)
        return log_comp.softmax(-1)


# ─────────────────────────────── presets ───────────────────────────────


def two_modes() -> GaussianMixture:
    """Section 3 of Song & Ermon (2019): 0.8 N((5, 5), I) + 0.2 N((-5, -5), I).

    Langevin dynamics with the exact score fails to recover the 0.8/0.2 weights.
    """
    return GaussianMixture([0.8, 0.2], [[5.0, 5.0], [-5.0, -5.0]], std=1.0)


def eight_gaussians(radius: float = 4.0, std: float = 0.3) -> GaussianMixture:
    angles = torch.arange(8, dtype=torch.float64) * (2 * math.pi / 8)
    means = radius * torch.stack([angles.cos(), angles.sin()], dim=1)
    return GaussianMixture(torch.ones(8), means, std=std)


def grid_gaussians(n: int = 5, spacing: float = 2.0, std: float = 0.2):
    """n x n equally weighted Gaussians on a square grid centered at the origin."""
    ax = (torch.arange(n, dtype=torch.float64) - (n - 1) / 2) * spacing
    means = torch.cartesian_prod(ax, ax)
    return GaussianMixture(torch.ones(n * n), means, std=std)


# ──────────────────── samplers without closed-form density ────────────────────


def sample_moons(
    n: int,
    noise: float = 0.05,
    gen: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Two interleaving half circles, centered and scaled to roughly [-3, 3] x [-1.5, 1.5]."""
    n_out = n // 2
    theta = math.pi * torch.rand(n, generator=gen, dtype=torch.float64)
    x = torch.stack([theta.cos(), theta.sin()], dim=1)
    x[n_out:] = torch.stack([1 - theta[n_out:].cos(), 0.5 - theta[n_out:].sin()], dim=1)
    x = x + noise * torch.randn(n, 2, generator=gen, dtype=torch.float64)
    x = 2 * (x - torch.tensor([0.5, 0.25], dtype=torch.float64))
    return x[torch.randperm(n, generator=gen)].to(dtype)


def sample_checkerboard(
    n: int,
    gen: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Uniform on the black cells of a 4 x 4 checkerboard covering [-4, 4]^2."""
    x1 = 4 * torch.rand(n, generator=gen, dtype=torch.float64) - 2
    x2 = torch.rand(n, generator=gen, dtype=torch.float64)
    x2 = x2 - 2 * torch.randint(0, 2, (n,), generator=gen) + x1.floor() % 2
    return (2 * torch.stack([x1, x2], dim=1)).to(dtype)


MIXTURES = {
    "two_modes": two_modes,
    "eight_gaussians": eight_gaussians,
    "grid": grid_gaussians,
}


def sample(name: str, n: int, gen: torch.Generator | None = None, **kwargs) -> Tensor:
    """Sample any dataset by name: a mixture preset, "moons" or "checkerboard"."""
    if name in MIXTURES:
        return MIXTURES[name]().sample(n, gen=gen, **kwargs)
    if name == "moons":
        return sample_moons(n, gen=gen, **kwargs)
    if name == "checkerboard":
        return sample_checkerboard(n, gen=gen, **kwargs)
    raise ValueError(f"unknown dataset {name!r}")


def names() -> Sequence[str]:
    return [*MIXTURES, "moons", "checkerboard"]
