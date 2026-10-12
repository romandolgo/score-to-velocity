from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor


def _as_pair(x, y) -> tuple[Tensor, Tensor]:
    x = torch.as_tensor(x)
    y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
    return x, y


def _subsample(x: Tensor, n: int | None, gen: torch.Generator) -> Tensor:
    if n is None or len(x) <= n:
        return x
    return x[torch.randperm(len(x), generator=gen)[:n].to(x.device)]


def sliced_wasserstein(x, y, n_proj: int = 256, p: int = 2, seed: int = 0) -> float:
    """Sliced p-Wasserstein distance between samples x (N, d) and y (M, d)."""
    x, y = _as_pair(x, y)
    gen = torch.Generator().manual_seed(seed)
    dirs = torch.randn(n_proj, x.shape[1], generator=gen).to(x)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    px = (x @ dirs.T).sort(dim=0).values
    py = (y @ dirs.T).sort(dim=0).values
    if len(px) != len(py):
        n = min(len(px), len(py))
        q = (torch.arange(n, device=x.device) + 0.5) / n
        px = px[(q * len(px)).long()]
        py = py[(q * len(py)).long()]
    return (px - py).abs().pow(p).mean().pow(1 / p).item()


def sliced_w1(x, y, n_proj: int = 512, seed: int = 0) -> float:
    """Sliced 1-Wasserstein distance of P2 and P4; the longer sample is truncated.

    Separate from `sliced_wasserstein` because the stored results of P2 and P4
    depend on how this one draws its directions.
    """
    x, y = _as_pair(x, y)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    gen = torch.Generator().manual_seed(seed)
    dirs = torch.randn(x.shape[1], n_proj, generator=gen).to(x)
    dirs = dirs / dirs.norm(dim=0, keepdim=True)
    px = (x @ dirs).sort(dim=0).values
    py = (y @ dirs).sort(dim=0).values
    return (px - py).abs().mean().item()


def mmd2(
    x,
    y,
    bandwidth: float | None = None,
    scales: Sequence[float] = (0.5, 1.0, 2.0),
    max_samples: int | None = 2000,
    seed: int = 0,
) -> float:
    """Unbiased estimate of squared MMD with a mixture of RBF kernels.

    Can be slightly negative when the two distributions match. If `bandwidth` is
    None it is the median pairwise distance within y, so pass the reference
    sample as y to keep the kernel fixed across the models being compared.
    Memory is O(max_samples^2); larger samples are subsampled using `seed`.
    """
    x, y = _as_pair(x, y)
    gen = torch.Generator().manual_seed(seed)
    x, y = _subsample(x, max_samples, gen), _subsample(y, max_samples, gen)
    if bandwidth is None:
        bandwidth = torch.pdist(y).median().item()

    def kernel(a: Tensor, b: Tensor) -> Tensor:
        d2 = torch.cdist(a, b).pow(2)
        return sum(torch.exp(-d2 / (2 * (s * bandwidth) ** 2)) for s in scales) / len(
            scales
        )

    n, m = len(x), len(y)
    kxx, kyy, kxy = kernel(x, x), kernel(y, y), kernel(x, y)
    within_x = (kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1))
    within_y = (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
    return (within_x + within_y - 2 * kxy.mean()).item()


def mode_fractions(x, centers, radius: float | None = None) -> Tensor:
    """Fraction of samples assigned to each mode by nearest center, shape (K,).

    With `radius` set, samples farther than `radius` from every center are left
    unassigned, so the fractions sum to less than 1.
    """
    x, centers = _as_pair(x, centers)
    dist, idx = torch.cdist(x, centers).min(dim=1)
    if radius is not None:
        idx = idx[dist <= radius]
    return torch.bincount(idx, minlength=len(centers)) / len(x)


def mode_tv(x, centers, weights) -> float:
    """Total variation between the mode fractions of x and the true weights."""
    fractions = mode_fractions(x, centers)
    weights = torch.as_tensor(weights).to(fractions)
    return 0.5 * (fractions - weights).abs().sum().item()


@torch.no_grad()
def straightness(drift: Callable[[Tensor, Tensor], Tensor], traj, ts) -> float:
    """S = int E||(x_1 - x_0) - v(x_t, t)||^2 dt along simulated trajectories.

    traj has shape (T, B, d) and was produced on the time grid ts of shape (T,);
    drift is the field v that generated it. S = 0 iff every path is a straight
    line traversed at constant speed.
    """
    traj = torch.as_tensor(traj)
    ts = torch.as_tensor(ts, dtype=traj.dtype, device=traj.device)
    disp = (traj[-1] - traj[0]) / (ts[-1] - ts[0])
    err = torch.stack(
        [(disp - drift(x, t)).pow(2).sum(-1).mean() for x, t in zip(traj, ts)]
    )
    return torch.trapezoid(err, ts).item()
