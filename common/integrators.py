from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

Drift = Callable[[Tensor, Tensor], Tensor]
Diffusion = Callable[[Tensor], Tensor | float] | float


def time_grid(t0: float, t1: float, n_steps: int, dtype=torch.float32) -> Tensor:
    """Uniform grid of n_steps + 1 points; t1 < t0 integrates backward in time."""
    return torch.linspace(t0, t1, n_steps + 1, dtype=dtype)


def integrate(
    drift: Drift,
    x0: Tensor,
    ts: Tensor,
    g: Diffusion | None = None,
    gen: torch.Generator | None = None,
    return_traj: bool = False,
    noise_last: bool = True,
) -> Tensor:
    """Euler-Maruyama for dx = f(x, t) dt + g(t) dW on the time grid ts.

        x_{k+1} = x_k + f(x_k, t_k) dt_k + g(t_k) sqrt(|dt_k|) xi_k,  xi_k ~ N(0, I)

    With g = None (or 0) this is the explicit Euler method for the ODE dx = f dt.
    ts may be decreasing (dt < 0) to run a reverse-time SDE from t = 1 to t = 0;
    the drift is then the full reverse drift, e.g. f - g^2 score.

    drift(x, t) gets x of shape (B, d) and a 0-d tensor t, as `models.MLP` expects.
    g is a constant or a callable of t. Returns x at ts[-1], or the whole
    trajectory of shape (T, B, d) if return_traj, which is what `straightness`
    takes. With noise_last = False the last step adds no noise, so a sampler
    returns the denoised point. Gradients are tracked, so wrap sampling in
    torch.no_grad().
    """
    ts = torch.as_tensor(ts, dtype=x0.dtype, device=x0.device)
    x = x0
    traj = [x] if return_traj else None
    last = len(ts) - 2
    for k, (t, t_next) in enumerate(zip(ts[:-1], ts[1:])):
        dt = t_next - t
        x = x + drift(x, t) * dt
        if g is not None and (noise_last or k < last):
            gt = g(t) if callable(g) else g
            noise = torch.randn(x.shape, generator=gen, dtype=x.dtype).to(x.device)
            x = x + gt * dt.abs().sqrt() * noise
        if return_traj:
            traj.append(x)
    return torch.stack(traj) if return_traj else x


def euler(drift: Drift, x0: Tensor, ts: Tensor, return_traj: bool = False) -> Tensor:
    """Explicit Euler for dx = f(x, t) dt."""
    return integrate(drift, x0, ts, return_traj=return_traj)


def euler_maruyama(
    drift: Drift,
    g: Diffusion,
    x0: Tensor,
    ts: Tensor,
    gen: torch.Generator | None = None,
    return_traj: bool = False,
    noise_last: bool = True,
) -> Tensor:
    """Euler-Maruyama for dx = f(x, t) dt + g(t) dW."""
    return integrate(
        drift, x0, ts, g=g, gen=gen, return_traj=return_traj, noise_last=noise_last
    )
