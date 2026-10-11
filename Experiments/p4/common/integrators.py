"""Интеграторы ODE и SDE с общим интерфейсом (П0, пара B).

Интерфейс один и тот же для всех задач проекта:
    drift(x, t) -> тензор той же формы, что x      — дрейф
    g(t)        -> число или тензор формы (batch,1) — коэффициент диффузии
    ts          -> 1D-тензор сетки по времени

Сетка может идти как вперёд (0 -> T), так и назад (T -> 0). Шаг dt = t[k+1] - t[k]
берётся со знаком, поэтому одна и та же функция решает и прямое, и обратное SDE:
для обратного SDE dt < 0, а шум масштабируется на sqrt(|dt|).
"""
import torch


def _as_col(v, x):
    """Привести g(t) к форме, которая корректно умножается на x формы (batch, d)."""
    v = torch.as_tensor(v, dtype=x.dtype, device=x.device)
    return v.reshape(-1, 1) if v.dim() > 0 else v


@torch.no_grad()
def euler_ode(x0, drift, ts, return_traj=False):
    """Метод Эйлера для dx = drift(x, t) dt.

    x_{k+1} = x_k + drift(x_k, t_k) * (t_{k+1} - t_k)
    Число вычислений drift (NFE) = len(ts) - 1.
    """
    x = x0.clone()
    traj = [x.clone()] if return_traj else None
    for k in range(len(ts) - 1):
        t, dt = ts[k], ts[k + 1] - ts[k]
        x = x + drift(x, t) * dt
        if return_traj:
            traj.append(x.clone())
    return (x, torch.stack(traj)) if return_traj else x


@torch.no_grad()
def euler_maruyama(x0, drift, g, ts, generator=None, noise_last=True, return_traj=False):
    """Метод Эйлера–Маруямы для dx = drift(x, t) dt + g(t) dW.

    x_{k+1} = x_k + drift(x_k, t_k) * dt + g(t_k) * sqrt(|dt|) * xi_k,  xi_k ~ N(0, I).

    noise_last=False: на последнем шаге шум не добавляется. Это стандартный приём
    при генерации: последний шаг выдаёт «среднее», а не ещё раз зашумлённую точку.
    """
    x = x0.clone()
    traj = [x.clone()] if return_traj else None
    n = len(ts) - 1
    for k in range(n):
        t, dt = ts[k], ts[k + 1] - ts[k]
        x = x + drift(x, t) * dt
        if noise_last or k < n - 1:
            xi = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
            x = x + _as_col(g(t), x) * torch.sqrt(torch.abs(dt)) * xi
        if return_traj:
            traj.append(x.clone())
    return (x, torch.stack(traj)) if return_traj else x
