"""Сэмплеры и правдоподобие для Score SDE (П2, пара B; пригодится в П4).

score_fn(x, t) -> grad log p_t(x): либо обученная сеть, либо точный score оракула.
"""
import torch
from .integrators import euler_maruyama, euler_ode


def time_grid(sde, n_steps, eps=1e-3, reverse=True):
    """Равномерная сетка от T до eps (reverse=True) или от eps до T."""
    ts = torch.linspace(sde.T, eps, n_steps + 1)
    return ts if reverse else ts.flip(0)


def reverse_sde_drift(sde, score_fn):
    """Дрейф обратного SDE: f - g^2 * score  (Song et al., ур. 6)."""
    return lambda x, t: sde.f(x, t) - sde.g(t) ** 2 * score_fn(x, t)


def pf_ode_drift(sde, score_fn):
    """Дрейф probability flow ODE: f - 1/2 g^2 * score  (Song et al., ур. 13)."""
    return lambda x, t: sde.f(x, t) - 0.5 * sde.g(t) ** 2 * score_fn(x, t)


def sample_reverse_sde(sde, score_fn, n, n_steps, eps=1e-3, generator=None):
    """Эйлер–Маруяма для обратного SDE от T до eps. NFE = n_steps."""
    x = sde.prior_sample(n, generator=generator)
    ts = time_grid(sde, n_steps, eps)
    return euler_maruyama(x, reverse_sde_drift(sde, score_fn), sde.g, ts,
                          generator=generator, noise_last=False)


def sample_pf_ode(sde, score_fn, n, n_steps, eps=1e-3, generator=None):
    """Эйлер для probability flow ODE от T до eps. NFE = n_steps."""
    x = sde.prior_sample(n, generator=generator)
    ts = time_grid(sde, n_steps, eps)
    return euler_ode(x, pf_ode_drift(sde, score_fn), ts)


def _div_exact(F, x):
    """Точная дивергенция sum_i dF_i/dx_i через autograd (d проходов назад; в 2D — два)."""
    div = torch.zeros(x.shape[0])
    for i in range(x.shape[1]):
        div += torch.autograd.grad(F[:, i].sum(), x, retain_graph=True)[0][:, i]
    return div


def _div_hutchinson(F, x, v):
    """Оценка Скиллинга–Хатчинсона: div F ~ v^T (dF/dx) v, E[v v^T] = I. Один проход назад."""
    vjp = torch.autograd.grad((F * v).sum(), x, retain_graph=True)[0]
    return (vjp * v).sum(1)


def pf_ode_loglik(sde, score_fn, x0, n_steps=1000, eps=1e-3, method="exact", generator=None):
    """log p_eps(x0) через probability flow ODE (Song et al., приложение D.2).

    Интегрируем ODE dx = F(x, t) dt вперёд от eps до T и одновременно
        d/dt log p_t(x(t)) = -div F(x(t), t),
    откуда  log p_eps(x(eps)) = log p_T(x(T)) + int_eps^T div F dt.

    method = "exact"        — точная дивергенция через autograd;
             "hutchinson"   — оценка Хатчинсона с одним вектором Радемахера на точку,
                              фиксированным вдоль всей траектории (так делается на практике).
    Шаг интегрирования — метод Эйлера, F и div F вычисляются в одной и той же точке.
    """
    ts = time_grid(sde, n_steps, eps, reverse=False)
    x = x0.clone()
    integral = torch.zeros(x.shape[0])
    if method == "hutchinson":
        v = torch.randint(0, 2, x.shape, generator=generator).float() * 2 - 1
    for k in range(n_steps):
        t, dt = ts[k], ts[k + 1] - ts[k]
        with torch.enable_grad():
            xr = x.detach().requires_grad_(True)
            s = score_fn(xr, t)
            F_score = -0.5 * sde.g(t) ** 2 * s          # часть дрейфа, зависящая от score
            if method == "exact":
                div = _div_exact(F_score, xr)
            else:
                div = _div_hutchinson(F_score, xr, v)
        div = div.detach() + sde.div_f(x, t)              # div f считается аналитически
        F = (sde.f(x, t) + F_score).detach()
        integral += div * dt
        x = x + F * dt
    return sde.prior_logp(x) + integral
