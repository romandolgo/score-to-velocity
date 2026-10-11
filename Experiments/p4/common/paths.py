"""Гауссовы пути, пересчёт score <-> скорость и семейство SDE с общими маргиналами (П4).

Соглашение П4 (как в Т4 и на слайде 39): время t идёт от шума к данным,
    x_t = alpha_t * x1 + sigma_t * x0,   x0 ~ N(0, I) — шум,   x1 ~ p_data — данные.
Для Rectified Flow alpha_t = t, sigma_t = 1 - t.

Связь скорости и score (Т4, п. 3):
    v_t(x) = (da/a) x + sigma (da sigma / a - dsigma) * grad log p_t(x),
    grad log p_t(x) = (v_t(x) - (da/a) x) / (sigma (da sigma / a - dsigma)).
Для RF: v = x/t + (1-t)/t * s,   s = (t v - x)/(1 - t).

Семейство SDE с теми же маргиналами, что ODE dx = v dt (Т4, п. 4):
    dx = [v_t(x) + 1/2 g^2 grad log p_t(x)] dt + g dW.
"""
import math
import torch

from .integrators import euler_maruyama


class RFPath:
    """Прямой путь Rectified Flow: alpha = t, sigma = 1 - t."""
    name = "RF"

    def alpha(self, t):
        return t

    def sigma(self, t):
        return 1 - t

    def dalpha(self, t):
        return torch.ones_like(torch.as_tensor(t, dtype=torch.float32))

    def dsigma(self, t):
        return -torch.ones_like(torch.as_tensor(t, dtype=torch.float32))


def _coefs(path, t):
    """Коэффициенты формулы связи: v = A x + C s."""
    t = torch.as_tensor(t, dtype=torch.float32)
    a, s, da, ds = path.alpha(t), path.sigma(t), path.dalpha(t), path.dsigma(t)
    A = da / a
    C = s * (da * s / a - ds)
    return A, C


def field_from_score(path, score_fn):
    """Поле (v, s) из score-модели: один вызов модели на точку и момент."""
    def field(x, t):
        A, C = _coefs(path, t)
        s = score_fn(x, t)
        return A * x + C * s, s
    return field


def field_from_velocity(path, v_fn):
    """Поле (v, s) из модели скорости: один вызов модели на точку и момент."""
    def field(x, t):
        A, C = _coefs(path, t)
        v = v_fn(x, t)
        return v, (v - A * x) / C
    return field


def oracle_score(mix, path):
    """Точный score: p_t — смесь с центрами alpha*mu и дисперсиями alpha^2 s^2 + sigma^2."""
    return lambda x, t: mix.score_t(x, path.alpha(torch.as_tensor(t)), path.sigma(torch.as_tensor(t)))


def vp_score_on_path(vp_sde, vp_score_fn, path):
    """Перенести score-модель, обученную на VP-SDE (П2), на любой гауссов путь.

    Все гауссовы пути отличаются только масштабом и «шумностью» lambda = sigma/alpha:
        путь:  x = alpha_t (x1 + lambda z),     VP:  x' = a(tau) (x1 + lambda z),  a = 1/sqrt(1 + lambda^2).
    Значит x' = (a / alpha_t) x и  grad log p_t(x) = (a / alpha_t) * score_VP(x', tau),
    где tau находится из  B(tau) = ln(1 + lambda^2)  (у VP  sigma/a = sqrt(e^B - 1)).
    """
    bmin, bmax = vp_sde.beta_min, vp_sde.beta_max

    def tau_of(lam):
        B = torch.log1p(lam ** 2)
        return (-bmin + torch.sqrt(bmin ** 2 + 2 * (bmax - bmin) * B)) / (bmax - bmin)

    def s(x, t):
        t = torch.as_tensor(t, dtype=torch.float32)
        al, sg = path.alpha(t), path.sigma(t)
        lam = sg / al
        tau = tau_of(lam)
        scale = (1.0 / torch.sqrt(1 + lam ** 2)) / al
        return scale * vp_score_fn(scale * x, tau)

    return s


def sample_family(field, g, n, nfe, t0=0.01, t1=0.99, generator=None, d=2, schedule='const'):
    """Эйлер–Маруяма для dx = [v + 1/2 g_t^2 s] dt + g_t dW от t0 до t1 (g = 0 — ODE, метод Эйлера).

    schedule='const': g_t = g (как в задании П4);
    schedule='sigma': g_t = g * sigma_t = g (1 - t) — шум затухает у данных.
    Маргиналы p_t одинаковы при любом g_t (Т4, п. 4), так что это тоже член семейства.

    field(x, t) -> (v, s). Старт: x ~ N(0, (1 - t0)^2 I) — это p_{t0} с точностью до вклада
    данных порядка t0. В конце делим на t1 (= alpha_{t1}), чтобы вернуть масштаб данных;
    оставшееся размытие sigma_{t1} = 0.01 пренебрежимо. NFE = nfe.
    """
    gfun = (lambda t: g) if schedule == 'const' else (lambda t: g * (1 - float(t)))
    x = (1 - t0) * torch.randn(n, d, generator=generator)
    ts = torch.linspace(t0, t1, nfe + 1)

    def drift(x, t):
        v, s = field(x, t)
        return v + 0.5 * gfun(t) ** 2 * s

    x = euler_maruyama(x, drift, gfun, ts, generator=generator, noise_last=False)
    return x / t1
