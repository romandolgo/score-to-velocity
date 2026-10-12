"""VE- и VP-SDE из статьи Song et al. (2021), раздел 3.4 (П2, пара B).

Для обоих SDE переходное распределение гауссово:
    x_t | x_0 ~ N(alpha(t) * x_0, std(t)^2 I),
поэтому x_t = alpha(t) * x_0 + std(t) * z строится одной формулой.
"""
import math
import torch


class VPSDE:
    """dx = -1/2 beta(t) x dt + sqrt(beta(t)) dW,  beta(t) = b_min + t (b_max - b_min).

    Непрерывный DDPM. alpha(t) = exp(-B(t)/2), std(t)^2 = 1 - exp(-B(t)),
    где B(t) = int_0^t beta(s) ds. Prior: N(0, I).
    """
    name = "VP"

    def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0):
        self.beta_min, self.beta_max, self.T = beta_min, beta_max, T

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def B(self, t):
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2

    def f(self, x, t):
        return -0.5 * self.beta(t) * x

    def div_f(self, x, t):
        """Дивергенция дрейфа f: нужна для правдоподобия (d log p / dt = -div F)."""
        return torch.full((x.shape[0],), -0.5 * float(self.beta(t)) * x.shape[1],
                          dtype=x.dtype)

    def g(self, t):
        return torch.sqrt(torch.as_tensor(self.beta(t)))

    def alpha(self, t):
        return torch.exp(-0.5 * torch.as_tensor(self.B(t)))

    def std(self, t):
        return torch.sqrt(1.0 - torch.exp(-torch.as_tensor(self.B(t))))

    def prior_sample(self, n, d=2, generator=None):
        return torch.randn(n, d, generator=generator)

    def prior_logp(self, x):
        d = x.shape[1]
        return -0.5 * (x ** 2).sum(1) - 0.5 * d * math.log(2 * math.pi)


class VESDE:
    """dx = sqrt(d[sigma^2(t)]/dt) dW,  sigma(t) = s_min (s_max / s_min)^t.

    Непрерывный NCSN/SMLD. alpha(t) = 1, std(t) = sigma(t).
    g(t)^2 = d sigma^2 / dt = 2 sigma(t)^2 ln(s_max / s_min).
    Prior: N(0, s_max^2 I).

    Как в коде Song et al., ядро берётся N(x_0, sigma(t)^2 I), а не
    N(x_0, (sigma(t)^2 - s_min^2) I): разница s_min^2 = 1e-4 пренебрежимо мала.
    """
    name = "VE"

    def __init__(self, sigma_min=0.01, sigma_max=10.0, T=1.0):
        self.sigma_min, self.sigma_max, self.T = sigma_min, sigma_max, T
        self.log_ratio = math.log(sigma_max / sigma_min)

    def sigma(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** torch.as_tensor(t)

    def f(self, x, t):
        return torch.zeros_like(x)

    def div_f(self, x, t):
        return torch.zeros(x.shape[0], dtype=x.dtype)

    def g(self, t):
        return self.sigma(t) * math.sqrt(2 * self.log_ratio)

    def alpha(self, t):
        return torch.ones_like(torch.as_tensor(t, dtype=torch.float32))

    def std(self, t):
        return self.sigma(t)

    def prior_sample(self, n, d=2, generator=None):
        return self.sigma_max * torch.randn(n, d, generator=generator)

    def prior_logp(self, x):
        d = x.shape[1]
        s2 = self.sigma_max ** 2
        return -0.5 * (x ** 2).sum(1) / s2 - 0.5 * d * math.log(2 * math.pi * s2)


def perturb(sde, x0, t, generator=None):
    """x_t = alpha(t) x_0 + std(t) z. Возвращает (x_t, z, std(t)) с формами (B,d),(B,d),(B,1)."""
    t = t.reshape(-1, 1)
    z = torch.randn(x0.shape, generator=generator)
    a, s = sde.alpha(t), sde.std(t)
    return a * x0 + s * z, z, s
