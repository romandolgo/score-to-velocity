"""Смесь гауссиан как оракул (временная версия; общий модуль ведёт пара A).

Если данные — смесь  sum_k w_k N(mu_k, s_k^2 I),  а  x_t = alpha * x_0 + std * z,
то p_t — тоже смесь:  sum_k w_k N(alpha mu_k, (alpha^2 s_k^2 + std^2) I).
Поэтому log p_t и score  grad log p_t  известны точно.
"""
import math
import torch


class GaussianMixture:
    def __init__(self, means, stds, weights):
        self.means = torch.as_tensor(means, dtype=torch.float32)          # (K, d)
        self.stds = torch.as_tensor(stds, dtype=torch.float32)            # (K,)
        w = torch.as_tensor(weights, dtype=torch.float32)
        self.weights = w / w.sum()                                        # (K,)
        self.K, self.d = self.means.shape

    def sample(self, n, generator=None):
        k = torch.multinomial(self.weights, n, replacement=True, generator=generator)
        z = torch.randn(n, self.d, generator=generator)
        return self.means[k] + self.stds[k, None] * z

    # ---- точные величины для зашумлённого распределения p_t ----
    def _params_t(self, alpha, std):
        alpha = torch.as_tensor(alpha, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        m = alpha * self.means                                            # (K, d)
        v = alpha ** 2 * self.stds ** 2 + std ** 2                        # (K,)
        return m, v

    def _log_terms(self, x, alpha, std):
        """log w_k + log N(x; m_k, v_k I) для всех k, форма (B, K)."""
        m, v = self._params_t(alpha, std)
        sq = ((x[:, None, :] - m[None]) ** 2).sum(-1)                    # (B, K)
        return (torch.log(self.weights)[None] - 0.5 * sq / v[None]
                - 0.5 * self.d * torch.log(2 * math.pi * v)[None])

    def log_prob_t(self, x, alpha=1.0, std=0.0):
        return torch.logsumexp(self._log_terms(x, alpha, std), dim=1)

    def score_t(self, x, alpha=1.0, std=0.0):
        """grad log p_t(x) = sum_k r_k(x) * (-(x - m_k) / v_k), r_k — апостериорные веса."""
        m, v = self._params_t(alpha, std)
        r = torch.softmax(self._log_terms(x, alpha, std), dim=1)          # (B, K)
        comp = -(x[:, None, :] - m[None]) / v[None, :, None]              # (B, K, d)
        return (r[..., None] * comp).sum(1)

    def nearest_mode(self, x):
        return torch.cdist(x, self.means).argmin(1)


def eight_gaussians(radius=3.0, std=0.25, weights=None):
    """8 гауссиан по кругу. По умолчанию веса неравные, чтобы проверять доли мод."""
    ang = torch.arange(8) * (2 * math.pi / 8)
    means = radius * torch.stack([torch.cos(ang), torch.sin(ang)], 1)
    if weights is None:
        weights = [0.25, 0.05, 0.15, 0.05, 0.2, 0.05, 0.2, 0.05]
    return GaussianMixture(means, [std] * 8, weights)
