"""Метрики качества сэмплов (временная версия; общий модуль ведёт пара C)."""
import torch


def sliced_wasserstein(a, b, p=1, n_proj=512, seed=0):
    """Sliced Wasserstein-p: средний по случайным направлениям W_p между проекциями.

    В 1D W_p между выборками равного размера считается через отсортированные значения.
    Направления фиксированы через seed, чтобы метрика была сравнима между запусками.
    p=1 устойчивее к случайным колебаниям долей мод в конечной выборке, чем p=2.
    """
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    gen = torch.Generator().manual_seed(seed)
    dirs = torch.randn(a.shape[1], n_proj, generator=gen)
    dirs = dirs / dirs.norm(dim=0, keepdim=True)
    pa, _ = torch.sort(a @ dirs, dim=0)
    pb, _ = torch.sort(b @ dirs, dim=0)
    return (((pa - pb).abs() ** p).mean() ** (1 / p)).item()


def mmd_rbf(a, b, bandwidths=(0.1, 0.5, 1.0, 2.0)):
    """Квадрат MMD с суммой RBF-ядер (смещённая оценка)."""
    def k(x, y):
        d2 = torch.cdist(x, y) ** 2
        return sum(torch.exp(-d2 / (2 * h ** 2)) for h in bandwidths)
    return (k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean()).item()


def mode_fractions(x, mixture):
    """Доли сэмплов у каждой моды (по ближайшему центру)."""
    idx = mixture.nearest_mode(x)
    return torch.bincount(idx, minlength=mixture.K).float() / len(x)


def mode_tv(x, mixture):
    """Расстояние полной вариации между долями мод и истинными весами."""
    return 0.5 * (mode_fractions(x, mixture) - mixture.weights).abs().sum().item()
