"""
Checks of the mixture oracle and a picture of all datasets.

1. score == central finite-difference gradient of log_prob, for the clean and
   noised mixtures, with scalar and per-sample (alpha, sigma).
2. The noised mixture matches samples of alpha x + sigma eps: first two moments
   and the average log-likelihood (E log p_t should equal -entropy, not -inf).

Run from this directory:  python check_oracle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

sys.path.append(str(Path(__file__).resolve().parents[2]))

from common import datasets


def fd_grad(f, x: Tensor, h: float = 1e-5) -> Tensor:
    """Central finite differences of a scalar-per-sample function f: (B, d) -> (B,)."""
    grad = torch.empty_like(x)
    for i in range(x.shape[1]):
        e = torch.zeros_like(x)
        e[:, i] = h
        grad[:, i] = (f(x + e) - f(x - e)) / (2 * h)
    return grad


def check_score(gen: torch.Generator) -> float:
    worst = 0.0
    for name, make in datasets.MIXTURES.items():
        gm = make()
        x = 3 * torch.randn(2000, 2, generator=gen, dtype=torch.float64)
        x = x + gm.sample(2000, gen=gen, dtype=torch.float64)

        b = len(x)
        t = torch.rand(b, generator=gen, dtype=torch.float64)
        cases = {
            "clean": (1.0, 0.0),
            "scalar": (0.6, 0.8),
            "per-sample VP": ((1 - t**2).sqrt(), t),
            "per-sample linear": (t, 1 - t + 1e-2),
        }
        for case, (alpha, sigma) in cases.items():
            s = gm.score(x, alpha, sigma)
            s_fd = fd_grad(lambda y: gm.log_prob(y, alpha, sigma), x)
            err = ((s - s_fd).norm(dim=1) / (s_fd.norm(dim=1) + 1e-8)).max().item()
            worst = max(worst, err)
            print(f"  {name:<16}{case:<20}max rel err {err:.1e}")
    return worst


def check_noised(gen: torch.Generator) -> None:
    gm = datasets.two_modes()
    alpha, sigma = 0.5, 1.5
    x = gm.sample(400_000, gen=gen, dtype=torch.float64)
    xt = alpha * x + sigma * torch.randn(x.shape, generator=gen, dtype=torch.float64)
    ref = gm.noised(alpha, sigma).sample(400_000, gen=gen, dtype=torch.float64)

    print(f"  mean  empirical {xt.mean(0).numpy().round(3)}  mixture {ref.mean(0).numpy().round(3)}")
    print(f"  cov   empirical {xt.T.cov().flatten().numpy().round(3)}")
    print(f"        mixture   {ref.T.cov().flatten().numpy().round(3)}")
    ll_xt = gm.log_prob(xt, alpha, sigma).mean().item()
    ll_ref = gm.noised(alpha, sigma).log_prob(ref).mean().item()
    print(f"  E log p_t  on alpha x + sigma eps {ll_xt:.4f}   on mixture samples {ll_ref:.4f}")


def plot_datasets(gen: torch.Generator, out: str = "datasets.png") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = datasets.names()
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4))
    for ax, name in zip(axes, names):
        x = datasets.sample(name, 5000, gen=gen)
        ax.scatter(x[:, 0], x[:, 1], s=1, alpha=0.4)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")


def plot_noised_score(out: str = "oracle_two_modes.png") -> None:
    """log p_t and the score field of the 0.8/0.2 mixture at several noise levels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gm = datasets.two_modes()
    lim = 10
    g = torch.linspace(-lim, lim, 200, dtype=torch.float64)
    xx, yy = torch.meshgrid(g, g, indexing="xy")
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    q = torch.linspace(-lim, lim, 21, dtype=torch.float64)
    qx, qy = torch.meshgrid(q, q, indexing="xy")
    qgrid = torch.stack([qx.flatten(), qy.flatten()], dim=1)

    sigmas = [0.0, 1.0, 3.0, 8.0]
    fig, axes = plt.subplots(1, len(sigmas), figsize=(3.4 * len(sigmas), 3.6))
    for ax, s in zip(axes, sigmas):
        lp = gm.log_prob(grid, 1.0, s).reshape(xx.shape)
        ax.contourf(xx, yy, lp.clamp(min=lp.max() - 30), levels=30, cmap="viridis")
        sc = gm.score(qgrid, 1.0, s)
        sc = sc / (sc.norm(dim=1, keepdim=True) + 1e-8)
        ax.quiver(qgrid[:, 0], qgrid[:, 1], sc[:, 0], sc[:, 1], color="w", scale=30, width=0.004)
        ax.set_title(rf"$x + {s:g}\,\varepsilon$")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    gen = torch.Generator().manual_seed(0)

    print("Score vs finite-difference gradient of log p_t:")
    worst = check_score(gen)
    print("\nNoised mixture vs alpha x + sigma eps (two_modes, alpha = 0.5, sigma = 1.5):")
    check_noised(gen)
    print()
    plot_datasets(gen)
    plot_noised_score()

    assert worst < 1e-5, f"score does not match finite differences: {worst:.1e}"
    print("\nOracle check passed.")


if __name__ == "__main__":
    main()
