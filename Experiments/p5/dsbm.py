"""
P5. Bridge matching and IMF (DSBM) from moons to eight Gaussians.

The reference process is dX = sqrt(eps) dW, so the bridge pinned at (x0, x1) is

    x_t = (1 - t) x0 + t x1 + sqrt(eps t (1 - t)) z,   z ~ N(0, I),

and the network regresses the bridge drift (x1 - x_t) / (1 - t). Samples come
from dX = v(X, t) dt + sqrt(eps) dW started at x0 ~ moons. With eps = 0 this is
the Rectified Flow of P3: straight interpolant, target x1 - x0, Euler ODE.

1. Bridge matching on the independent coupling for eps in {0, 0.1, 1}. At
   eps = 0 a model trained on the Rectified Flow batch of P3 with the same seed
   is compared with the bridge matching one.
2. IMF: refit on the coupling (x0, X_1) simulated by the previous model, which
   at eps = 0 is reflow. Only the forward drift is learned (DSBM alternates
   forward and backward fits so that neither marginal drifts). Per iteration:
   transport cost E||x0 - x1||^2, straightness, sliced Wasserstein, MMD and
   mode fractions against the target, and the fraction of samples outside
   every mode. The last one grows with the iterations: X_1 drifts away from
   the target, which also lowers the transport cost.
3. References on the same samples: exact discrete OT and Sinkhorn (POT). The
   static Schrodinger bridge of dX = sqrt(eps) dW is entropic OT with cost
   ||x - y||^2 and regularization 2 eps, the coupling IMF converges to.

Run from this directory:  python dsbm.py
"""

from __future__ import annotations

import math
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.append(str(Path(__file__).resolve().parents[1]))

from p0.datasets import eight_gaussians, sample_moons
from p0.integrators import integrate, time_grid
from p0.metrics import mmd2, mode_fractions, sliced_wasserstein, straightness
from p0.models import MLP

EPS_GRID = (0.0, 0.1, 1.0)
N_ITERS = 4
N_TRAIN, N_EVAL, N_OT = 20_000, 10_000, 5_000
TRAIN_STEPS, BATCH, LR = 5_000, 512, 1e-3
N_STEPS = 100
T_MAX = 1 - 1e-3
MODE_RADIUS = 1.2

Batch = Callable[[Tensor, Tensor, float, torch.Generator], tuple[Tensor, Tensor, Tensor]]


def bridge_batch(x0: Tensor, x1: Tensor, eps: float, gen: torch.Generator):
    """(t, x_t, target) for bridge matching.

    t stops at T_MAX because the target has variance eps t / (1 - t).
    """
    t = T_MAX * torch.rand(len(x0), 1, generator=gen)
    z = torch.randn(x0.shape, generator=gen)
    xt = (1 - t) * x0 + t * x1 + (eps * t * (1 - t)).sqrt() * z
    return t, xt, (x1 - xt) / (1 - t)


def rf_batch(x0: Tensor, x1: Tensor, eps: float, gen: torch.Generator):
    """The Rectified Flow batch of P3: the same x_t at eps = 0, target x1 - x0."""
    t, xt, _ = bridge_batch(x0, x1, 0.0, gen)
    return t, xt, x1 - x0


def fit(
    x0: Tensor,
    x1: Tensor,
    eps: float,
    seed: int,
    batch: Batch = bridge_batch,
    n_steps: int = TRAIN_STEPS,
) -> MLP:
    """Markovian projection of the coupling given by the rows of (x0, x1).

    The learning rate is annealed to zero: with a constant one the last iterate
    is noisy enough to double the sliced Wasserstein distance of the samples.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)
    for _ in range(n_steps):
        idx = torch.randint(0, len(x0), (BATCH,), generator=gen)
        t, xt, target = batch(x0[idx], x1[idx], eps, gen)
        loss = F.mse_loss(model(xt, t), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model.eval()


@torch.no_grad()
def sample(
    model: MLP,
    x0: Tensor,
    eps: float,
    gen: torch.Generator,
    return_traj: bool = False,
) -> Tensor:
    """Euler-Maruyama for dX = v dt + sqrt(eps) dW from t = 0 to 1; an ODE at eps = 0."""
    ts = time_grid(0.0, 1.0, N_STEPS)
    g = math.sqrt(eps) if eps > 0 else None
    return integrate(model, x0, ts, g=g, gen=gen, return_traj=return_traj)


def imf(x0: Tensor, x1: Tensor, eps: float, n_iters: int, seed: int) -> list[MLP]:
    """n_iters Markovian projections, the first one of the coupling (x0, x1).

    Each next model is fit on the bridges of the coupling (x0, X_1) simulated by
    the previous one. With eps = 0 the list is 1-RF, 2-RF, ...
    """
    gen = torch.Generator().manual_seed(seed)
    models = [fit(x0, x1, eps, seed=seed)]
    for k in range(1, n_iters):
        x1 = sample(models[-1], x0, eps, gen)
        models.append(fit(x0, x1, eps, seed=seed + k))
    return models


def transport_cost(x0: Tensor, x1: Tensor) -> float:
    return (x1 - x0).pow(2).sum(-1).mean().item()


def evaluate(model: MLP, x0: Tensor, x1: Tensor, eps: float, seed: int):
    """Metrics of one model on the evaluation samples, and its trajectories.

    The transport cost uses the first N_OT points, the ones OT is solved on.
    """
    gen = torch.Generator().manual_seed(seed)
    traj = sample(model, x0, eps, gen, return_traj=True)
    centers = eight_gaussians().means
    fractions = mode_fractions(traj[-1], centers)
    inside = mode_fractions(traj[-1], centers, radius=MODE_RADIUS).sum().item()
    row = {
        "cost": transport_cost(x0[:N_OT], traj[-1, :N_OT]),
        "straightness": straightness(model, traj, time_grid(0.0, 1.0, N_STEPS)),
        "sw": sliced_wasserstein(traj[-1], x1),
        "mmd2": mmd2(traj[-1], x1),
        "mode_err": (fractions - 1 / len(centers)).abs().max().item(),
        "outside": 1 - inside,
    }
    return row, traj


def ot_references(x0: Tensor, x1: Tensor) -> dict[float, float]:
    """Transport cost <C, P> of the exact OT plan (key 0) and of Sinkhorn plans.

    Sinkhorn runs with regularization 2 eps for every eps > 0 of EPS_GRID.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        import ot

    cost = torch.cdist(x0.double(), x1.double()).pow(2)
    a = torch.full((len(x0),), 1 / len(x0), dtype=torch.float64)
    b = torch.full((len(x1),), 1 / len(x1), dtype=torch.float64)
    refs = {0.0: (ot.emd(a, b, cost, numItermax=1_000_000) * cost).sum().item()}
    for eps in EPS_GRID:
        if eps > 0:
            plan = ot.sinkhorn(
                a, b, cost, reg=2 * eps, method="sinkhorn_log", numItermax=20_000, stopThr=1e-5
            )
            refs[eps] = (plan * cost).sum().item()
    return refs


def check_rectified_flow(x0: Tensor, x1: Tensor, z0: Tensor, bm: MLP, seed: int) -> float:
    """Relative distance between samples of the eps = 0 model and of P3's Rectified Flow."""
    rf = fit(x0, x1, 0.0, seed=seed, batch=rf_batch)
    gen = torch.Generator().manual_seed(seed)
    t, xt, target = bridge_batch(x0[:BATCH], x1[:BATCH], 0.0, gen)
    target_err = (target - (x1[:BATCH] - x0[:BATCH])).abs().max().item()
    x_bm, x_rf = sample(bm, z0, 0.0, gen), sample(rf, z0, 0.0, gen)
    rel = ((x_bm - x_rf).norm(dim=1).mean() / (x_rf - z0).norm(dim=1).mean()).item()
    print(f"  max |(x1 - x_t) / (1 - t) - (x1 - x0)| on a batch   {target_err:.1e}")
    print(f"  mean |X_1 - X_1^RF| / mean |X_1^RF - x0|             {rel:.1e}")
    print(f"  transport cost   bridge matching {transport_cost(z0, x_bm):.4f}"
          f"   rectified flow {transport_cost(z0, x_rf):.4f}")
    return rel


COLORS = {0.0: "#2a78d6", 0.1: "#eb6834", 1.0: "#1baf7a"}


def label(eps: float) -> str:
    return "reflow" if eps == 0 else rf"IMF, $\varepsilon = {eps:g}$"


def plot_metrics(rows, refs, baseline: float, out: str = "transport_cost.png") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iters = range(1, N_ITERS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    for eps in EPS_GRID:
        ax.plot(iters, [r["cost"] for r in rows[eps]], "o-", color=COLORS[eps], label=label(eps))
    ax.axhline(refs[0.0], color="k", lw=1, ls="--", label="exact OT")
    for eps in EPS_GRID:
        if eps > 0:
            ax.axhline(refs[eps], color=COLORS[eps], lw=1, ls=":",
                       label=rf"Sinkhorn, reg $= 2\varepsilon = {2 * eps:g}$")
    ax.set(ylabel=r"$E\|x_0 - x_1\|^2$", title="Transport cost")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for eps in EPS_GRID:
        ax.semilogy(iters, [r["straightness"] for r in rows[eps]], "o-", color=COLORS[eps],
                    label=label(eps))
    ax.set(title=r"Straightness $\int E\|(x_1 - x_0) - v(x_t, t)\|^2 dt$")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    for eps in EPS_GRID:
        ax.plot(iters, [r["outside"] for r in rows[eps]], "o-", color=COLORS[eps], label=label(eps))
    ax.axhline(baseline, color="gray", lw=0.8, ls=":", label="target data")
    ax.set(ylim=0, title=rf"Fraction of $X_1$ farther than {MODE_RADIUS:g} from every mode")
    ax.legend(frameon=False, fontsize=8)

    for a in axes:
        a.set(xlabel="iteration (1 = independent coupling)", xticks=list(iters))
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved {out}")


def plot_trajectories(trajs, n_show: int = 150, out: str = "bridge_matching.png") -> None:
    """Paths of every model: rows are eps, columns are IMF iterations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(EPS_GRID), N_ITERS, figsize=(3.4 * N_ITERS, 3.4 * len(EPS_GRID)))
    for i, eps in enumerate(EPS_GRID):
        for k, traj in enumerate(trajs[eps]):
            ax = axes[i, k]
            path = traj[:, :n_show]
            ax.plot(path[..., 0], path[..., 1], color="gray", lw=0.4, alpha=0.5)
            ax.scatter(path[0, :, 0], path[0, :, 1], s=5, color="#2a78d6", zorder=3, label="$x_0$")
            ax.scatter(path[-1, :, 0], path[-1, :, 1], s=5, color="#e34948", zorder=3, label="$X_1$")
            ax.set_title(rf"$\varepsilon = {eps:g}$, iteration {k + 1}")
            ax.set(xlim=(-5.5, 5.5), ylim=(-5.5, 5.5), xticks=[], yticks=[])
            ax.set_aspect("equal")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    torch.set_num_threads(2)
    gen = torch.Generator().manual_seed(0)
    target = eight_gaussians()

    x0, x1 = sample_moons(N_TRAIN, gen=gen), target.sample(N_TRAIN, gen=gen)
    z0, z1 = sample_moons(N_EVAL, gen=gen), target.sample(N_EVAL, gen=gen)
    baseline_sw = sliced_wasserstein(target.sample(N_EVAL, gen=gen), z1)
    baseline_mmd = mmd2(target.sample(N_EVAL, gen=gen), z1)
    baseline_out = 1 - mode_fractions(z1, target.means, radius=MODE_RADIUS).sum().item()

    refs = ot_references(z0[:N_OT], z1[:N_OT])
    print(f"Transport cost on {N_OT} evaluation points:")
    print(f"  independent coupling  {transport_cost(z0[:N_OT], z1[:N_OT]):.4f}")
    print(f"  exact OT              {refs[0.0]:.4f}")
    for eps in EPS_GRID:
        if eps > 0:
            print(f"  Sinkhorn, reg = {2 * eps:<5g} {refs[eps]:.4f}")

    rows, trajs, first = {}, {}, {}
    print(f"\n{N_ITERS} iterations, {TRAIN_STEPS} steps each:")
    print(f"  {'eps':>5}{'iter':>6}{'cost':>9}{'straight':>10}{'SW':>8}{'MMD^2':>10}{'mode err':>10}{'outside':>9}")
    for eps in EPS_GRID:
        models = imf(x0, x1, eps, N_ITERS, seed=0)
        first[eps] = models[0]
        results = [evaluate(m, z0, z1, eps, seed=1) for m in models]
        rows[eps] = [r for r, _ in results]
        trajs[eps] = [t for _, t in results]
        for k, r in enumerate(rows[eps], start=1):
            print(f"  {eps:5g}{k:6d}{r['cost']:9.4f}{r['straightness']:10.4f}"
                  f"{r['sw']:8.4f}{r['mmd2']:10.2e}{r['mode_err']:10.4f}{r['outside']:9.4f}")
    print(f"  data vs data: SW {baseline_sw:.4f}, MMD^2 {baseline_mmd:.2e};"
          f" outside, target data: {baseline_out:.4f}")

    print("\nBridge matching at eps = 0 vs Rectified Flow (same seed):")
    rel = check_rectified_flow(x0, x1, z0, first[0.0], seed=0)
    print()

    plot_metrics(rows, refs, baseline_out)
    plot_trajectories(trajs)

    assert rel < 1e-2, f"eps = 0 does not reproduce Rectified Flow: {rel:.1e}"
    print("\nP5 check passed.")


if __name__ == "__main__":
    main()
