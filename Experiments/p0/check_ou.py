"""
Integrator test on the Ornstein-Uhlenbeck process dx = -theta x dt + s dW.

Exact solution from a point x0:
    E[X_t]   = x0 e^{-theta t}
    Var[X_t] = s^2 / (2 theta) (1 - e^{-2 theta t})
Stationary distribution (Fokker-Planck, T1): N(0, s^2 / (2 theta)).

1. Empirical mean and variance of Euler-Maruyama paths vs exact, over time.
2. Histogram at large t vs the stationary density.
3. Weak error |E phi(X_T) - E phi(X_T^exact)| vs dt on a log-log plot, slope ~ 1.
   phi(x) = x and phi(x) = x^2; with these test functions the Monte Carlo noise
   is far below the discretization error. Same for the Euler ODE (g = 0).

Run from this directory:  python check_ou.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from common.integrators import euler, euler_maruyama, time_grid

THETA, S, X0, T = 1.0, 1.0, 3.0, 1.0


def drift(x, t):
    return -THETA * x


def exact_moments(t):
    mean = X0 * np.exp(-THETA * t)
    var = S**2 / (2 * THETA) * (1 - np.exp(-2 * THETA * t))
    return mean, var


def moments_over_time(gen, n=100_000, n_steps=200, t_end=5.0):
    x0 = torch.full((n, 1), X0, dtype=torch.float64)
    ts = time_grid(0.0, t_end, n_steps, dtype=torch.float64)
    traj = euler_maruyama(drift, S, x0, ts, gen=gen, return_traj=True)[..., 0]
    return ts.numpy(), traj.mean(1).numpy(), traj.var(1).numpy(), traj[-1].numpy()


def weak_errors(gen, dts, n=2_000_000):
    """Errors in E[X_T] and E[X_T^2] for EM, and in X_T for the Euler ODE."""
    mean_ex, var_ex = exact_moments(T)
    m2_ex = mean_ex**2 + var_ex
    x0 = torch.full((n, 1), X0, dtype=torch.float64)
    rows = []
    for dt in dts:
        ts = time_grid(0.0, T, round(T / dt), dtype=torch.float64)
        x = euler_maruyama(drift, S, x0, ts, gen=gen)
        err_mean = abs(x.mean().item() - mean_ex)
        err_m2 = abs(x.pow(2).mean().item() - m2_ex)
        x_ode = euler(drift, x0[:1], ts).item()
        err_ode = abs(x_ode - mean_ex)
        rows.append((dt, err_mean, err_m2, err_ode))
    # Monte Carlo standard errors of the two estimates, for reference
    se_mean = math.sqrt(var_ex / n)
    se_m2 = math.sqrt((4 * mean_ex**2 * var_ex + 2 * var_ex**2) / n)
    return np.array(rows), se_mean, se_m2


def slope(dts, errs):
    return np.polyfit(np.log(dts), np.log(errs), 1)[0]


def plot(ts, mean, var, x_last, rows, se, out="ou_check.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_ex, var_ex = exact_moments(ts)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(ts, mean, label="mean, EM")
    ax.plot(ts, mean_ex, "k--", lw=1, label="mean, exact")
    ax.plot(ts, var, label="var, EM")
    ax.plot(ts, var_ex, "k:", lw=1, label="var, exact")
    ax.set(xlabel="$t$", title=rf"OU, $x_0 = {X0:g}$, $\theta = {THETA:g}$, $s = {S:g}$")
    ax.legend(frameon=False)

    ax = axes[1]
    v_inf = S**2 / (2 * THETA)
    grid = np.linspace(-4, 4, 400) * math.sqrt(v_inf)
    ax.hist(x_last, bins=120, density=True, alpha=0.5, label=f"EM, $t = {ts[-1]:g}$")
    ax.plot(grid, np.exp(-(grid**2) / (2 * v_inf)) / math.sqrt(2 * math.pi * v_inf),
            "k", lw=1.2, label=r"$\mathcal{N}(0, s^2 / 2\theta)$")
    ax.set(xlabel="$x$", title="Stationary distribution")
    ax.legend(frameon=False)

    ax = axes[2]
    dts = rows[:, 0]
    ax.loglog(dts, rows[:, 1], "o-", label=f"EM, $E[X_T]$ (slope {slope(dts, rows[:, 1]):.2f})")
    ax.loglog(dts, rows[:, 2], "s-", label=f"EM, $E[X_T^2]$ (slope {slope(dts, rows[:, 2]):.2f})")
    ax.loglog(dts, rows[:, 3], "^-", label=f"Euler ODE (slope {slope(dts, rows[:, 3]):.2f})")
    ax.loglog(dts, rows[0, 2] * dts / dts[0], "k--", lw=1, label="slope 1")
    ax.axhline(se, color="gray", lw=0.8, ls=":", label="MC std. error")
    ax.set(xlabel=r"$\Delta t$", ylabel="weak error", title=f"Weak error at $T = {T:g}$")
    ax.legend(frameon=False, fontsize=8)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    gen = torch.Generator().manual_seed(0)

    ts, mean, var, x_last = moments_over_time(gen)
    mean_ex, var_ex = exact_moments(ts)
    print("Moments over time (dt = 0.025, 1e5 paths):")
    for t in (0.5, 1.0, 2.0, 5.0):
        k = int(np.argmin(np.abs(ts - t)))
        print(f"  t = {t:<4} mean {mean[k]:.4f} (exact {mean_ex[k]:.4f})"
              f"   var {var[k]:.4f} (exact {var_ex[k]:.4f})")
    dt = ts[1] - ts[0]
    print(f"  stationary var s^2/(2 theta) = {S**2 / (2 * THETA):.4f},"
          f" EM with this dt: {S**2 / (2 * THETA - THETA**2 * dt):.4f}")

    dts = np.array([2.0**-k for k in range(1, 8)])
    rows, se_mean, se_m2 = weak_errors(gen, dts)
    print("\nWeak error at T = 1 (2e6 paths):")
    print(f"  {'dt':>9}{'E[X]':>11}{'E[X^2]':>11}{'ODE':>11}")
    for dt, e1, e2, e3 in rows:
        print(f"  {dt:9.5f}{e1:11.2e}{e2:11.2e}{e3:11.2e}")
    print(f"  MC std. error: E[X] {se_mean:.1e}, E[X^2] {se_m2:.1e}")
    s1, s2, s3 = (slope(dts, rows[:, i]) for i in (1, 2, 3))
    print(f"  slopes: E[X] {s1:.2f}, E[X^2] {s2:.2f}, ODE {s3:.2f}")

    plot(ts, mean, var, x_last, rows, se_m2)

    # O(dt) bias: EM's own stationary variance is s^2 / (2 theta - theta^2 dt)
    assert abs(mean[-1] - mean_ex[-1]) < 0.03 and abs(var[-1] - var_ex[-1]) < 0.03
    assert all(0.85 < s < 1.15 for s in (s1, s2, s3)), "weak order is not 1"
    print("\nOU check passed.")


if __name__ == "__main__":
    main()
