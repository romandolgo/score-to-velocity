"""
Poster figures from the output of dsbm.py.

    python figures.py runs/eps1 [runs/eps0 ...]

For every run directory:
    trajectories.png   samples at t = 0 ... 1, forward and backward
    iterations.png     the same inputs transported after each IMF iteration
    samples.png        several outputs per input (the SDE is stochastic for eps > 0)
and, in the first directory, cost.png: transport cost by IMF iteration, one line
per run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAD = 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

plt.rcParams.update(
    {
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": "#c3c2b7",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def tile(images: np.ndarray) -> np.ndarray:
    """(R, C, H, W) uint8 -> one image with PAD background pixels between cells."""
    r, c, h, w = images.shape
    canvas = np.zeros((r * (h + PAD) - PAD, c * (w + PAD) - PAD), dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            y, x = i * (h + PAD), j * (w + PAD)
            canvas[y : y + h, x : x + w] = images[i, j]
    return canvas


def show_grid(ax, images: np.ndarray, col_labels: list[str], title: str, cmap: str):
    w = images.shape[-1]
    ax.imshow(tile(images), cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([j * (w + PAD) + (w - 1) / 2 for j in range(len(col_labels))])
    ax.set_xticklabels(col_labels)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_title(title, pad=10)


def pair_figure(panels, path: Path, cmap: str) -> None:
    """Two image grids side by side; panels = [(images, col_labels, title), ...]."""
    rows = panels[0][0].shape[0]
    cols = [p[0].shape[1] for p in panels]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(0.62 * sum(cols) + 1, 0.62 * rows + 1),
        gridspec_kw={"width_ratios": cols},
    )
    for ax, (images, labels, title) in zip(axes, panels):
        show_grid(ax, images, labels, title, cmap)
    fig.savefig(path)
    plt.close(fig)
    print(path)


def run_figures(run: Path, rows: int, cmap: str) -> None:
    config = json.loads((run / "metrics.json").read_text())["config"]
    src, dst, eps = config["source"], config["target"], config["eps"]
    names = {"fwd": f"{src} → {dst}", "bwd": f"{dst} → {src}"}
    with np.load(run / "samples.npz") as f:
        data = {k: f[k] for k in f}
    tag = f"ε = {eps:g}"

    panels = []
    for key in ("fwd", "bwd"):
        traj = data[f"{key}_traj"][-1][:, :rows]
        frames = np.linspace(0, len(traj) - 1, 9).round().astype(int)
        times = frames / (len(traj) - 1)
        labels = [f"{t if key == 'fwd' else 1 - t:.2g}" for t in times]
        labels[0] = f"t = {labels[0]}"
        panels.append((traj[frames].swapaxes(0, 1), labels, f"{names[key]}, {tag}"))
    pair_figure(panels, run / "trajectories.png", cmap)

    panels = []
    for key in ("fwd", "bwd"):
        traj = data[f"{key}_traj"][:, :, :rows]
        images = np.concatenate([traj[:1, 0], traj[:, -1]]).swapaxes(0, 1)
        labels = ["input", "BM"] + [f"IMF {k}" for k in range(1, len(traj))]
        panels.append((images, labels, f"{names[key]}, {tag}"))
    pair_figure(panels, run / "iterations.png", cmap)

    panels = []
    for key in ("fwd", "bwd"):
        inputs = data[f"{key}_traj"][-1][:1, :rows]
        images = np.concatenate([inputs, data[f"{key}_multi"][-1][:, :rows]])
        labels = ["input"] + [""] * (len(images) - 1)
        panels.append((images.swapaxes(0, 1), labels, f"{names[key]}, {tag}"))
    pair_figure(panels, run / "samples.png", cmap)


def cost_figure(runs: list[Path], path: Path) -> None:
    """Transport cost E|x0 - x1|^2 / d of the learned coupling at every stage."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, key in zip(axes, ("fwd", "bwd")):
        for color, run in zip(SERIES, runs):
            m = json.loads((run / "metrics.json").read_text())
            stages = [s["stage"] for s in m["stages"]]
            cost = [s[f"cost_{key}"] for s in m["stages"]]
            src, dst = m["config"]["source"], m["config"]["target"]
            ax.plot(
                stages,
                cost,
                color=color,
                lw=2,
                marker="o",
                ms=8,
                mec="white",
                mew=2,
                label=f"ε = {m['config']['eps']:g}",
            )
        direction = f"{src} → {dst}" if key == "fwd" else f"{dst} → {src}"
        ax.set_title(direction, loc="left", color=INK)
        ax.set_xticks(stages)
        ax.set_xticklabels(["BM"] + [f"IMF {k}" for k in stages[1:]])
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel(r"$\mathbb{E}\,\|x_0 - x_1\|^2 / d$")
    if len(runs) > 1:
        axes[1].legend(frameon=False, labelcolor=INK)
    fig.text(
        0.01,
        -0.02,
        f"independent coupling: {m['cost_independent']:.2f}",
        color=MUTED,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--rows", type=int, default=10, help="inputs per grid")
    p.add_argument("--dark", action="store_true", help="white ink on black")
    args = p.parse_args()

    for run in args.runs:
        run_figures(run, args.rows, "gray" if args.dark else "gray_r")
    cost_figure(args.runs, args.runs[0] / "cost.png")


if __name__ == "__main__":
    main()
