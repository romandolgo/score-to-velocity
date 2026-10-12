"""
P2 poster figure: sample quality by NFE for the reverse SDE and the probability
flow ODE, on the VE and VP SDEs, with the score network and with the exact oracle.

Top row: sliced W1 to the data. Bottom row: total variation between the mode
shares and the true weights. Points are means over the seeds, whiskers are one
standard deviation, the gray band is the metric floor (mean +- std between
independent samples of the data).

The numbers are read from results/ and computed by p2_experiments.py if missing.

Run from this directory:  python p2_quality_vs_nfe.py
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from p2_experiments import HERE, NFES, SDES, load_or_run, metric_floor, sweep

BLUE, RED = "#4A68DA", "#CA3142"
FONT_SIZE, TITLE_SIZE = 20, 30
NETWORK, ORACLE = "сеть", "оракул"

SERIES = {
    ("SDE", NETWORK): dict(
        name="SDE, network", color=RED, dash="solid", symbol="circle"
    ),
    ("SDE", ORACLE): dict(
        name="SDE, oracle", color=RED, dash="dash", symbol="circle-open"
    ),
    ("ODE", NETWORK): dict(
        name="ODE, network", color=BLUE, dash="solid", symbol="square"
    ),
    ("ODE", ORACLE): dict(
        name="ODE, oracle", color=BLUE, dash="dash", symbol="square-open"
    ),
}
METRICS = {
    "sw": ("Sliced W<sub>1</sub>", [0.02, 0.05, 0.1, 0.2, 0.5]),
    "tv": ("TV over mode shares", [0.01, 0.02, 0.05, 0.1, 0.2]),
}

pio.templates["compact"] = go.layout.Template(
    layout=dict(font=dict(size=FONT_SIZE), margin=dict(l=10, r=10, t=40, b=10))
)
pio.templates.default = "plotly_white+compact"

X_RANGE = [8, 1300]


def curve(values, name: str, color: str, dash: str, symbol: str, **kwargs):
    """Mean over the seeds with one-std whiskers; values has shape (NFE, seeds)."""
    return go.Scatter(
        x=NFES,
        y=values.mean(1),
        error_y=dict(array=values.std(1), thickness=1, width=0),
        mode="lines+markers",
        line=dict(color=color, dash=dash),
        marker=dict(symbol=symbol, size=8),
        name=name,
        legendgroup=name,
        **kwargs,
    )


def band(low: float, high: float, **kwargs):
    """Filled strip low..high across the whole panel; it also sets the x range."""
    left, right = X_RANGE
    return go.Scatter(
        x=[left, right, right, left],
        y=[low, low, high, high],
        fill="toself",
        mode="lines",
        line=dict(width=0),
        fillcolor="rgba(128, 128, 128, 0.2)",
        hoverinfo="skip",
        name="metric floor",
        legendrank=1001,
        **kwargs,
    )


def main() -> None:
    floor = load_or_run(HERE / "results/floor.pt", metric_floor)
    results = {
        (name, sampler): load_or_run(
            HERE / f"results/sweep_{name}_{sampler}.pt", sweep, name, sampler
        )
        for name in SDES
        for sampler in ("SDE", "ODE")
    }

    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        column_titles=[f"{name} SDE" for name in SDES],
        horizontal_spacing=0.03,
        vertical_spacing=0.04,
    )
    fig.update_annotations(font_size=TITLE_SIZE)
    for row, (metric, (title, ticks)) in enumerate(METRICS.items(), 1):
        low = floor[metric].mean() - floor[metric].std()
        high = floor[metric].mean() + floor[metric].std()
        for col, name in enumerate(SDES, 1):
            first = row == 1 and col == 1
            fig.add_trace(band(low, high, showlegend=first), row=row, col=col)
            for (sampler, source), style in SERIES.items():
                values = results[name, sampler][source][metric]
                fig.add_trace(
                    curve(values, showlegend=first, **style), row=row, col=col
                )
        fig.update_yaxes(type="log", tickvals=ticks, row=row)
        fig.update_yaxes(title_text=title, row=row, col=1)

    fig.update_xaxes(type="log", dtick=1)
    fig.update_xaxes(title_text="NFE", row=2)
    fig.update_layout(
        width=1000,
        height=760,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12),
    )

    out = HERE / "figures/p2_quality_vs_nfe.pdf"
    fig.write_image(out)
    print(f"Saved figures/{out.name}")


if __name__ == "__main__":
    main()
