"""
P4 poster figures: sample quality over (sigma, NFE) for the SDE family

    dx = [v + 1/2 g_t^2 grad log p_t] dt + g_t dW,

one heatmap per field (exact oracle, learned velocity, learned score) and per
noise schedule (constant g_t = sigma, and g_t = sigma (1 - t) decaying towards
the data). sigma is the noise scale, called g in p4_experiments.py.

A cell is the sliced W1 to the data, averaged over the seeds; the color scale is
logarithmic and shared by all heatmaps. The frame marks the best sigma at every
NFE.

The numbers are read from results/ and computed by p4_experiments.py if missing.

Run from this directory:  python p4_quality_heatmaps.py
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from p4_experiments import GS, HERE, NFES4, grid, load_or_run

BLUE, RED = "#4A68DA", "#CA3142"
TITLE_SIZE = 25
SW_RANGE = (0.02, 1.5)

FIELDS = {
    "oracle": "Exact oracle",
    "rf": "Learned velocity",
    "vp": "Learned score",
}
SCHEDULES = {
    "const": ("", "g<sub>t</sub> = σ"),
    "sigma": ("_sigma", "g<sub>t</sub> = σ (1 − t)"),
}

pio.templates["compact"] = go.layout.Template(
    layout=dict(font=dict(size=16), margin=dict(l=10, r=10, t=50, b=10))
)
pio.templates.default = "plotly_white+compact"


def heatmap(sw: np.ndarray, title: str) -> go.Figure:
    """Heatmap of sw, rows sigma and columns NFE, the best sigma of each NFE framed."""
    fig = go.Figure(
        go.Heatmap(
            z=np.log10(sw),
            x=[str(nfe) for nfe in NFES4],
            y=[f"{g:g}" for g in GS],
            text=[[f"{value:.3f}" for value in row] for row in sw],
            texttemplate="%{text}",
            textfont=dict(size=18, color="white"),
            colorscale=[BLUE, RED],
            zmin=np.log10(SW_RANGE[0]),
            zmax=np.log10(SW_RANGE[1]),
            xgap=3,
            ygap=3,
            showscale=False,
        )
    )
    for col, row in enumerate(sw.argmin(axis=0)):
        fig.add_shape(
            type="rect",
            x0=col - 0.5,
            x1=col + 0.5,
            y0=row - 0.5,
            y1=row + 0.5,
            line=dict(color="black", width=3),
        )
    fig.update_xaxes(title_text="NFE", type="category", showgrid=False)
    fig.update_yaxes(
        title_text="σ", type="category", autorange="reversed", showgrid=False
    )
    fig.update_layout(
        title=dict(text=title, font_size=TITLE_SIZE, x=0.5, xanchor="center"),
        width=620,
        height=380,
    )
    return fig


def main() -> None:
    for field, field_title in FIELDS.items():
        for schedule, (suffix, schedule_title) in SCHEDULES.items():
            path = HERE / f"results/p4_grid_{field}{suffix}.pt"
            sw = np.nanmean(load_or_run(path, grid, field, schedule)["sw"], axis=2)
            fig = heatmap(sw, f"{field_title}, {schedule_title}")
            out = HERE / f"figures/p4_heatmap_{field}{suffix}.pdf"
            fig.write_image(out)
            print(f"Saved figures/{out.name}")


if __name__ == "__main__":
    main()
