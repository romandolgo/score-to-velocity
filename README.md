# score-to-velocity
From score to velocity: NCSN, Score SDE, Rectified Flow and DSBM reproduced on 2D data. One learned vector field, many samplers, with the Fokker–Planck equation as the common thread.

## Layout

```
common/            shared code (P0), imported by every experiment
Experiments/
  p0/              checks of the shared code: mixture oracle, integrators on the OU process
  p2/              Score SDE: VE and VP, reverse SDE vs probability flow ODE, log-likelihood
  p3/              Rectified Flow and reflow
  p4/              one model, a family of samplers: quality vs (g, NFE)
  p5/              bridge matching and IMF (DSBM) on 2D data
Theory/            notes for the theory tasks
```

`common/` holds the datasets and the mixture oracle, the integrators, the metrics, the MLP, the VE/VP SDEs with their samplers, and the score <-> velocity conversion. Every script and notebook adds the repository root to `sys.path` itself, so nothing has to be installed; run each one from its own folder.

Needed packages: torch, numpy, matplotlib, scikit-learn, pandas, POT, plotly, kaleido, jupyter. All experiments fix their random seeds and run on a CPU.

## Reproducing the figures

| Task | Run | Output |
|---|---|---|
| P0 | `python check_oracle.py`, `python check_ou.py` | `datasets.png`, `oracle_two_modes.png`, `ou_check.png` |
| P2 | `p2_score_sde.ipynb`, `python p2_quality_vs_nfe.py` | `figures/p2_quality_vs_nfe.png`, `figures/p2_loglik_scatter.png`, `figures/p2_quality_vs_nfe.pdf` |
| P3 | `reflow_try.ipynb` | trajectories before and after reflow, quality by NFE |
| P4 | `p4_one_model_many_samplers.ipynb`, `python p4_quality_heatmaps.py` | `figures/p4_heatmaps.png` and the other `figures/p4_*.png`, `figures/p4_heatmap_*.pdf` |
| P5 | `python dsbm.py`, `moons_to_swiss_roll.ipynb` | `figures/transport_cost.png`, `figures/bridge_matching.png`, `figures/*.pdf` |

P2 and P4 keep their trained models in `checkpoints/` and their computed tables in `results/`, so the notebooks rerun in a minute or two. To recompute from scratch, delete those folders (about 30 minutes on a CPU for P2, 15 more for P4) or run the stages listed at the top of `p2_experiments.py` and `p4_experiments.py`. P4 reads the VP score model from `Experiments/p2/checkpoints/`.

P5 writes its figures to `Experiments/p5/figures/`, which is not tracked.
