"""Shared code of the project (task P0), imported by every experiment.

datasets.py     Gaussian mixtures with exact log p_t and score, moons, checkerboard
integrators.py  Euler and Euler-Maruyama behind one interface
metrics.py      sliced Wasserstein, MMD, mode fractions, straightness
models.py       MLP with a time embedding
sde.py          VE and VP SDEs
samplers.py     reverse SDE, probability flow ODE, log-likelihood
paths.py        Gaussian paths, score <-> velocity, SDEs sharing the marginals

nets.py and oracle.py are the network and the isotropic mixture that the
checkpoints and results of P2 and P4 were produced with.
"""
