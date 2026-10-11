"""П4 (пары B и C): одна модель — семейство сэмплеров.

Все сэмплеры работают на одном пути Rectified Flow: x_t = t x1 + (1 - t) x0, t от шума (0) к данным (1).
Семейство:  dx = [v_t(x) + 1/2 g^2 grad log p_t(x)] dt + g dW,  g = 0 — это ODE.
Маргиналы p_t одинаковы при любом g (Т4, п. 4), отличаются только траектории и ошибки.

Три поля (v, s):
  'оракул'          — точные v и s смеси гауссиан;
  'скорость (RF)'   — сеть, обученная как Rectified Flow (лосс слайда 39); s = (t v - x)/(1 - t).
                      Временная версия: пара C заменит её моделью из П3 (load_rf_model);
  'score (VP, П2)'  — score-сеть из П2, перенесённая на путь RF; v = x/t + (1 - t)/t * s.

Запуск по частям из терминала (результаты кешируются в results/):
    python p4_experiments.py train_rf          # ~3 мин
    python p4_experiments.py grid oracle       # ~1 мин
    python p4_experiments.py grid rf           # ~3 мин
    python p4_experiments.py grid vp           # ~3 мин
    python p4_experiments.py grid oracle sigma # то же для g_t = g (1 - t)
    python p4_experiments.py grid rf sigma
    python p4_experiments.py grid vp sigma
    python p4_experiments.py field_error       # ~10 с
"""
import copy
import sys
import time

import numpy as np
import torch

from common.metrics import mode_tv, sliced_wasserstein
from common.nets import MLP
from common.paths import (RFPath, field_from_score, field_from_velocity, oracle_score,
                          sample_family, vp_score_on_path)
from p2_experiments import SDES, load_model, load_or_run, mix, reference_sample

GS = [0.0, 0.5, 1.0, 2.0]
NFES4 = [5, 10, 20, 50, 100]
SEEDS4 = [0, 1, 2]
N4 = 10_000
T0, T1 = 0.01, 0.99
path = RFPath()


# ---------------- модель скорости (временная замена П3) ----------------
def train_rf(steps=10_000, bs=1024, lr=2e-3, seed=0):
    """Rectified Flow, слайд 39: || v_theta(x_t, t) - (x1 - x0) ||^2,  x_t = t x1 + (1 - t) x0."""
    torch.manual_seed(seed)
    net = MLP(hidden=256, layers=4)
    ema = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    losses, t0 = [], time.time()
    for _ in range(steps):
        x1 = mix.sample(bs)
        x0 = torch.randn(bs, 2)
        t = torch.rand(bs)
        xt = t[:, None] * x1 + (1 - t[:, None]) * x0
        loss = ((net(xt, t) - (x1 - x0)) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), net.parameters()):
                pe.mul_(0.999).add_(p, alpha=0.001)
        losses.append(loss.item())
    print(f'RF: {steps} шагов за {time.time() - t0:.0f} с, лосс {np.mean(losses[-500:]):.3f}')
    return {'state_dict': ema.state_dict(), 'losses': losses}


def load_rf_model():
    """Модель скорости. Когда будет готова модель пары C из П3 — загрузить её здесь."""
    ck = load_or_run('checkpoints/velocity_RF.pt', train_rf)
    net = MLP(hidden=256, layers=4)
    net.load_state_dict(ck['state_dict'])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return lambda x, t: net(x, torch.as_tensor(t, dtype=x.dtype)), ck['losses']


def get_field(name):
    if name == 'oracle':
        return field_from_score(path, oracle_score(mix, path))
    if name == 'rf':
        v, _ = load_rf_model()
        return field_from_velocity(path, v)
    if name == 'vp':
        model, _ = load_model('VP')
        return field_from_score(path, vp_score_on_path(SDES['VP'], model.score, path))
    raise ValueError(name)


FIELD_NAMES = {'oracle': 'оракул', 'rf': 'скорость (RF)', 'vp': 'score (VP, П2)'}


# ---------------- сетка (g, NFE) ----------------
def grid(name, schedule='const'):
    field = get_field(name)
    ref = reference_sample()
    sw = np.zeros((len(GS), len(NFES4), len(SEEDS4)))
    tv = np.zeros_like(sw)
    t0 = time.time()
    for i, g in enumerate(GS):
        for j, nfe in enumerate(NFES4):
            for k, s in enumerate(SEEDS4):
                x = sample_family(field, g, N4, nfe, T0, T1, generator=torch.Generator().manual_seed(s),
                                  schedule=schedule)
                bad = ~torch.isfinite(x).all(1)
                x = x[~bad]
                sw[i, j, k] = sliced_wasserstein(x, ref) if len(x) > 100 else np.nan
                tv[i, j, k] = mode_tv(x, mix) if len(x) > 100 else np.nan
    print(f'grid {name} {schedule}: {time.time() - t0:.0f} с')
    return {'sw': sw, 'tv': tv}


# ---------------- ошибка полей по t ----------------
def field_error(n=4000, n_t=40):
    """Относительная ошибка v и s обеих сетей против оракула вдоль t."""
    ts = torch.linspace(T0, T1, n_t)
    oracle = get_field('oracle')
    fields = {k: get_field(k) for k in ['rf', 'vp']}
    out = {'t': ts.numpy()}
    for k, f in fields.items():
        ev, es = [], []
        for t in ts:
            x1 = mix.sample(n, generator=torch.Generator().manual_seed(1))
            x0 = torch.randn(n, 2, generator=torch.Generator().manual_seed(2))
            x = t * x1 + (1 - t) * x0
            v0, s0 = oracle(x, t)
            v, s = f(x, t)
            ev.append((((v - v0) ** 2).sum(1).mean() / (v0 ** 2).sum(1).mean()).item())
            es.append((((s - s0) ** 2).sum(1).mean() / (s0 ** 2).sum(1).mean()).item())
        out[k] = {'v': np.array(ev), 's': np.array(es)}
    return out


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'train_rf':
        load_rf_model()
    elif stage == 'grid':
        sched = sys.argv[3] if len(sys.argv) > 3 else 'const'
        suffix = '' if sched == 'const' else f'_{sched}'
        load_or_run(f'results/p4_grid_{sys.argv[2]}{suffix}.pt', grid, sys.argv[2], sched)
    elif stage == 'field_error':
        load_or_run('results/p4_field_error.pt', field_error)
