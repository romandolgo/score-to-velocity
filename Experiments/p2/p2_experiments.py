"""П2 (пара B): все тяжёлые вычисления в одном месте.

Ноутбук p2_score_sde.ipynb вызывает эти функции через load_or_run(): если результат
уже лежит в results/ (или checkpoints/), он загружается за секунду, иначе считается.
Так ноутбук можно быстро перезапускать, а долгие куски — считать отдельно из терминала:

    python p2_experiments.py train VE          # ~3 мин
    python p2_experiments.py train VP
    python p2_experiments.py sweep VE SDE      # ~5 мин каждая
    python p2_experiments.py sweep VE ODE
    python p2_experiments.py sweep VP SDE
    python p2_experiments.py sweep VP ODE
    python p2_experiments.py loglik VE         # ~4 мин
    python p2_experiments.py loglik VP
    python p2_experiments.py hutch VP          # ~2 мин
    python p2_experiments.py prior VE          # ~1 мин
    python p2_experiments.py prior VP

Чтобы пересчитать что-то заново, удалите соответствующий файл в results/ или checkpoints/.
"""
import copy
import os
import sys
import time

import numpy as np
import torch

from common.metrics import mode_tv, sliced_wasserstein
from common.nets import ScoreModel
from common.oracle import eight_gaussians
from common.samplers import pf_ode_loglik, sample_pf_ode, sample_reverse_sde
from common.sde import VESDE, VPSDE, perturb

# ---------------- общие настройки эксперимента ----------------
EPS = 1e-3                                   # интегрируем до t = eps, а не до 0 (как в статье)
NFES = [10, 20, 50, 100, 200, 500, 1000]     # число шагов сэмплера = число вызовов сети
SEEDS = [0, 1, 2]
N_EVAL = 10_000                              # точек на один запуск сэмплера
N_TEST, LL_STEPS = 2000, 1000                # точки и шаги для log-likelihood
HIDDEN, LAYERS = 256, 4

mix = eight_gaussians()                      # 8 гауссиан по кругу, неравные веса
SDES = {'VE': VESDE(sigma_min=0.01, sigma_max=10.0), 'VP': VPSDE(beta_min=0.1, beta_max=20.0)}
SAMPLERS = {'SDE': sample_reverse_sde, 'ODE': sample_pf_ode}

os.makedirs('results', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('figures', exist_ok=True)


def _pack(obj):
    """numpy-массивы -> тензоры с меткой: такой файл читается через torch.load(weights_only=True)."""
    if isinstance(obj, np.ndarray):
        return {'__numpy__': torch.from_numpy(obj)}
    if isinstance(obj, dict):
        return {k: _pack(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_pack(v) for v in obj]
    return obj


def _unpack(obj):
    if isinstance(obj, dict):
        if set(obj) == {'__numpy__'}:
            return obj['__numpy__'].numpy()
        return {k: _unpack(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unpack(v) for v in obj]
    return obj


def load_or_run(path, fn, *args):
    """Загрузить результат из файла или посчитать и сохранить.

    weights_only=True: из файла читаются только тензоры и простые типы, без произвольного pickle.
    """
    if os.path.exists(path):
        return _unpack(torch.load(path, weights_only=True))
    res = fn(*args)
    torch.save(_pack(res), path)
    return res


def oracle_score(sde):
    """Точный score зашумлённой смеси: grad log p_t(x)."""
    return lambda x, t: mix.score_t(x, sde.alpha(t), sde.std(t))


def reference_sample():
    return mix.sample(N_EVAL, generator=torch.Generator().manual_seed(999))


# ---------------- 1. обучение ----------------
def train(name, steps=10_000, bs=1024, lr=2e-3, seed=0):
    """DSM с непрерывным t: || eps_theta(x_t, t) - z ||^2, EMA весов 0.999."""
    sde = SDES[name]
    torch.manual_seed(seed)
    model = ScoreModel(sde, hidden=HIDDEN, layers=LAYERS)
    ema = copy.deepcopy(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    losses, t0 = [], time.time()
    for _ in range(steps):
        x0 = mix.sample(bs)
        t = EPS + (sde.T - EPS) * torch.rand(bs)
        xt, z, _ = perturb(sde, x0, t)
        loss = ((model.eps(xt, t) - z) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.999).add_(p, alpha=0.001)
        losses.append(loss.item())
    print(f'{name}: {steps} шагов за {time.time() - t0:.0f} с, лосс {np.mean(losses[-500:]):.3f}')
    return {'state_dict': ema.state_dict(), 'losses': losses}


def load_model(name):
    ck = load_or_run(f'checkpoints/score_{name}.pt', train, name)
    model = ScoreModel(SDES[name], hidden=HIDDEN, layers=LAYERS)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck['losses']


# ---------------- 2. качество по NFE ----------------
def sweep(name, sampler_name):
    """Для сети и для оракула: SW1 и TV мод на каждом NFE, по всем сидам."""
    sde, sampler = SDES[name], SAMPLERS[sampler_name]
    model, _ = load_model(name)
    ref = reference_sample()
    out, t0 = {}, time.time()
    for src, fn in [('сеть', model.score), ('оракул', oracle_score(sde))]:
        sw = np.zeros((len(NFES), len(SEEDS)))
        tv = np.zeros_like(sw)
        for i, nfe in enumerate(NFES):
            for j, s in enumerate(SEEDS):
                x = sampler(sde, fn, N_EVAL, nfe, eps=EPS, generator=torch.Generator().manual_seed(s))
                sw[i, j] = sliced_wasserstein(x, ref)
                tv[i, j] = mode_tv(x, mix)
        out[src] = {'sw': sw, 'tv': tv}
    print(f'sweep {name} {sampler_name}: {time.time() - t0:.0f} с')
    return out


def metric_floor():
    """Пол метрик: расстояние между двумя независимыми выборками из самих данных."""
    ref = reference_sample()
    sw = [sliced_wasserstein(mix.sample(N_EVAL, generator=torch.Generator().manual_seed(500 + s)), ref)
          for s in range(10)]
    tv = [mode_tv(mix.sample(N_EVAL, generator=torch.Generator().manual_seed(600 + s)), mix) for s in range(10)]
    return {'sw': np.array(sw), 'tv': np.array(tv)}


# ---------------- 3. log-likelihood ----------------
def test_points(name):
    """Тестовые точки из p_eps — распределения, для которого считаем правдоподобие."""
    sde = SDES[name]
    x0 = mix.sample(N_TEST, generator=torch.Generator().manual_seed(7))
    xt, _, _ = perturb(sde, x0, torch.full((N_TEST,), EPS), generator=torch.Generator().manual_seed(8))
    return xt


def loglik(name):
    sde = SDES[name]
    model, _ = load_model(name)
    x = test_points(name)
    te = torch.tensor(EPS)
    t0 = time.time()
    res = {
        'x': x,
        'true': mix.log_prob_t(x, sde.alpha(te), sde.std(te)),
        'model': pf_ode_loglik(sde, model.score, x, LL_STEPS, EPS, method='exact'),
        'model_h': pf_ode_loglik(sde, model.score, x, LL_STEPS, EPS, method='hutchinson',
                                 generator=torch.Generator().manual_seed(9)),
        'oracle': pf_ode_loglik(sde, oracle_score(sde), x, LL_STEPS, EPS, method='exact'),
    }
    print(f'loglik {name}: {time.time() - t0:.0f} с')
    return res


def oracle_loglik_error(name, steps_list=(1000, 4000)):
    """Откуда ошибка log p при точном score: схема Эйлера или приор вместо настоящего p_T.

    Для каждого числа шагов — MAE до истинного log p_eps с приором SDE и с точным p_T смеси.
    Ошибка схемы падает с числом шагов, ошибка приора — нет.
    """
    sde = SDES[name]
    x = test_points(name)
    te, tT = torch.tensor(EPS), torch.tensor(sde.T)
    true = mix.log_prob_t(x, sde.alpha(te), sde.std(te))
    exact_pT = lambda xT: mix.log_prob_t(xT, sde.alpha(tT), sde.std(tT))
    out = {'steps': list(steps_list), 'prior': [], 'exact_pT': []}
    for steps in steps_list:
        for key, prior_logp in [('prior', None), ('exact_pT', exact_pT)]:
            est = pf_ode_loglik(sde, oracle_score(sde), x, steps, EPS, prior_logp=prior_logp)
            out[key].append((est - true).abs().mean().item())
    return out


def hutchinson_vs_k(name, n_pts=500, steps=400, Ks=(1, 4, 16, 64)):
    """Разброс оценки Хатчинсона при усреднении по K векторам Радемахера."""
    sde = SDES[name]
    model, _ = load_model(name)
    x = test_points(name)[:n_pts]
    exact = pf_ode_loglik(sde, model.score, x, steps, EPS, method='exact')
    out = {'K': list(Ks), 'diff': []}
    for K in Ks:
        est = pf_ode_loglik(sde, model.score, x.repeat(K, 1), steps, EPS, method='hutchinson',
                            generator=torch.Generator().manual_seed(K)).reshape(K, -1).mean(0)
        out['diff'].append((est - exact).numpy())
    return out


if __name__ == '__main__':
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    stage, name = sys.argv[1], sys.argv[2]
    if stage == 'train':
        load_model(name)
    elif stage == 'sweep':
        load_or_run(f'results/sweep_{name}_{sys.argv[3]}.pt', sweep, name, sys.argv[3])
    elif stage == 'loglik':
        load_or_run(f'results/loglik_{name}.pt', loglik, name)
    elif stage == 'hutch':
        load_or_run(f'results/hutch_{name}.pt', hutchinson_vs_k, name)
    elif stage == 'prior':
        load_or_run(f'results/oracle_prior_{name}.pt', oracle_loglik_error, name)
