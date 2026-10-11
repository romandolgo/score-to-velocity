# Пара B: интеграторы (П0), Score SDE (П2), семейство сэмплеров (П4)

## Состав
- `common/` — общий модуль проекта:
  - `integrators.py` — Эйлер для ODE, Эйлер–Маруяма для SDE (П0);
  - `sde.py` — VE- и VP-SDE (П2);
  - `samplers.py` — обратное SDE, probability flow ODE, log-likelihood (П2);
  - `paths.py` — путь Rectified Flow, пересчёт score <-> скорость, перенос VP-score на любой гауссов путь, семейство SDE с общими маргиналами (П4);
  - `oracle.py`, `nets.py`, `metrics.py` — временные версии; пары A и C заменят их своими.
- `p0_integrator_ou_test.ipynb` — тест интегратора на процессе Орнштейна–Уленбека.
- `p2_score_sde.ipynb` + `p2_experiments.py` — П2.
- `p4_one_model_many_samplers.ipynb` + `p4_experiments.py` — П4.
- `figures/` — все рисунки. Для постера: `p2_quality_vs_nfe.png`, `p2_loglik_scatter.png`, `p4_heatmaps.png`.

Ноутбуки берут готовые результаты из `checkpoints/` и `results/` и перезапускаются за 1–2 минуты.
Пересчитать с нуля — удалить эти папки (около 45 минут на CPU) или по частям: команды в начале `p2_experiments.py` и `p4_experiments.py`.

## П4: что подставить от пары C
Модель скорости в П4 — временная: обучена тем же лоссом, что в П3 (слайд 39).
Чтобы использовать модель из П3, замените тело `load_rf_model()` в `p4_experiments.py`
и удалите `results/p4_grid_rf*.pt` и `results/p4_field_error.pt`.

## Главный вывод П4
При малом NFE шум, затухающий к данным (g_t = g(1 - t), g ≈ 2), улучшает качество и оракула, и обученных моделей
(у score-модели из П2 в 4 раза сокращает ошибку модели), а постоянный g помогает только умеренным и ломает сэмплер, если он велик.
