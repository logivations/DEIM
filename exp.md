# exp2-ema-foreach

## Що змінено в цьому експерименті

**EMA через foreach** (`engine/optim/ema.py::ModelEMA`):

Стара реалізація на кожному кроці ітерувала повний `state_dict` у Python-циклі і
робила два окремі CUDA-кернели на КОЖЕН тензор (`v *= d; v += (1-d)*m`) — сотні
дрібних кернелів на крок, ~0.06s (~11% часу кроку в старих логах, стадія
«Other (EMA/LR)»).

Нова реалізація:
- один раз кешує списки float-тензорів EMA та моделі (`_build_cache`; тензори
  `state_dict` шарять storage з живими параметрами, оптимізатор і
  `load_state_dict` оновлюють їх in-place, тому кеш валідний);
- на кроці — два fused-виклики на ВСІ тензори одразу:
  `torch._foreach_mul_(ema, d)` + `torch._foreach_add_(ema, model, alpha=1-d)`;
- кеш інвалідується при зміні об'єкта моделі, `.to()` та `load_state_dict`.

Математика та сама (`v = d·v + (1−d)·m`); fused-акумуляція може відрізнятись
в останніх бітах fp32 (заміряно: max abs diff ~1.5e-08 після 10 оновлень).
`decay_fn`/warmup/`start` не чіпались.

**A/B тест**: `tools/tests/test_ema_foreach.py` — порівнює нову реалізацію зі
старим циклом (залишений у тесті як референс) на 10 оновленнях маленької моделі
з conv/BN/linear (float-параметри, float- та int-бафери). Запуск у контейнері:

```bash
docker run --rm -v /data/DEIM_worktrees/exp2-ema-foreach:/DEIM -w /DEIM \
  quay.io/logivations/ml_all:LS_dfine_latest python3 tools/tests/test_ema_foreach.py
```

## Очікуваний ефект

Стадія «Other (EMA/LR)»: ~0.06s → <0.01s на крок (порядку −10% загального часу кроку).

## Нюанси порівняння з exp1

Практично жодних: результат EMA збігається зі старою реалізацією з точністю до
останніх бітів fp32 (тест пройдено в контейнері, max abs diff 1.49e-08).

## Унаслідовано від попередніх експериментів

- **exp0-baseline**: реверт RTDT-7618 (старий формат датасету); флаг `profile_sync`;
  override `stg1_epochs_perc`; `run_experiments.sh`; `tools/benchmark_decode.py`; `speedup.md`.
- **exp1-tf32-nan-gate**: TF32 + cudnn.benchmark (`allow_tf32`/`cudnn_benchmark`,
  дефолт on); NaN-перевірка за флагом `debug_nan` (дефолт off).

## Як запускати

```bash
./run_experiments.sh exp2-ema-foreach
```

## На що дивитися в TensorBoard (vs exp1-tf32-nan-gate)

- `Profiling/time_other` — має впасти з ~0.06s до <0.01s.
- `Profiling/time_total` на епохах 1–4.
- `Loss/*`, `Test/*` — мають збігатися з exp1 у межах шуму (EMA-математика ідентична).
