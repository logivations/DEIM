# exp1-tf32-nan-gate

## Що змінено в цьому експерименті

1. **TF32 + cudnn.benchmark** (`engine/solver/_solver.py::_setup`, флаги
   `allow_tf32: True`, `cudnn_benchmark: True` в `configs/runtime.yml`):
   - TF32 вмикає швидкий fp32 matmul/conv на Ampere+ (RTX 30xx/A100+). Criterion
     рахується з вимкненим autocast (чистий fp32) — TF32 прискорює його напряму,
     а також усі fp32-шматки forward/backward.
   - `cudnn.benchmark = True` — автотюнінг conv-алгоритмів на кожну форму входу.
     Мультискейл до stop_epoch використовує ~8 фіксованих розмірів, кеш прогрівається
     на перших кроках кожного розміру.
   - Вимкнути можна через `-u allow_tf32=False cudnn_benchmark=False`.

2. **NaN-перевірка за флагом** (`engine/solver/det_engine.py`, флаг `debug_nan: False`):
   стара перевірка `torch.isnan(pred_boxes).any()` виконувалась КОЖЕН крок і
   примусово синхронізувала GPU у вікні, яке не покривав жоден таймер (прихована
   втрата ~0.02–0.05s/крок). FP16-overflow фікс (clamp + LQE guard) давно залитий,
   тому перевірка тепер лише для дебагу: `-u debug_nan=True`. Страховка від
   дивергенції лишилась: `math.isfinite(loss)` зупиняє тренування як і раніше,
   criterion робить `nan_to_num` для лосів.

## Очікуваний ефект

- TF32/cudnn: −5–15% на fwd/bwd/criterion математиці.
- NaN-gate: −0.02–0.05s/крок (прибраний прихований sync).

## Нюанси порівняння з exp0

- TF32 змінює округлення fp32 → криві лосів не бітово-ідентичні exp0 (у межах шуму).
- Перша епоха повільніша через прогрів cudnn.benchmark — епоху 0 виключати з
  порівняння (як і завжди).

## Унаслідовано від попередніх експериментів

- **exp0-baseline**: реверт RTDT-7618 (старий формат датасету: /dataset/train,
  /dataset/test, /dataset/coco_annotations.json); флаг `profile_sync` (правдивий
  профайлінг, оркестратор вмикає завжди); override `stg1_epochs_perc`;
  `run_experiments.sh`; `tools/benchmark_decode.py`; `speedup.md`.

## Як запускати

```bash
./run_experiments.sh exp1-tf32-nan-gate
```

## На що дивитися в TensorBoard (vs exp0-baseline)

- `Profiling/time_forward`, `time_backward`, `time_criterion` — мають впасти (TF32).
- `Profiling/time_total` на епохах 1–4 — головна метрика.
- `Loss/*` — криві мають іти поруч з exp0 (невеликий дрейф від TF32 — норма).
