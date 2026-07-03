# exp3-persistent-workers

## Що змінено в цьому експерименті

1. **`persistent_workers: True` + `prefetch_factor: 4`**
   (`configs/label_studio/ls_dataloader.yml`, train і val лоадери):
   раніше всі 16 воркерів вбивались і створювались наново на КОЖНІЙ епосі —
   спостережувана затримка першого батча ~80s/епоху (форк процесів + повторне
   завантаження COCO-індексу + прогрів prefetch-черги). Тепер воркери живуть
   увесь ран.

2. **Патч `warp_loader`** (`engine/misc/dist_utils.py`): при DDP лоадер
   пересоздається з DistributedSampler — раніше він ГУБИВ `persistent_workers`
   і `prefetch_factor`. На одному GPU цей шлях не викликається, патч — страховка.

3. **Пропагація епохи в живі воркери** (`engine/data/dataset/_dataset.py`,
   `engine/data/dataloader.py::BaseCollateFunction`) — критично для коректності:
   політика вимкнення аугментацій (`stop_epoch_forward` читає `dataset.epoch`
   ВСЕРЕДИНІ воркера) і mixup-логіка collate залежать від епохи. З персистентними
   воркерами звичайний атрибут, виставлений у головному процесі, до них не
   доходить — RandomZoomOut/RandomIoUCrop ніколи б не вимкнулись на stop_epoch.
   Фікс: епоха зберігається в shared-memory тензорі
   (`torch.tensor([...]).share_memory_()`) — живі воркери бачать оновлення одразу.

**Тест**: `tools/tests/test_persistent_epoch.py` — піднімає DataLoader з
persistent_workers=True/num_workers=2 і перевіряє, що після кожного `set_epoch`
і датасет, і collate_fn всередині воркерів бачать нову епоху (пройдено в контейнері).

```bash
docker run --rm -v /data/DEIM_worktrees/exp3-persistent-workers:/DEIM -w /DEIM \
  --shm-size=2g quay.io/logivations/ml_all:LS_dfine_latest \
  python3 tools/tests/test_persistent_epoch.py
```

## Очікуваний ефект

- Мінус ~80s стартової затримки на КОЖНІЙ епосі (крім нульової) — на 6-епоховому
  рані це ~7 хв, на повних 72 епохах ~1.6 год.
- Рівніша подача даних (prefetch_factor 4 замість 2).

## Ціна / нюанси

- RAM: 32 живі воркери (16 train + 16 val) тримають COCO-індекс (fork → copy-on-write,
  переважно шариться) + prefetch-буфери. Якщо тісно — знизити `prefetch_factor` до 2.
- `persistent_workers=True` вимагає `num_workers > 0` (у конфізі 16 — ок).
- RNG воркерів тепер продовжується між епохами замість respawn-reseed → потоки
  аугментацій розходяться з exp0–2 починаючи з епохи 1+ (у межах шуму тренування;
  порядок семплів незмінний — його задає sampler у головному процесі).

## Унаслідовано від попередніх експериментів

- **exp0-baseline**: реверт RTDT-7618 (старий формат датасету); флаг `profile_sync`;
  override `stg1_epochs_perc`; `run_experiments.sh`; `tools/benchmark_decode.py`; `speedup.md`.
- **exp1-tf32-nan-gate**: TF32 + cudnn.benchmark (дефолт on); NaN-перевірка за
  флагом `debug_nan` (дефолт off).
- **exp2-ema-foreach**: EMA через `torch._foreach_*` з кешем тензорів
  (+ `tools/tests/test_ema_foreach.py`).

## Як запускати

```bash
./run_experiments.sh exp3-persistent-workers
```

## На що дивитися в TensorBoard (vs exp2-ema-foreach)

- `Profiling/time_data_load` на епохах 1–5 — перший батч епохи більше не коштує
  ~80s, середнє data_load має помітно впасти саме на непершій епосі.
- Загальний wall time епох 1–5 у train.log.
- У train.log близько stop_epoch (епоха 5 при EPOCHS=6): рядок
  «Multi-scale Training until ...» і поведінка аугментацій — політика має
  перемкнутись (це і перевіряє shared-epoch фікс).
