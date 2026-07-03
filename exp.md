# exp4-criterion-desync

## Що змінено в цьому експерименті

Прибрані приховані GPU→CPU синхронізації в лос-шляху. Criterion викликає
класифікаційний лос ~8 разів на крок (main + 2 aux + pre + enc + 3 dn), і кожен
виклик містив Python-цикли з `.item()`/`.any()` — сумарно сотні примусових
синхронізацій на крок, які серіалізували чергу CUDA (це головна причина, чому
стадія «Criterion» їла ~29% часу).

1. **`filter_suppress_source_targets`** (`engine/deim/utils.py`): маска
   suppress-source лейблів через `torch.isin` на GPU замість `.item()`-циклу
   по кожному лейблу кожного зображення. Семантика ідентична.

2. **`_get_ignore_tag_mask`** (`engine/deim/deim_criterion.py`): векторизований
   збір значень тегів замість `.item()` на кожне зображення; маска повертається
   безумовно (старий early-out `mask.any()` — це теж синк; порожня маска
   математично no-op нижче по коду).

3. **`_get_suppress_mask` + `_prepare_suppress_cache`**: таргет-залежні частини
   (xyxy source-бокси і рядки пригнічуваних класів через статичну LUT
   `[src_id, dst_class]`) рахуються тепер ОДИН раз на крок замість перерахунку
   в кожному з ~8 викликів; у per-layer частині прибрані `.item()`-цикл по
   лейблах і early-out `overlapping.any()`. Guard за shape зберігає стару
   поведінку для class-agnostic enc-гілки.

4. **`_get_go_indices`**: вибір найчастішої (row, col) пари векторизовано через
   `torch.unique` + стабільний argsort + `scatter_reduce(amin)` замість
   `.item()`-циклу по КОЖНІЙ унікальній парі. Нюанс: при точних нічиїх
   лічильників вибір колонки тепер детермінований (лексикографічний); старий
   код покладався на нестабільний argsort, тобто його вибір на нічиїх був
   довільний. У межах шуму тренування.

Батчинг matcher-а (один `.cpu()` на всі 5 cost-матриць) свідомо ВІДКЛАДЕНО —
робити лише якщо профайлінг цього рану досі покаже домінування matcher-стоянок.

**A/B тест**: `tools/tests/test_criterion_vectorized.py` — старі реалізації
збережені в тесті вербатим як референс:
- по-функційно: точна рівність на 20 випадкових конфігураціях кожна;
- `_get_go_indices`: рівність множин рядків + перевірка max-count кожного вибору
  + точна рівність на рядках без нічиїх;
- інтеграційно: повний `criterion.forward` (vfl+boxes, групи main/aux/pre/enc)
  старий vs новий шлях — збіг до atol 1e-6 (5 трейлів).

```bash
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 \
  -v /data/DEIM_worktrees/exp4-criterion-desync:/DEIM -w /DEIM \
  quay.io/logivations/ml_all:LS_dfine_latest \
  python3 tools/tests/test_criterion_vectorized.py
```

## Очікуваний ефект

Вікно «Criterion»: ~0.15s → ~0.05–0.08s на крок, плюс через десеріалізацію
CUDA-черги покращується перекриття criterion/backward.

## Нюанси порівняння з exp3

- На точних нічиїх у go-indices вибір пари може відрізнятись від старого коду
  (який сам був недетермінований) → криві boxes/local лосів можуть мікроскопічно
  розійтись; vfl не зачеплений.

## Унаслідовано від попередніх експериментів

- **exp0-baseline**: реверт RTDT-7618 (старий формат датасету); флаг `profile_sync`;
  override `stg1_epochs_perc`; `run_experiments.sh`; `tools/benchmark_decode.py`; `speedup.md`.
- **exp1-tf32-nan-gate**: TF32 + cudnn.benchmark (дефолт on); NaN-перевірка за
  флагом `debug_nan` (дефолт off).
- **exp2-ema-foreach**: EMA через `torch._foreach_*` (+ `tools/tests/test_ema_foreach.py`).
- **exp3-persistent-workers**: persistent_workers + prefetch_factor 4 + shared-memory
  епоха у воркери (+ `tools/tests/test_persistent_epoch.py`).

## Як запускати

```bash
./run_experiments.sh exp4-criterion-desync
```

## На що дивитися в TensorBoard (vs exp3-persistent-workers)

- `Profiling/time_criterion` — головна метрика цього експерименту, має впасти
  щонайменше вдвічі.
- `Profiling/time_backward` — може теж впасти (краще перекриття черги).
- `Loss/*` — мають іти впритул до exp3 (мікророзбіжності на нічиїх — норма).
