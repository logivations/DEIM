# exp2.5-draft-decode

## Що змінено в цьому експерименті

**`presize_res` — JPEG-декод одразу в зменшеному розмірі** (DCT-domain scaling,
`PIL Image.draft`). Гілка створена від exp2-ema-foreach спеціально для датасетів,
що читаються напряму з NFS без локальної копії: офлайн pre-resize там неможливий,
а draft-декод дає той самий ефект без жодного дотику до файлів на диску.

1. **`CocoDetection`** (`engine/data/dataset/coco_dataset.py`): новий параметр
   `presize_res` (за замовчуванням вимкнено — інші конфіги не зачеплені):
   - `_load_image` викликає `image.draft('RGB', (presize_res, presize_res))`
     перед декодом — libjpeg декодує в 1/2, 1/4 чи 1/8 розміру (найбільший
     масштаб, за якого обидві сторони ще ≥ presize_res). Це МЕНШЕ роботи в
     IDCT, тобто декод швидший за повний, а не повільніший. Для не-JPEG і
     малих зображень — no-op.
   - bbox та area анотацій масштабуються під фактичний розмір декоду
     (КОПІЇ словників — спільний pycocotools-індекс не мутується).
   - `orig_size` примусово лишається ОРИГІНАЛЬНИМ розміром з json — постпроцесор
     мапить нормалізовані передбачення саме через нього, а COCO GT евалюатора
     живе в оригінальних координатах (без цього mAP поїхав би вдвічі).

2. **`configs/label_studio/ls_coco_detection.yml`**: `presize_res: 768` для
   train і val датасетів. Для 5MP джерел (2592×1944) це декод у 1296×972 —
   запас ~1.9× над тренувальними 512 для якості аугментацій.

**A/B тест**: `tools/tests/test_presize_decode.py` (пройдено в контейнері):
- декод рівно 1/2 для 5MP-подібного JPEG;
- нормалізовані бокси збігаються з повнорозмірним шляхом точно, area ×0.25;
- `orig_size` лишається оригінальним в обох шляхах;
- повторні читання ідентичні (захист від мутації спільного COCO-індексу);
- повний ланцюг трансформів (Resize 512): бокси/лейбли збігаються.

```bash
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 \
  -v /data/DEIM_worktrees/exp2.5-draft-decode:/DEIM -w /DEIM \
  quay.io/logivations/ml_all:LS_dfine_latest \
  python3 tools/tests/test_presize_decode.py
```

## Очікуваний ефект

- Декод 5MP → 1.26MP: у ~3–4 рази дешевший на CPU воркера.
- Канвас RandomZoomOut: до 4× менший по площі (найдорожча CPU-аугментація).
- NFS I/O НЕ зменшується (по мережі їде повний JPEG) — виграш суто CPU-шний.
- Дивитись на `Profiling/time_data_load` і загальний wall time епохи; на
  слабких CPU-ранерах ефект найбільший.

## Нюанси порівняння з exp2

- СЕМАНТИЧНА зміна: аугментації бачать 1/2-зображення (еквівалент офлайн
  pre-resize до ~1024–1300px). Дрібні об'єкти (<~4px після зменшення) можуть
  губитись у SanitizeBoundingBoxes; mAP по area-бакетах може трохи зсунутись.
  Криві лосів НЕ порівнюються бітово з exp2.
- Тому це exp2.5 — окрема точка порівняння: exp2 vs exp2.5 показує чисту ціну
  draft-декоду і по швидкості, і по mAP.

## Унаслідовано від попередніх експериментів

- **exp0-baseline**: реверт RTDT-7618 (старий формат датасету); `profile_sync`;
  `stg1_epochs_perc`; `run_experiments.sh`; `tools/benchmark_decode.py`; `speedup.md`.
- **exp1-tf32-nan-gate**: TF32 + cudnn.benchmark (на Turing TF32 — no-op);
  NaN-перевірка за флагом `debug_nan` (дефолт off).
- **exp2-ema-foreach**: EMA через `torch._foreach_*` (+ `tools/tests/test_ema_foreach.py`).
- НЕ містить exp3 (persistent workers), exp4 (criterion) і exp5 (torchvision.io).

## Як запускати

```bash
./run_experiments.sh exp2.5-draft-decode
```

## На що дивитися в TensorBoard (vs exp2-ema-foreach)

- `Profiling/time_data_load` — головна метрика (особливо на епохах 1–4).
- Загальний wall time епохи в train.log.
- `Test/*` mAP — порівняти з exp2 на останній епосі: якщо просідання суттєве
  (більше за шум ~±0.02), draft-розмір можна підняти (`presize_res: 1024`
  дасть 1/2 лише для картинок, де обидві сторони ≥ 2048).
