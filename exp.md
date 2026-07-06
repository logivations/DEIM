# exp5-decode-backend

## Що змінено в цьому експерименті

**JPEG-декодування через torchvision.io замість PIL** — за вердиктом бенчмарку
на сервері (`tools/benchmark_decode.py`, 200 реальних 5MP зображень):
torchvision.io = **1.60× проти PIL** на чистому декоді (OpenCV/PyTurboJPEG в
образі відсутні); end-to-end з трансформами ≈ 1.37× (канвас RandomZoomOut домінує).

1. **`CocoDetection`** (`engine/data/dataset/coco_dataset.py`): новий параметр
   `decode_backend: pil | torchvision` (дефолт `pil` — інші конфіги не зачеплені).
   З `torchvision`: `_load_image` декодує напряму в uint8 CHW `tv_tensors.Image`
   (libjpeg-turbo, без PIL-обгортки); `ConvertCocoPolysToMask` і `load_item`
   вміють брати розміри і з тензора, і з PIL.

2. **`ConvertPILImage`** (`engine/data/transforms/_transforms.py`): приймає
   тепер і `tv_tensors.Image` (dtype/scale без `pil_to_tensor`). Свідомо НЕ
   доданий голий `torch.Tensor` у `_transformed_types` — `BoundingBoxes`/`Mask`
   теж Tensor-сабкласи і були б помилково відскейлені.

3. **`configs/label_studio/ls_coco_detection.yml`**: `decode_backend: torchvision`
   у train і val датасетах.

**A/B тест**: `tools/tests/test_decode_backend.py` (пройдено в контейнері):
- піксельна рівність декодерів (max diff ≤ 2 з 255 — той самий libjpeg-turbo);
- `ConvertPILImage` дає ідентичний float-вихід для обох типів входу, бокси не чіпає;
- повний пайплайн `CocoDetection` (load + Resize + Convert) — рівність зображень,
  боксів і лейблів між бекендами.

```bash
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 \
  -v /data/DEIM_worktrees/exp5-decode-backend:/DEIM -w /DEIM \
  quay.io/logivations/ml_all:LS_dfine_latest \
  python3 tools/tests/test_decode_backend.py
```

## Очікуваний ефект

Декод — це воркерний CPU-час, у здорових ранах прихований за GPU (Data Load
2–3%). Реальна цінність: запас міцності проти storage/CPU contention (як у
старих ранах 252/253, що втрачали 8–15 год) + швидший перший батч + швидша
валідація. На чистому `Profiling/time_data_load` очікуй помірне падіння.

## Нюанси порівняння з exp4

- Піксельні відмінності декодування ≤ 2/255 (різні шляхи одного libjpeg-turbo) +
  мікровідмінності антиаліасингу Resize на тензорі vs PIL → криві лосів не
  бітово-ідентичні exp4, у межах шуму.
- Формат PNG теж підтримується (`decode_image`), а EXIF-орієнтація ігнорується
  обома бекендами однаково — паритет збережено.

## Унаслідовано від попередніх експериментів

- **exp0-baseline**: реверт RTDT-7618 (старий формат датасету); `profile_sync`;
  `stg1_epochs_perc`; `run_experiments.sh`; `tools/benchmark_decode.py`; `speedup.md`.
- **exp1-tf32-nan-gate**: TF32 + cudnn.benchmark; `debug_nan` gate.
  ⚠ На Quadro RTX 5000 (Turing) TF32 — no-op, виграш exp1 там лише від
  cudnn.benchmark + NaN-gate.
- **exp2-ema-foreach**: EMA через `torch._foreach_*`.
- **exp3-persistent-workers**: persistent workers + shared-memory епоха.
- **exp4-criterion-desync**: векторизований criterion без `.item()`/`.any()` синків.

## Як запускати

```bash
./run_experiments.sh exp5-decode-backend
```

## На що дивитися в TensorBoard (vs exp4-criterion-desync)

- `Profiling/time_data_load` — має впасти (особливо перший батч епохи).
- `Profiling/time_total` — головний підсумок УСЬОГО стеку проти exp0-baseline.
- `Loss/*`, `Test/*` — в межах шуму від exp4.
