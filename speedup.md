# Прискорення тренування DEIM — фінальний план

## Контекст

Тренування (135 тис. зображень ~5MP, 512×512, batch 32, 72 епохи) займає ~38 год у найкращому випадку та до ~53 год у найгіршому. Часу на численні експерименти немає, тому замість них проведено аналіз prof.txt та коду. Висновки:

- **Теза «data loading займає 50% часу» — хибна.** У здорових ранах (251, 254) Data Load — це 2–3% часу кроку. Рани 252/253 втратили 8–15 год через сплески Data Load (19–22%) при *ідентичному коді* — це contention на storage/CPU вузла, а не проблема пайплайна. «Bulk» — це зовнішня підготовка датасету (Label Studio), а не код у репозиторії. «Top-K Guard» коштує ~нічого (поелементні `torch.where`/`clamp`; ран 254 з guard ≈ 251 без нього).
- **Профайлер бреше**: таймери в `engine/solver/det_engine.py` — звичайний `time.time()` без жодного `torch.cuda.synchronize()`. CUDA асинхронна → GPU-час «зливається» на прихованих точках синхронізації і приписується не тим стадіям. Відомі точки синхронізації: незамірюваний `torch.isnan(pred_boxes).any()` на det_engine.py:177 (кожен крок, залишок дебагу), `.cpu()` у matcher (5×/крок), ~250 `.item()`/`.any()` синків усередині criterion, `math.isfinite(loss)` на :254.
- **Реальні витрати на крок (0.53s загалом)**: вікно Criterion ~29% (CPU-синки + fp32 математика лосу), Backward ~23%, Forward ~16%, EMA ~11% (Python-цикл по всьому state_dict, сотні дрібних кернелів на крок), Transfer+GPU-аугментації ~10%.
- **Фіксовані витрати на епоху поза профайлером (~7 год розриву між профайлером і wall time)**: 16 воркерів даталоадера перезапускаються щоепохи (persistent_workers=False → ~80s затримка першого батча × 72), валідація щоепохи, збереження чекпоінтів.
- AMP увімкнений, EMA увімкнена (base/optimizer.yml перекриває runtime.yml за порядком include). TF32/cudnn.benchmark ніде не вмикаються.

## Рішення користувача

- Скоуп: повний пакет змін у коді. **Один GPU** (патч warp_loader додаємо як дворядкову страховку).
- NaN-перевірка → за флагом, за замовчуванням вимкнена.
- **Без змін, що впливають на семантику тренування**: val_freq (пропуск валідацій) та офлайн pre-resize датасету — ВІДХИЛЕНО користувачем. Викинуто з плану.

## Вердикти щодо початково запропонованих оптимізацій

1. Офлайн pre-resize датасету — технічно розумно як *страховка від нестабільності* на кшталт ранів 252/253 (зменшує декодування ~5×, канвас RandomZoomOut ~16×; потребує перезапису COCO json: bbox, area, width/height), але НЕ прискорює здорові рани. **Викинуто за рішенням користувача (без семантичних змін).**
2. Налаштування даталоадера — persistent_workers: **прийнято** (з фіксом пропагації епохи, див. пункт A). prefetch_factor 4: прийнято. num_workers 24→16: нічого не змінює, у конфізі вже 16. pin_memory: нічого не змінює, вже увімкнено та передається.
3. accimage/turbojpeg/FFCV/GPU-декодування — **відхилено**: PIL і так використовує libjpeg-turbo; nvJPEG конфліктує з PIL-базованими CPU-аугментаціями; FFCV — переписування всього пайплайна заради стадії, що займає 2–3%.

## Імплементація (за пріоритетом (вплив × безпечність)/зусилля)

### H. Правдивий профайлінг (флаг `profile_sync`) — робити першим
Файли: `engine/solver/det_engine.py`, `engine/core/_config.py`, `engine/solver/det_solver.py`, `configs/runtime.yml`.
- `_config.py __init__`: `self.profile_sync: bool = False` (нові top-level yaml-флаги читаються як `args.<attr>` лише якщо попередньо оголошені в BaseConfig — стосується всіх нових флагів нижче). `runtime.yml`: `profile_sync: False`.
- det_solver передає його в `train_one_epoch`; там хелпер `_t()` = опціональний `torch.cuda.synchronize()` + `time.time()`, що замінює кожне зчитування таймера в циклі (обидві гілки — AMP і non-AMP).
- За замовчуванням вимкнено (нуль overhead); увімкнено → точна атрибуція стадій для замірів кожного пункту нижче.

### B. TF32 + cudnn.benchmark
Файли: `engine/solver/_solver.py` (`BaseSolver._setup`, перед model.to(device)), `_config.py`, `runtime.yml`.
```python
if torch.cuda.is_available():
    if getattr(cfg/args, 'allow_tf32', True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    if getattr(cfg/args, 'cudnn_benchmark', True):
        torch.backends.cudnn.benchmark = True
```
Флаги `allow_tf32: True`, `cudnn_benchmark: True`. Criterion рахується у fp32 (autocast вимкнено на det_engine.py:186) → TF32 прискорює математику лосу напряму. Мультискейл = ~8 розмірів входу до stop_epoch → кеш benchmark прогрівається на перших кроках для кожного розміру, це нормально. Очікувано: 5–15% на fp32 conv/matmul шляхах.

### E. NaN-перевірка за флагом
Файли: `engine/solver/det_engine.py:177-182`, `_config.py`, `det_solver.py`, `runtime.yml`.
- `debug_nan: False` за замовчуванням; `if debug_nan and (torch.isnan(...).any() or torch.isinf(...).any()):`. Повторне увімкнення через `-u debug_nan=True`.
- Страховка лишається: criterion робить `nan_to_num` для лосів (deim_criterion.py:543), а `math.isfinite(loss_value)` на det_engine.py:254 зупиняє тренування при дивергенції.
- Прибирає жорсткий per-step GPU-синк у незамірюваному вікні; оцінка 0.02–0.05s/крок.

### C. EMA через foreach
Файл: `engine/optim/ema.py` (`ModelEMA.update`).
- Один раз зібрати кешовані списки тензорів (float-елементи state_dict EMA та моделі, рівність ключів перевіряється assert), далі на кожному кроці:
  `d = self.decay_fn(self.updates); torch._foreach_mul_(ema_tensors, d); torch._foreach_add_(ema_tensors, model_tensors, alpha=1-d)`.
- Кеш ключується `id(model)`; інвалідація в `load_state_dict` (nn.Module копіює in-place, але інвалідація — безкоштовна страховка). Логіку `start`/`before_start` та `decay_fn` не чіпати.
- **Знайдений наявний баг (НЕ виправляти мовчки)**: `det_solver.py:108/203` присвоюють `self.ema.decay`, але `decay_fn` — lambda, що захопила локальну змінну `decay` конструктора → «Refresh EMA at stop_epoch» зі зміною decay ніколи не діяв. Поведінку зберегти точно; винести як окрему проблему коректності.
- Очікувано: ~0.06s → <0.005s/крок (~10% wall time).
- Перевірка: allclose A/B старого й нового update() ×10 на маленькій моделі.

### A. persistent_workers + prefetch_factor (+ фікс пропагації епохи)
Файли: `configs/label_studio/ls_dataloader.yml`, `engine/misc/dist_utils.py:158-170`, `engine/data/dataset/_dataset.py`, `engine/data/dataloader.py`.
1. yaml (train + val лоадери): `persistent_workers: True`, `prefetch_factor: 4`. На одному GPU працює одразу — `workspace.create` передає всі yaml-ключі в конструктор DataLoader.
2. `warp_loader` (DDP-шлях, страховка): передавати `persistent_workers=loader.persistent_workers`, `prefetch_factor=(loader.prefetch_factor if loader.num_workers > 0 else None)`.
3. **Критично для коректності**: персистентні воркери тримають знімок датасету → `set_epoch` з головного процесу до них не доходить → політика вимкнення аугментацій (RandomZoomOut/IoUCrop вимикаються на епосі 60; фактичний stop_epoch задає `tools/utils.py apply_ls_params` = int(5/6·epoches)) ніколи б не спрацювала. Фікс: зберігати епоху в shared-memory тензорі в `DetDataset.set_epoch` / `BaseCollateFunction.set_epoch`:
   `self._shared_epoch = torch.tensor([epoch], dtype=torch.int64).share_memory_()` (створити один раз, далі оновлювати in-place); property `epoch` читає `int(self._shared_epoch[0])`. Shared-тензор переживає і fork, і spawn-піклінг воркерів; `set_epoch` викликається до створення ітератора епохи, тож розсинхрону на межі немає.
- Очікувано: прибирає ~80s/епоху затримки на respawn ≈ 1.6 год за 72 епохи. Ціна в RAM: 16 живих воркерів (COCO-індекс COW-шариться) + буфери префетчу (~кілька ГБ); якщо тісно — знизити prefetch_factor до 2.
- Перевірка: короткий ран (epoches=4, епоха політики примусово 2 через `-u`) з тимчасовим debug-принтом `dataset.epoch` всередині `stop_epoch_forward` — має перемкнутися на епосі 2 всередині воркерів; у профайлінгу з епохи 2 не повинно бути затримки першого батча.

### D. Зменшення CPU-синків у criterion (окремий PR, обов'язковий fixed-seed A/B)
Файли: `engine/deim/deim_criterion.py`, `engine/deim/utils.py`.
Конфіг suppress/ignore АКТИВНИЙ (ls_coco_detection.yml) → ці шляхи виконуються щокроку, а маски перераховуються всередині КОЖНОГО виклику класифікаційного лосу (~8 груп шарів/крок: main + 2 aux + pre + enc + 3 dn). Найгірші винуватці — приховані `.any()`/`.item()` синки (~250/крок), гірші за сам `.cpu()` matcher-а.
1. **utils.py:40-43 `filter_suppress_source_targets`**: замінити цикл з `.item()` по лейблах на `torch.isin(t['labels'], ids_tensor)`. Семантика ідентична, чистий рефакторинг.
2. **`_get_ignore_tag_mask` (deim_criterion.py:361-380)**: векторизувати цикл `t[tag_name].item()` (stack значень тегів, `rows = present & (vals == 0)`), повертати маску безумовно (прибрати синк `mask.any()` — маска з усіма False далі є no-op).
3. **`_get_suppress_mask` (deim_criterion.py:321-359)**: не можна порахувати один раз на крок (залежить від `pred_boxes` та `indices` конкретного шару), але *таргет-залежні* частини — можна: раз на forward попередньо порахувати (a) `box_cxcywh_to_xyxy(st['boxes'])` для кожного зображення, (b) маску `[bs, num_classes]` пригнічуваних класів через статичну LUT `[num_classes, num_classes]` з `suppress_classes` — замінює цикл по боксах з `.item()` по лейблах. У per-call частині: будувати `matched` через scatter з `_get_src_permutation_idx` (без Python-циклу), лишити per-image цикл `box_iof`, але прибрати всі early-out `.any()`; комбінувати `overlapping.unsqueeze(-1) & cls_mask.unsqueeze(1)`. Захист за shape, коли кількість класів шару ≠ повній (клас-агностична enc-гілка тимчасово ставить `self.num_classes = 1`, deim_criterion.py:491-496) — поведінку зберегти точно.
4. **`_get_go_indices` (deim_criterion.py:246-265)**: замінити цикл dict з `.item()` по парах на векторизований `torch.unique(dim=0)` + стабільний argsort за лічильниками + first-per-row через scatter_reduce. Нюанс: при точних нічиїх лічильників вибрана колонка може відрізнятися від старого коду — старий порядок нічиїх сам був недетермінований (нестабільний argsort); нова версія детермінована. У межах шуму тренування; зазначити в PR.
5. Батчинг matcher-а (один `.cpu()` для всіх 5 cost-матриць) — **відкладено**; лише якщо після пунктів 1–4 профайлінг з profile_sync все ще покаже домінування стоянок на matcher.
- Очікувано: вікно criterion 0.15s → ~0.05–0.08s/крок + перестає серіалізувати чергу CUDA (покращується перекриття з backward).
- Перевірка: fixed-seed A/B на 200 кроків з дампом loss_dict щокроку — точна рівність для пунктів 1–2, статистична рівність для 3–4 (нічиї); далі 2-епоховий sanity-ран з mAP.

## Крок 0.5: бенчмарк бекендів декодування — `tools/benchmark_decode.py` (новий, на `exp0-baseline`)

За запитом користувача: замість відхилення бекендів декодування «на папері» — виміряти їх емпірично ДО будь-якого тренування. Окремий скрипт, без тренування:
- Семплить N (за замовчуванням 200) реальних зображень з `/data/GM_dataset/train` (арг `--img_dir`, опціонально `--ann_file` для семплінгу з анотацій).
- Міряє для кожного бекенда: (а) чисте декодування в RGB, (б) декодування + реальний ланцюг train-трансформів (RandomZoomOut→IoUCrop→Resize 512), де застосовно.
- Бекенди: PIL (поточний), OpenCV `cv2.imdecode`/`imread`, `torchvision.io.decode_jpeg` (CPU), PyTurboJPEG (якщо імпортується; інакше пропустити без падіння).
- Вивід: таблиця imgs/sec + прискорення відносно PIL, друк + збереження в json.
- Правило рішення: якщо переможець швидший за PIL >1.5× на чистому декодуванні → впроваджуємо його в умовній гілці `exp5-decode-backend` (зміна обмежена завантаженням зображення в `CocoDetection`, engine/data/dataset/coco_dataset.py; трансформи torchvision v2 приймають і тензори, і PIL — сумісність пайплайна вирішується в цій гілці).
- Нотатка про чесність (у виводі скрипта + exp.md): різні реалізації libjpeg дають піксельні відмінності округлення (chroma upsampling) — семантично це шум, але бітової ідентичності з baseline не буде; тому це окремий експеримент у хвості стеку, а не зміна в усіх експериментах.

## Гарантії відтворюваності та порівнюваності (документуються в кожному exp.md)

Ідентичне в УСІХ ранах: seed (`--seed=0` захардкоджено в оркестраторі), docker-образ, машина/GPU, датасет read-only, порядок семплів (sampler сідиться від seed), конфіг, епохи, роздільна здатність.
НЕ бітово-ідентичне між гілками (очікувано, в межах шуму тренування):
- exp1+: TF32 змінює округлення fp32 у matmul/conv.
- exp3+: персистентні воркери зберігають стан RNG між епохами замість патерну respawn-reseed → потоки аугментацій розходяться з exp0–2 починаючи з епохи 2.
- exp4: зміна детермінізму нічиїх у matcher (`_get_go_indices`).
- exp5 (умовно): піксельні відмінності округлення JPEG-декодування.
Наслідок: порівняння ШВИДКОСТІ (скаляри Profiling/*, wall time епохи) повністю валідне між усіма гілками — це головна метрика. mAP (Test/*) порівнюється в межах нормального шуму тренування; коректність кожної зміни доводиться закоміченими fixed-seed A/B скриптами (exp2 allclose, exp4 рівність дампів лосів), які порівнюють старий і новий код за ідентичного RNG в межах однієї гілки.

## Викинуто / відхилено

- val_freq (валідація раз на N епох) — відхилено користувачем (семантика).
- Офлайн pre-resize + перезапис COCO json — відхилено користувачем (семантика). Примітка в звіті лишається: це єдина мітигація втрат через storage contention на кшталт ранів 252/253.
- Бекенди декодування — повернуто в гру за запитом користувача: тепер вимірюються емпірично через бенчмарк Кроку 0.5; впроваджуються як `exp5-decode-backend` лише якщо бенчмарк покаже виграш >1.5× на декодуванні.
- Твіки num_workers/pin_memory — у конфізі вже оптимально.

## Крок 0 (ПЕРШИЙ): оркестратор контейнерних експериментів — `run_experiments.sh` (новий, на `exp0-baseline`)

Скрипт запускається на хості; приймає назви гілок-експериментів і запускає кожне тренування послідовно у власному контейнері. Змодельований за наявними `run_docker.sh` + `train.sh`.

Використання: `./run_experiments.sh exp0-baseline exp1-tf32-nan-gate ...` (без аргументів → усі exp-гілки в порядку стеку, починаючи з `exp0-baseline`). Перевизначається через env: `EPOCHS` (за замовчуванням 6 — вибір користувача: мінімум для надійного заміру швидкості, ~2.5–3 год/експеримент), `RES` (за замовчуванням "512 512"), `IMAGE` (quay.io/logivations/ml_all:LS_dfine_latest), `DATASET_DIR` (/data/GM_dataset), `RESULTS_DIR` (/data/GM_results).

Для кожної гілки скрипт:
1. Створює/оновлює git worktree у `/data/DEIM_worktrees/{exp_name}` для цієї гілки (основний checkout репозиторію не чіпається; кожен контейнер отримує точний код гілки).
2. `mkdir -p /data/GM_results/{exp_name}`.
3. Запускає у foreground (послідовно):
```bash
docker run --name deim_train_{exp_name} --rm --gpus all --shm-size=16g \
  --network host -w /DEIM \
  -v {worktree}:/DEIM \
  -v /data/GM_dataset:/dataset:ro \
  -v /data/GM_results/{exp_name}:/result \
  -v /data/DEIM/.cache/torch:/root/.cache/torch \
  -v /data/DEIM/.cache/triton:/root/.triton/cache \
  -v /data/DEIM/.cache/nv:/root/.nv \
  $IMAGE bash -c "torchrun --master_port=7777 --nproc_per_node=1 train.py \
    -c configs/label_studio/ls_dfine_hgnetv2_s_coco.yml \
    --train-epochs $EPOCHS --training-res $RES --use-amp --seed=0 \
    -u profile_sync=True" 2>&1 | tee /data/GM_results/{exp_name}/train.log
```
4. При падінні: зупиняє послідовність (лог + ненульовий exit), якщо не задано `CONTINUE_ON_FAIL=1`.

Нотатки:
- `-u profile_sync=True` гарантує, що скаляри Profiling/* у TensorBoard заміряні з cuda.synchronize (без старих помилок асинхронної атрибуції). Скаляри Test/* (mAP тощо) вже пишуться щоепохи (det_solver.py:166-169). Все це потрапляє в /result → /data/GM_results/{exp_name} → оцінка через файли TensorBoard. ВАЖЛИВО: `main` не знає ключа `profile_sync` — флаг з'являється в exp0-baseline, тому оркестратор запускає лише exp-гілки (задокументовано в шапці скрипта). Базовим порівняльним раном Є `exp0-baseline` (лише фікс профайлера, без прискорень).
- shm-size підвищено 8g→16g (persistent workers + prefetch_factor 4 в exp3+ тримають більше тензорів у shared memory).
- Датасет монтується read-only. Використовується СТАРИЙ формат датасету: на exp0-baseline ревертнуто мердж PR RTDT-7618 (9bb25de), тож конфіг чекає /dataset/train/, /dataset/test/ і /dataset/coco_annotations.json; хостовий каталог $DATASET_DIR має мати саме цей лейаут.
- Кеш torch спільний між ранами → pretrained-ваги HGNetv2 качаються один раз.
- Поділ епох: `apply_ls_params` ставить вимкнення аугментацій/рестарт EMA на int(5/6·epoches) → з EPOCHS=6 межа — рівно епоха 5 (5 епох stage1 + 1 stage2), override не потрібен. Епоха 0 — прогрів (тюнінг cudnn.benchmark, старт воркерів, кеші) і виключається з порівняння між гілками; епохи 1–4 дають чисту швидкість stage1, епоха 5 — stage2. exp0 все одно додає опціональний override `stg1_epochs_perc` в apply_ls_params для гнучкості з іншими значеннями EPOCHS. Задокументовано в exp.md.

## Воркфлоу доставки: стековані гілки-експерименти

Кожен експеримент — окрема гілка, яка запускається в контейнері на ідентичному датасеті (135 тис. зображень), 12+12 епох, з порівнянням профайлінг-таблиць між гілками. Кожна гілка будується поверх попередньої і містить `exp.md` у корені репозиторію з описом: (а) що саме змінено в ЦЬОМУ експерименті (файли, флаги, очікуваний ефект, на що дивитися в профайлінгу) та (б) що унаслідувано від попередніх експериментів.

Стек гілок (кожна створюється від попередньої). **ПЕРШЕ тренування — це baseline** (exp0):

- **`main`**: не змінюється — всі скрипти і документація живуть на `exp0-baseline` (робоча гілка на сервері).
0. **`exp0-baseline`** (від `main`): реверт мерджу RTDT-7618 (повернення до старого формату датасету: /dataset/train, /dataset/test, /dataset/coco_annotations.json) + пункт H (флаг profile_sync + опціональний override `stg1_epochs_perc` в apply_ls_params) + `run_experiments.sh` + `tools/benchmark_decode.py` + цей документ. Нуль змін швидкості/семантики тренування — але скаляри Profiling/* у TB правдиві (cuda.synchronize). Це референсний ран, з яким порівнюються всі інші експерименти, і робоча гілка для сервера. exp.md: пояснює, що це baseline лише з фіксом вимірювань, нічого не унаслідовано.
1. **`exp1-tf32-nan-gate`** (від exp0): пункт B (TF32 + cudnn.benchmark) + пункт E (debug_nan за флагом, вимкнено за замовчуванням). exp.md: ці зміни + унаслідоване з exp0.
2. **`exp2-ema-foreach`** (від exp1): пункт C (переписаний ModelEMA на foreach) + allclose sanity-скрипт. exp.md: ця зміна + унаслідоване з exp0–1.
3. **`exp3-persistent-workers`** (від exp2): пункт A (yaml persistent_workers/prefetch_factor + патч warp_loader + shared-memory пропагація епохи). exp.md: ця зміна + унаслідоване з exp0–2; нотатка про ціну в RAM і поведінку пропагації епохи (перемикання політики аугментацій на stop_epoch).
4. **`exp4-criterion-desync`** (від exp3): пункт D 1–4 (векторизація criterion) + fixed-seed скрипт дампу лосів для A/B. exp.md: ця зміна + унаслідоване з exp0–3.
5. **`exp5-decode-backend`** (від exp4, УМОВНА): створюється лише якщо бенчмарк Кроку 0.5 покаже виграш >1.5× на декодуванні — перемикає завантаження зображень у `CocoDetection` на бекенд-переможець. exp.md: ця зміна + унаслідоване з exp0–4 + таблиця результатів бенчмарку.

Примітка для ранів 12+12 епох: `tools/utils.py apply_ls_params` ставить stop_epoch = int(5/6·epoches) — з epoches=24 межа вимкнення аугментацій/рестарту EMA припадає на епоху 20, тож розбивка 12+12 покриває обидві стадії лише за відповідного конфігурування; це буде зазначено в exp.md, щоб рани були порівнюваними.

## Верифікація (end-to-end)

1. Ран exp0-baseline з `profile_sync: True` → нова правдива базова атрибуція стадій (очікується, що «criterion 29%» перерозподілиться).
2. Кожна наступна гілка: ран з тим самим конфігом; порівняння per-stage таблиці профайлінгу з батьківською гілкою (однаковий датасет/епохи → таблиці прямо порівнювані).
3. Перед передачею: fixed-seed перевірка рівності лосів на 200 кроків для exp4 (criterion) та allclose для exp2 (EMA) — скрипти закомічені у відповідні гілки.
4. Очікуваний фінальний стан (exp4): крок 0.53s → ~0.35–0.42s; екстрапольовано на повні 72 епохи wall ~38 год → ~27–31 год.

## Підсумок очікуваного ефекту

| Пункт | Економія |
|-------|----------|
| B TF32/cudnn | 5–15% математики fwd/bwd/criterion |
| E NaN-gate | ~0.02–0.05s/крок |
| C EMA foreach | ~0.055s/крок (~10% wall) |
| A persistent workers | ~1.6 год фіксовано + рівніша подача даних |
| D десинк criterion | ~0.07–0.10s/крок + перекриття з backward |
| **Разом** | **~38 год → ~27–31 год**, семантика тренування ідентична |
