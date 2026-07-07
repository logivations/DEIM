"""
Benchmark JPEG decode backends on a sample of the real training dataset.

Run this BEFORE the training experiments (no GPU needed, a few minutes):

    python3 tools/benchmark_decode.py --img_dir /data/GM_dataset/images

Backends: PIL (current pipeline), OpenCV, torchvision.io.decode_jpeg (CPU),
PyTurboJPEG (skipped if not installed). Measures:
  1. pure decode to RGB,
  2. decode + the expensive CPU part of the train transform chain
     (RandomZoomOut canvas up to 4x/side + Resize to 512), which dominates
     worker CPU time on ~5MP sources.

Decision rule (printed at the end): if the best backend beats PIL by more
than 1.5x on pure decode, switching the dataset loader is worth a dedicated
experiment branch (exp5-decode-backend). Note that different libjpeg
implementations differ at the pixel level (chroma-upsampling rounding), so a
backend switch is not bitwise-identical to the PIL baseline.

All backends are forced to single-threaded decode to mimic a DataLoader
worker process.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import functional as VF


def sample_images(img_dir: str, ann_file: str, n: int, seed: int):
    if ann_file:
        with open(ann_file) as f:
            ann = json.load(f)
        paths = [Path(img_dir) / img['file_name'] for img in ann['images']]
    else:
        paths = [p for p in Path(img_dir).rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg')]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit(f'No jpeg images found under {img_dir}')
    random.seed(seed)
    return random.sample(paths, min(n, len(paths)))


def transform_chain(img, zoom_out, size):
    # Representative CPU-side cost: ZoomOut allocates a canvas up to 4x per
    # side of the full-res image before everything is resized down to 512.
    img = zoom_out(img)
    return VF.resize(img, [size, size])


def bench(name, paths, decode_fn, zoom_out, size, with_transforms):
    # warmup (fs cache + lazy imports)
    for p in paths[:5]:
        decode_fn(p)

    t0 = time.perf_counter()
    for p in paths:
        decode_fn(p)
    decode_s = time.perf_counter() - t0

    full_s = None
    if with_transforms:
        t0 = time.perf_counter()
        for p in paths:
            transform_chain(decode_fn(p), zoom_out, size)
        full_s = time.perf_counter() - t0

    n = len(paths)
    print(f'{name:<22} decode: {n / decode_s:8.1f} img/s'
          + (f'   decode+transforms: {n / full_s:8.1f} img/s' if full_s else ''))
    return {'decode_s': decode_s, 'decode_ips': n / decode_s,
            'full_s': full_s, 'full_ips': (n / full_s) if full_s else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--img_dir', default='/data/GM_dataset/train')
    parser.add_argument('--ann_file', default=None, help='optional COCO json to sample file_names from')
    parser.add_argument('-n', '--num_images', type=int, default=200)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='decode_benchmark.json')
    args = parser.parse_args()

    paths = sample_images(args.img_dir, args.ann_file, args.num_images, args.seed)
    from PIL import Image
    with Image.open(paths[0]) as im:
        w, h = im.size
    print(f'Benchmarking {len(paths)} images from {args.img_dir} (first image: {w}x{h})\n')

    zoom_out = T.RandomZoomOut(fill=0, p=1.0)  # p=1 so every image pays the canvas cost
    results = {}

    def pil_decode(p):
        with Image.open(p) as im:
            return im.convert('RGB')
    results['PIL'] = bench('PIL (current)', paths, pil_decode, zoom_out, args.size, True)

    try:
        import cv2
        cv2.setNumThreads(1)  # DataLoader workers are effectively single-threaded

        def cv2_decode(p):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(img).permute(2, 0, 1)
        results['OpenCV'] = bench('OpenCV', paths, cv2_decode, zoom_out, args.size, True)
    except ImportError:
        print('OpenCV                 not installed, skipped')

    try:
        from torchvision.io import decode_jpeg, read_file, ImageReadMode

        def tv_decode(p):
            return decode_jpeg(read_file(str(p)), mode=ImageReadMode.RGB)
        results['torchvision.io'] = bench('torchvision.io (CPU)', paths, tv_decode, zoom_out, args.size, True)
    except Exception as e:
        print(f'torchvision.io         failed ({type(e).__name__}), skipped')

    try:
        from turbojpeg import TurboJPEG
        tj = TurboJPEG()

        def turbo_decode(p):
            with open(p, 'rb') as f:
                img = tj.decode(f.read())  # BGR
            return torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)
        results['PyTurboJPEG'] = bench('PyTurboJPEG', paths, turbo_decode, zoom_out, args.size, True)
    except (ImportError, OSError):
        print('PyTurboJPEG            not installed, skipped')

    print()
    base = results['PIL']['decode_ips']
    best_name, best = max(results.items(), key=lambda kv: kv[1]['decode_ips'])
    for name, r in results.items():
        print(f'{name:<22} {r["decode_ips"] / base:5.2f}x vs PIL (pure decode)')

    speedup = best['decode_ips'] / base
    print(f'\nBest backend: {best_name} ({speedup:.2f}x vs PIL)')
    if best_name != 'PIL' and speedup > 1.5:
        print('VERDICT: >1.5x decode speedup -> the exp5-decode-backend experiment is worth running.')
    else:
        print('VERDICT: below the 1.5x threshold -> keep PIL, skip exp5-decode-backend.')

    with open(args.out, 'w') as f:
        json.dump({'img_dir': args.img_dir, 'num_images': len(paths),
                   'results': results, 'best': best_name, 'speedup_vs_pil': speedup}, f, indent=2)
    print(f'Saved raw numbers to {args.out}')


if __name__ == '__main__':
    main()
