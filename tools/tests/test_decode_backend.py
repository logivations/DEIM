"""
A/B check for the torchvision decode backend (exp5-decode-backend).

Verifies on a synthetic JPEG + COCO annotations that:
  1. torchvision.io decode matches PIL pixel-for-pixel (same libjpeg-turbo)
     or at worst within rounding noise;
  2. ConvertPILImage produces identical float output for both input kinds
     and does NOT touch BoundingBoxes (they are Tensor subclasses too);
  3. the full CocoDetection pipeline (load_item + train-style transforms)
     yields equal image tensors, boxes and labels for both backends.

Runs on CPU:

    python3 tools/tests/test_decode_backend.py
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from torchvision import tv_tensors
from torchvision.io import decode_image, read_file, ImageReadMode

from engine.data.transforms._transforms import ConvertPILImage
from engine.data.dataset.coco_dataset import CocoDetection
from engine.data.transforms.container import Compose


def make_dataset_dir(tmp):
    rng = np.random.default_rng(0)
    img_dir = os.path.join(tmp, 'train')
    os.makedirs(img_dir)
    images, annotations = [], []
    for i in range(3):
        w, h = 640 + 32 * i, 480 + 32 * i
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(img_dir, f'{i}.jpg'), quality=95)
        images.append({'id': i, 'file_name': f'{i}.jpg', 'width': w, 'height': h})
        for j in range(4):
            x, y = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 60))
            bw, bh = int(rng.integers(20, 60)), int(rng.integers(20, 60))
            annotations.append({'id': len(annotations), 'image_id': i, 'category_id': 1 + j % 2,
                                'bbox': [x, y, bw, bh], 'area': bw * bh, 'iscrowd': 0})
    ann_file = os.path.join(tmp, 'coco_annotations.json')
    with open(ann_file, 'w') as f:
        json.dump({'images': images, 'annotations': annotations,
                   'categories': [{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'b'}]}, f)
    return img_dir, ann_file


def check_decode_parity(img_dir):
    for name in os.listdir(img_dir):
        path = os.path.join(img_dir, name)
        with Image.open(path) as im:
            pil = torch.from_numpy(np.asarray(im.convert('RGB'))).permute(2, 0, 1)
        tv = decode_image(read_file(path), mode=ImageReadMode.RGB)
        diff = (pil.int() - tv.int()).abs().max().item()
        assert pil.shape == tv.shape and diff <= 2, f'{name}: decode diff {diff}'
    print('OK: decode parity PIL vs torchvision.io (max pixel diff <= 2)')


def check_convert_transform(img_dir):
    path = os.path.join(img_dir, os.listdir(img_dir)[0])
    t = ConvertPILImage(dtype='float32', scale=True)

    with Image.open(path) as im:
        out_pil = t(im.convert('RGB'))
    out_tv = t(tv_tensors.Image(decode_image(read_file(path), mode=ImageReadMode.RGB)))
    for out in (out_pil, out_tv):
        assert isinstance(out, tv_tensors.Image) and out.dtype == torch.float32
        assert 0.0 <= out.min() and out.max() <= 1.0, 'not scaled to [0, 1]'
    assert (out_pil - out_tv).abs().max().item() <= 2 / 255.

    # BoundingBoxes must pass through ConvertPILImage untouched
    boxes = tv_tensors.BoundingBoxes(torch.tensor([[1., 2., 30., 40.]]),
                                     format='XYXY', canvas_size=(480, 640))
    img_out, target_out = t(tv_tensors.Image(torch.zeros(3, 480, 640, dtype=torch.uint8)),
                            {'boxes': boxes})
    assert torch.equal(target_out['boxes'], boxes) and target_out['boxes'].dtype == boxes.dtype
    assert img_out.dtype == torch.float32
    print('OK: ConvertPILImage handles both input kinds, boxes untouched')


def check_full_pipeline(img_dir, ann_file):
    transforms = Compose(ops=[
        {'type': 'Resize', 'size': [512, 512]},
        {'type': 'ConvertPILImage', 'dtype': 'float32', 'scale': True},
        {'type': 'ConvertBoxes', 'fmt': 'cxcywh', 'normalize': True},
    ])
    common = dict(ann_file=ann_file, transforms=transforms, return_masks=False)
    ds_pil = CocoDetection(img_folder=img_dir, decode_backend='pil', **common)
    ds_tv = CocoDetection(img_folder=img_dir, decode_backend='torchvision', **common)
    assert len(ds_pil) == len(ds_tv) == 3

    for idx in range(3):
        img_p, tgt_p = ds_pil[idx]
        img_t, tgt_t = ds_tv[idx]
        assert img_p.shape == img_t.shape == (3, 512, 512)
        img_diff = (img_p - img_t).abs().max().item()
        # decode is near-identical; Resize antialias PIL-vs-tensor differs slightly
        assert img_diff <= 0.05, f'idx {idx}: image diff {img_diff}'
        assert torch.allclose(tgt_p['boxes'], tgt_t['boxes'], atol=1e-6), 'boxes differ'
        assert torch.equal(tgt_p['labels'], tgt_t['labels']), 'labels differ'
    print('OK: full CocoDetection pipeline parity (pil vs torchvision backend)')


def main():
    with tempfile.TemporaryDirectory() as tmp:
        img_dir, ann_file = make_dataset_dir(tmp)
        check_decode_parity(img_dir)
        check_convert_transform(img_dir)
        check_full_pipeline(img_dir, ann_file)
    print('\nAll decode-backend A/B checks passed.')


if __name__ == '__main__':
    main()
