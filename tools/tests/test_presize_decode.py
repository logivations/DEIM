"""
A/B check for presize_res (JPEG DCT-domain scaled decode, exp2.5).

On a synthetic COCO dataset verifies that with presize_res:
  1. the decoded image is the expected 1/2-scale;
  2. bboxes are rescaled consistently (normalized coords match the full-res
     path exactly) and 'area' scales by sx*sy;
  3. `orig_size` stays in ORIGINAL pixel space (the evaluator maps predictions
     with it against the original-coordinate COCO GT);
  4. repeated reads return identical targets (the shared pycocotools index
     must never be mutated in place);
  5. the full transform chain (Resize 512 + Convert) yields matching
     normalized boxes for both paths.

Runs on CPU:

    python3 tools/tests/test_presize_decode.py
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from engine.data.dataset.coco_dataset import CocoDetection
from engine.data.transforms.container import Compose

W, H = 2400, 1800  # "5MP-style" JPEG; draft to >=768 picks exactly 1/2


def make_dataset_dir(tmp):
    rng = np.random.default_rng(0)
    img_dir = os.path.join(tmp, 'train')
    os.makedirs(img_dir)
    images, annotations = [], []
    for i in range(2):
        arr = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(img_dir, f'{i}.jpg'), quality=95)
        images.append({'id': i, 'file_name': f'{i}.jpg', 'width': W, 'height': H})
        for j in range(5):
            x, y = int(rng.integers(0, W - 300)), int(rng.integers(0, H - 300))
            bw, bh = int(rng.integers(50, 300)), int(rng.integers(50, 300))
            annotations.append({'id': len(annotations), 'image_id': i, 'category_id': 1 + j % 2,
                                'bbox': [x, y, bw, bh], 'area': bw * bh, 'iscrowd': 0})
    ann_file = os.path.join(tmp, 'coco_annotations.json')
    with open(ann_file, 'w') as f:
        json.dump({'images': images, 'annotations': annotations,
                   'categories': [{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'b'}]}, f)
    return img_dir, ann_file


def norm_boxes(boxes, size_wh):
    w, h = size_wh
    return boxes / torch.tensor([w, h, w, h], dtype=boxes.dtype)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        img_dir, ann_file = make_dataset_dir(tmp)
        common = dict(ann_file=ann_file, transforms=None, return_masks=False)
        ds_full = CocoDetection(img_folder=img_dir, **common)
        ds_draft = CocoDetection(img_folder=img_dir, presize_res=768, **common)

        for idx in range(2):
            img_f, t_f = ds_full.load_item(idx)
            img_d, t_d = ds_draft.load_item(idx)

            assert img_d.size == (W // 2, H // 2), f'expected 1/2 decode, got {img_d.size}'

            nf = norm_boxes(t_f['boxes'], img_f.size)
            nd = norm_boxes(t_d['boxes'], img_d.size)
            assert torch.allclose(nf, nd, atol=1e-6), 'normalized boxes diverged'
            assert torch.allclose(t_d['area'], t_f['area'] * 0.25, rtol=1e-6), 'area not rescaled'

            assert torch.equal(t_f['orig_size'], t_d['orig_size']), \
                f"orig_size must stay original: {t_f['orig_size']} vs {t_d['orig_size']}"
            assert t_d['orig_size'].tolist() == [W, H]
        print('OK: 1/2 decode, boxes/area rescaled, orig_size stays original')

        # shared COCO index must not be mutated: repeated reads identical
        _, first = ds_draft.load_item(0)
        _, second = ds_draft.load_item(0)
        assert torch.equal(first['boxes'], second['boxes']), \
            'boxes changed between reads — shared annotation dicts were mutated!'
        _, full_after = ds_full.load_item(0)
        nf = norm_boxes(full_after['boxes'], (W, H))
        nd = norm_boxes(second['boxes'], (W // 2, H // 2))
        assert torch.allclose(nf, nd, atol=1e-6), 'full-res path polluted by draft path'
        print('OK: repeated reads stable, COCO index not mutated')

        # full pipeline parity (Resize 512 + Convert)
        transforms = Compose(ops=[
            {'type': 'Resize', 'size': [512, 512]},
            {'type': 'ConvertPILImage', 'dtype': 'float32', 'scale': True},
            {'type': 'ConvertBoxes', 'fmt': 'cxcywh', 'normalize': True},
        ])
        ds_full_t = CocoDetection(img_folder=img_dir, ann_file=ann_file,
                                  transforms=transforms, return_masks=False)
        ds_draft_t = CocoDetection(img_folder=img_dir, ann_file=ann_file, presize_res=768,
                                   transforms=transforms, return_masks=False)
        for idx in range(2):
            img_f, t_f = ds_full_t[idx]
            img_d, t_d = ds_draft_t[idx]
            assert img_f.shape == img_d.shape == (3, 512, 512)
            assert torch.allclose(t_f['boxes'], t_d['boxes'], atol=1e-5), 'pipeline boxes diverged'
            assert torch.equal(t_f['labels'], t_d['labels'])
            img_diff = (img_f - img_d).abs().max().item()
            assert img_diff <= 0.15, f'image content diverged too much: {img_diff}'
        print('OK: full transform pipeline parity (512 boxes match, labels equal)')

    print('\nAll presize_res A/B checks passed.')


if __name__ == '__main__':
    main()
