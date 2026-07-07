"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import os

import torch
import torch.utils.data

import torchvision

from PIL import Image
import faster_coco_eval
import faster_coco_eval.core.mask as coco_mask
from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register

torchvision.disable_beta_transforms_warning()
faster_coco_eval.init_as_pycocotools()
Image.MAX_IMAGE_PIXELS = None

__all__ = ['CocoDetection']


@register()
class CocoDetection(torchvision.datasets.CocoDetection, DetDataset):
    __inject__ = ['transforms', ]
    __share__ = ['remap_mscoco_category', 'ignore_tags', 'suppress_classes']

    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks=False,
        remap_mscoco_category=False,
        ignore_tags=None,
        suppress_classes=None,
        presize_res=None
    ):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.ignore_tags_cfg = ignore_tags or {}
        # JPEG DCT-domain scaled decode: images are decoded directly at 1/2,
        # 1/4 or 1/8 size (whichever still keeps both sides >= presize_res),
        # bboxes are rescaled to match. Cuts worker CPU cost of decode and of
        # the full-resolution augmentations without touching the files on disk.
        self.presize_res = presize_res

        self.prepare = ConvertCocoPolysToMask(
            return_masks,
            ignore_tag_names=list(self.ignore_tags_cfg.keys())
        )
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

        # Exclude images from folders with different tags
        self.ids = [
            id_ for id_ in self.coco.getImgIds()
            if os.path.exists(os.path.join(self.root, self.coco.loadImgs(id_)[0]['file_name']))
        ]
        
        # Resolve ignore class names -> category IDs for ignore_tags
        cat_name2id = {cat['name']: cat['id'] for cat in self.categories}

        self.ignore_tags_resolved = {}
        for tag_name, class_names in self.ignore_tags_cfg.items():
            unknown = [n for n in class_names if n not in cat_name2id]
            if unknown:
                print(f"Warning: ignore_tags classes {unknown} not found in dataset categories")
            self.ignore_tags_resolved[tag_name] = [cat_name2id[n] for n in class_names if n in cat_name2id]

        # Resolve suppress class names -> category IDs for suppress_classes
        suppress_cfg = suppress_classes or {}
        self.suppress_classes_resolved = {}
        for source_name, suppress_names in suppress_cfg.items():
            if source_name not in cat_name2id:
                print(f"Warning: suppress_classes key '{source_name}' not found in dataset categories")
                continue
            unknown = [n for n in suppress_names if n not in cat_name2id]
            if unknown:
                print(f"Warning: suppress_classes values {unknown} not found in dataset categories")
            source_id = cat_name2id[source_name]
            self.suppress_classes_resolved[source_id] = [cat_name2id[n] for n in suppress_names if n in cat_name2id]

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def _load_image(self, id: int):
        if self.presize_res:
            path = os.path.join(self.root, self.coco.loadImgs(id)[0]['file_name'])
            image = Image.open(path)
            # libjpeg decodes fewer DCT coefficients -> faster than full decode.
            # No-op for non-JPEG formats and for images already <= presize_res.
            image.draft('RGB', (self.presize_res, self.presize_res))
            return image.convert('RGB')
        return super(CocoDetection, self)._load_image(id)

    def load_item(self, idx):
        image, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]

        orig_wh = None
        if self.presize_res:
            info = self.coco.loadImgs(image_id)[0]
            orig_wh = (int(info['width']), int(info['height']))
            sx = image.size[0] / orig_wh[0]
            sy = image.size[1] / orig_wh[1]
            if sx != 1.0 or sy != 1.0:
                # Annotations are in original pixel coords; rescale COPIES to the
                # decoded size (the dicts belong to the shared COCO index —
                # mutating them in place would corrupt every following epoch).
                scaled = []
                for ann in target:
                    ann = dict(ann)
                    x, y, w, h = ann['bbox']
                    ann['bbox'] = [x * sx, y * sy, w * sx, h * sy]
                    if 'area' in ann:
                        ann['area'] = ann['area'] * sx * sy
                    scaled.append(ann)
                target = scaled

        target = {'image_id': image_id, 'annotations': target}

        if self.remap_mscoco_category:
            image, target = self.prepare(image, target, category2label=mscoco_category2label)
        else:
            image, target = self.prepare(image, target)

        if orig_wh is not None:
            # Keep evaluation in the original pixel space: the postprocessor maps
            # normalized predictions with orig_size, and the COCO GT used by the
            # evaluator is in original coordinates.
            target['orig_size'] = torch.as_tensor(orig_wh)

        target['idx'] = torch.tensor([idx])

        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=image.size[::-1])

        if 'masks' in target:
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks')

        return image, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        if hasattr(self, '_preset') and self._preset is not None:
            s += f' preset:\n   {repr(self._preset)}'
        return s

    @property
    def categories(self, ):
        return self.coco.dataset['categories']

    @property
    def category2name(self, ):
        return {cat['id']: cat['name'] for cat in self.categories}

    @property
    def category2label(self, ):
        return {cat['id']: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self, ):
        return {i: cat['id'] for i, cat in enumerate(self.categories)}


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, ignore_tag_names=None):
        self.return_masks = return_masks
        self.ignore_tag_names = ignore_tag_names or []

    def __call__(self, image: Image.Image, target, **kwargs):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        category2label = kwargs.get('category2label', None)
        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]

        labels = torch.tensor(labels, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        # target["size"] = torch.as_tensor([int(w), int(h)])

        # Extract ignore tag fields (image-level flags from annotations)
        for tag_name in self.ignore_tag_names:
            if anno and tag_name in anno[0]:
                tag_val = int(anno[0][tag_name]) # all annos on image have same value
            else:
                tag_val = 0  # default: tag missing = NOT annotated (suppress FP)
            target[tag_name] = torch.tensor(tag_val, dtype=torch.int64)  # 0-dim scalar

        return image, target


mscoco_category2name = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
