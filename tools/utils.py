import json
import argparse
import torch

from engine.core import YAMLConfig
from typing import Tuple

def apply_ls_params(args: argparse.Namespace, cfg: YAMLConfig, stg1_epochs_perc: float = 1 / 6):
    # Set epochs
    cfg.epoches = args.train_epochs
    cfg.yaml_cfg['train_dataloader']['dataset']['transforms']['policy']['epoch'] = int((1 - stg1_epochs_perc) * cfg.epoches)
    cfg.yaml_cfg['train_dataloader']['collate_fn']['stop_epoch'] = int((1 - stg1_epochs_perc) * cfg.epoches)

    # Set resolution
    cfg.yaml_cfg['eval_spatial_size'] = args.training_res

    for i, aug in enumerate(cfg.yaml_cfg['train_dataloader']['dataset']['transforms']['ops']):
        if aug['type'] == 'Resize':
            cfg.yaml_cfg['train_dataloader']['dataset']['transforms']['ops'][i]['size'] = args.training_res

    cfg.yaml_cfg['train_dataloader']['collate_fn']['base_size'] = args.training_res[0]
    cfg.yaml_cfg['val_dataloader']['dataset']['transforms']['ops'][0]['size'] = args.training_res
    if hasattr(args, 'num_classes'):
        # Inference
        num_classes = args.num_classes
    else:
        # Train
        with open(cfg.yaml_cfg['train_dataloader']['dataset']['ann_file'], "r") as f:
            coco_ann = json.load(f)
        num_classes = len(coco_ann['categories'])
    cfg.yaml_cfg['num_classes'] = num_classes
    return cfg

def scale_bbox_coordinates(
    bbox: torch.Tensor,
    source_image_shape: Tuple[int, int],
    target_image_shape: Tuple[int, int],
) -> torch.Tensor:
    """
    Scale bbox coordinates tensor from source_image_shape to target_image_shape.

    Args:
        bbox: torch.Tensor of shape [4] with (xmin, ymin, xmax, ymax)
        source_image_shape: (width, height) of original image
        target_image_shape: (width, height) of target image

    Returns:
        Scaled bbox tensor shape [4]
    """
    factor_x = source_image_shape[0] / target_image_shape[0]
    factor_y = source_image_shape[1] / target_image_shape[1]

    # Scale bbox
    scaled_bbox = bbox.clone()
    scaled_bbox[0] = bbox[0] * factor_x
    scaled_bbox[1] = bbox[1] * factor_y
    scaled_bbox[2] = bbox[2] * factor_x
    scaled_bbox[3] = bbox[3] * factor_y

    return scaled_bbox

def convert_bbox_format(bbox, from_format="xywh", to_format="xyxy"):
    """Convert bounding box format between xywh and xyxy."""
    x_min, y_min = bbox[0], bbox[1]
    if from_format == "xywh" and to_format == "xyxy":
        width, height = bbox[2], bbox[3]
        return [x_min, y_min, x_min + width, y_min + height]
    elif from_format == "xyxy" and to_format == "xywh":
        x_max, y_max = bbox[2], bbox[3]
        return [x_min, y_min, x_max - x_min, y_max - y_min]
    return bbox  # Return unchanged if format is the same
