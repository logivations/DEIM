import json
import argparse
from engine.core import YAMLConfig
from typing import List, Tuple

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
    bbox: List[float],
    source_image_shape: Tuple[int, int],
    target_image_shape: Tuple[int, int],
) -> List[float]:
    """
    Scale the given bbox from image coordinates of source_image_shape, to
    image coordinates of target_image_shape.
    """
    factor_x = source_image_shape[0] / target_image_shape[0]
    factor_y = source_image_shape[1] / target_image_shape[1]
    bbox[0] *= factor_x
    bbox[1] *= factor_y
    bbox[2] *= factor_x
    bbox[3] *= factor_y
    return bbox
