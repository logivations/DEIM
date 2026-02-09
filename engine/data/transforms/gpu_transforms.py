"""
GPU-accelerated transforms using Kornia.
These transforms run on batched GPU tensors for maximum performance.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional

try:
    import kornia.augmentation as K
    import kornia.geometry.transform as KG
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False



class GPUPhotometricDistort(nn.Module):
    """GPU equivalent of RandomPhotometricDistort using Kornia."""

    def __init__(
        self,
        brightness: Tuple[float, float] = (0.875, 1.125),
        contrast: Tuple[float, float] = (0.5, 1.5),
        saturation: Tuple[float, float] = (0.5, 1.5),
        hue: Tuple[float, float] = (-0.05, 0.05),
        p: float = 0.5,
    ):
        super().__init__()
        self.p = p
        # Kornia ColorJiggle expects ranges directly as tuples
        # brightness/contrast/saturation: multiplicative factor range
        # hue: additive shift range
        self.aug = K.ColorJiggle(
            brightness=brightness,  # (0.875, 1.125) -> multiply by factor in this range
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=p,
            same_on_batch=False,
        )

    def forward(self, images: torch.Tensor, targets: List[Dict]) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Args:
            images: [B, C, H, W] tensor on GPU, values in [0, 1]
            targets: list of target dicts (unchanged by color transforms)
        """
        images = self.aug(images)
        return images, targets


class GPUHorizontalFlip(nn.Module):
    """GPU equivalent of RandomHorizontalFlip using Kornia."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.aug = K.RandomHorizontalFlip(p=p, same_on_batch=False)

    def forward(self, images: torch.Tensor, targets: List[Dict]) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Flip images and adjust bounding boxes.
        Boxes are in cxcywh normalized format.
        """
        B, C, H, W = images.shape

        # Get flip params
        params = self.aug.forward_parameters(images.shape)
        do_flip = params['batch_prob']  # [B] tensor of bools

        # Apply flip to images
        images = self.aug(images, params=params)

        # Adjust boxes for flipped images
        new_targets = []
        for i, (t, flip) in enumerate(zip(targets, do_flip)):
            new_t = {k: v for k, v in t.items()}
            if flip and 'boxes' in t:
                boxes = t['boxes'].clone()
                # For cxcywh normalized: flip cx = 1 - cx
                boxes[:, 0] = 1.0 - boxes[:, 0]
                new_t['boxes'] = boxes
            new_targets.append(new_t)

        return images, new_targets

class GPUTransformPipeline(nn.Module):
    """
    Pipeline of GPU transforms to apply after data is moved to GPU.
    """

    def __init__(self, transforms: List[nn.Module]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, images: torch.Tensor, targets: List[Dict]) -> Tuple[torch.Tensor, List[Dict]]:
        for transform in self.transforms:
            images, targets = transform(images, targets)
        return images, targets


# Mapping from CPU transform names to GPU classes
# Only transforms that don't change image dimensions (batching requires same size)
GPU_TRANSFORM_MAP = {
    'RandomPhotometricDistort': GPUPhotometricDistort,
    'RandomHorizontalFlip': GPUHorizontalFlip,
}


def build_gpu_transforms(gpu_ops: List[Dict], device: torch.device = None) -> Optional[GPUTransformPipeline]:
    """
    Build GPU transform pipeline from explicit gpu_ops config.

    Args:
        gpu_ops: List of GPU transform configs from yaml
        device: Target device (optional, transforms work on any device)

    Returns:
        GPUTransformPipeline or None if no GPU transforms needed
    """
    if not KORNIA_AVAILABLE:
        return None

    if not gpu_ops:
        return None

    gpu_transforms = []

    for op in gpu_ops:
        if isinstance(op, dict):
            name = op.get('type')
            if name not in GPU_TRANSFORM_MAP:
                raise ValueError(
                    f"Unknown GPU transform: '{name}'. "
                    f"Available: {list(GPU_TRANSFORM_MAP.keys())}"
                )
            # Get params (excluding 'type')
            params = {k: v for k, v in op.items() if k != 'type'}
            transform = GPU_TRANSFORM_MAP[name](**params)
            gpu_transforms.append(transform)

    if not gpu_transforms:
        return None

    pipeline = GPUTransformPipeline(gpu_transforms)
    if device is not None:
        pipeline = pipeline.to(device)

    return pipeline
