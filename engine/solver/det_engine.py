"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import sys
import math
import time
import random
from typing import Iterable
from dataclasses import dataclass, field
import gc
import psutil

import torch
import torch.amp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


@dataclass
class ProfilingMetrics:
    """Structured profiling metrics for training loop."""
    # Time accumulators (in seconds)
    data_load: float = 0.0
    data_transfer: float = 0.0  # includes GPU transforms + interpolate
    forward: float = 0.0
    criterion: float = 0.0
    backward: float = 0.0
    optimizer: float = 0.0
    other: float = 0.0  # EMA, LR scheduling, logging

    num_batches: int = 0

    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, getattr(self, key) + value)

    @property
    def total_time(self) -> float:
        return (self.data_load + self.data_transfer + self.forward +
                self.criterion + self.backward + self.optimizer + self.other)

    def get_averages(self) -> dict:
        """Get average times per batch."""
        n = max(self.num_batches, 1)
        return {
            'data_load': self.data_load / n,
            'data_transfer': self.data_transfer / n,
            'forward': self.forward / n,
            'criterion': self.criterion / n,
            'backward': self.backward / n,
            'optimizer': self.optimizer / n,
            'other': self.other / n,
            'total': self.total_time / n,
        }

    def get_percentages(self) -> dict:
        """Get percentage breakdown."""
        total = max(self.total_time, 1e-9)
        return {
            'data_load': self.data_load / total * 100,
            'data_transfer': self.data_transfer / total * 100,
            'forward': self.forward / total * 100,
            'criterion': self.criterion / total * 100,
            'backward': self.backward / total * 100,
            'optimizer': self.optimizer / total * 100,
            'other': self.other / total * 100,
        }

    def log_to_tensorboard(self, writer: SummaryWriter, epoch: int):
        """Log profiling metrics to tensorboard."""
        if writer is None:
            return
        avgs = self.get_averages()
        pcts = self.get_percentages()

        # Log average times
        for key, value in avgs.items():
            writer.add_scalar(f'Profiling/time_{key}', value, epoch)

        # Log percentages
        for key, value in pcts.items():
            writer.add_scalar(f'Profiling/pct_{key}', value, epoch)

    def print_summary(self, epoch: int):
        """Print formatted profiling summary."""
        avgs = self.get_averages()
        pcts = self.get_percentages()

        print("\n" + "=" * 70)
        print(f"[PROFILING] EPOCH {epoch} SUMMARY ({self.num_batches} batches)")
        print("=" * 70)
        print(f"  {'Stage':<20} {'Avg Time':>12} {'Percentage':>12}")
        print("-" * 70)
        print(f"  {'Data Load':<20} {avgs['data_load']:>10.4f}s {pcts['data_load']:>10.1f}%")
        print(f"  {'Data Transfer+GPU':<20} {avgs['data_transfer']:>10.4f}s {pcts['data_transfer']:>10.1f}%")
        print(f"  {'Forward Pass':<20} {avgs['forward']:>10.4f}s {pcts['forward']:>10.1f}%")
        print(f"  {'Criterion (Loss)':<20} {avgs['criterion']:>10.4f}s {pcts['criterion']:>10.1f}%")
        print(f"  {'Backward Pass':<20} {avgs['backward']:>10.4f}s {pcts['backward']:>10.1f}%")
        print(f"  {'Optimizer Step':<20} {avgs['optimizer']:>10.4f}s {pcts['optimizer']:>10.1f}%")
        print(f"  {'Other (EMA/LR)':<20} {avgs['other']:>10.4f}s {pcts['other']:>10.1f}%")
        print("-" * 70)
        print(f"  {'TOTAL':<20} {avgs['total']:>10.4f}s {100.0:>10.1f}%")
        print(f"  {'Epoch Time':<20} {self.total_time:>10.2f}s")
        print("=" * 70 + "\n")


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}')) # there shouldn't be fstring!
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)
    gpu_transforms = kwargs.get('gpu_transforms', None)
    multiscale_cfg = kwargs.get('multiscale_cfg', None)  # NEW: GPU multi-scale interpolate
    profile_sync = kwargs.get('profile_sync', False)
    debug_nan = kwargs.get('debug_nan', False)

    def _t():
        # CUDA kernels are async: without a synchronize the timers measure kernel
        # launch, not execution, and GPU time gets attributed to whatever stage
        # hits a sync point first. profile_sync=True trades a small overhead for
        # truthful per-stage attribution.
        if profile_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

    cur_iters = epoch * len(data_loader)

    # Structured profiling metrics
    profiling = ProfilingMetrics()
    t_load_start = _t()

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Measure data loading time
        t_data_load = _t() - t_load_start
        profiling.data_load += t_data_load

        # Time data transfer to GPU + GPU transforms + interpolate
        t_transfer_start = _t()
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Apply GPU transforms (if enabled)
        if gpu_transforms is not None:
            samples, targets = gpu_transforms(samples, targets)

        # Apply GPU multi-scale interpolate (if enabled)
        if multiscale_cfg is not None and epoch < multiscale_cfg['stop_epoch']:
            sz = random.choice(multiscale_cfg['scales'])
            samples = F.interpolate(samples, size=sz)

        t_data_transfer = _t() - t_transfer_start
        profiling.data_transfer += t_data_transfer

        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if scaler is not None:
            # Forward pass with AMP
            t_forward_start = _t()
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)
            t_forward = _t() - t_forward_start
            profiling.forward += t_forward

            # Debug-only: .any() forces a GPU sync every step. The math.isfinite(loss)
            # check below still aborts on divergence when this is off.
            if debug_nan and (torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any()):
                print("NaN or Inf detected")
                outputs['pred_boxes'] = torch.nan_to_num(outputs['pred_boxes'], nan=0.0)
                state = model.state_dict()
                new_state = {'model': {k.replace('module.', ''): v for k, v in state.items()}}
                dist_utils.save_on_master(new_state, "./NaN.pth")

            # Criterion
            t_criterion_start = _t()
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)
            t_criterion = _t() - t_criterion_start
            profiling.criterion += t_criterion
            loss = sum(loss_dict.values())

            # Backward
            t_backward_start = _t()
            scaler.scale(loss).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            t_backward = _t() - t_backward_start
            profiling.backward += t_backward

            # Optimizer
            t_optimizer_start = _t()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            t_optimizer = _t() - t_optimizer_start
            profiling.optimizer += t_optimizer

        else:
            # Forward pass without AMP
            t_forward_start = _t()
            outputs = model(samples, targets=targets)
            t_forward = _t() - t_forward_start
            profiling.forward += t_forward

            # Criterion
            t_criterion_start = _t()
            loss_dict = criterion(outputs, targets, **metas)
            t_criterion = _t() - t_criterion_start
            profiling.criterion += t_criterion

            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()

            # Backward
            t_backward_start = _t()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            t_backward = _t() - t_backward_start
            profiling.backward += t_backward

            # Optimizer
            t_optimizer_start = _t()
            optimizer.step()
            t_optimizer = _t() - t_optimizer_start
            profiling.optimizer += t_optimizer

        # Other overhead (EMA, LR scheduling, logging)
        t_other_start = _t()

        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

        t_other = _t() - t_other_start
        profiling.other += t_other
        profiling.num_batches += 1

        # Reset timer for next batch data loading
        t_load_start = _t()

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print and log profiling summary
    if profiling.num_batches > 0:
        profiling.print_summary(epoch)
        if dist_utils.is_main_process():
            profiling.log_to_tensorboard(writer, epoch)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator
