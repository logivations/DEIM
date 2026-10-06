"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch
import torch.nn as nn

import math
from copy import deepcopy

from ..core import register
from ..misc import dist_utils

__all__ = ['ModelEMA']


@register()
class ModelEMA(object):
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model: nn.Module, decay: float=0.9999, warmups: int=1000, start: int=0):
        super().__init__()

        self.module = deepcopy(dist_utils.de_parallel(model)).eval()
        # if next(model.parameters()).device.type != 'cpu':
        #     self.module.half()  # FP16 EMA

        self.decay = decay
        self.warmups = warmups
        self.before_start = 0
        self.start = start
        self.updates = 0  # number of EMA updates
        # Read self.decay (not the captured constructor arg) so runtime changes
        # like `ema.decay = ema_restart_decay` at stop_epoch actually take effect.
        if warmups == 0:
            self.decay_fn = lambda x: self.decay
        else:
            self.decay_fn = lambda x: self.decay * (1 - math.exp(-x / warmups))  # decay exponential ramp (to help early epochs)

        for p in self.module.parameters():
            p.requires_grad_(False)

        self._cache_model_id = None

    def _build_cache(self, model: nn.Module):
        # state_dict tensors share storage with the live parameters/buffers, and
        # both the optimizer and load_state_dict update them in-place, so the
        # cached references stay valid. The cache is invalidated whenever the
        # model object changes, the EMA is moved, or a state dict is loaded.
        msd = dist_utils.de_parallel(model).state_dict()
        esd = self.module.state_dict()
        keys = [k for k, v in esd.items() if v.dtype.is_floating_point]
        assert all(k in msd for k in keys), 'EMA and model state_dict keys diverged'
        self._ema_tensors = [esd[k] for k in keys]
        self._model_tensors = [msd[k].detach() for k in keys]
        self._cache_model_id = id(model)

    def update(self, model: nn.Module):
        if self.before_start < self.start:
            self.before_start += 1
            return
        # Update EMA parameters: ema = d * ema + (1 - d) * model, fused across
        # all tensors with foreach ops instead of one tiny kernel per tensor.
        with torch.no_grad():
            self.updates += 1
            d = self.decay_fn(self.updates)
            if self._cache_model_id != id(model):
                self._build_cache(model)
            torch._foreach_mul_(self._ema_tensors, d)
            torch._foreach_add_(self._ema_tensors, self._model_tensors, alpha=1.0 - d)

    def to(self, *args, **kwargs):
        self.module = self.module.to(*args, **kwargs)
        self._cache_model_id = None  # .to() may reallocate storages
        return self

    def state_dict(self, ):
        return dict(module=self.module.state_dict(), updates=self.updates)

    def load_state_dict(self, state, strict=True):
        self.module.load_state_dict(state['module'], strict=strict)
        self._cache_model_id = None
        if 'updates' in state:
            self.updates = state['updates']

    def forwad(self, ):
        raise RuntimeError('ema...')

    def extra_repr(self) -> str:
        return f'decay={self.decay}, warmups={self.warmups}'



class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """
    def __init__(self, model, decay, device="cpu", use_buffers=True):

        self.decay_fn = lambda x: decay * (1 - math.exp(-x / 2000))

        def ema_avg(avg_model_param, model_param, num_averaged):
            decay = self.decay_fn(num_averaged)
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=use_buffers)
