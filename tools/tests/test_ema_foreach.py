"""
A/B check: the foreach ModelEMA.update must produce the same result as the
original per-tensor loop. Runs on CPU, no dataset or GPU needed:

    python3 tools/tests/test_ema_foreach.py
"""
import copy
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from engine.optim.ema import ModelEMA


class ReferenceEMA:
    """The original (pre-foreach) update loop, kept verbatim as the reference."""

    def __init__(self, model, decay=0.9999, warmups=1000):
        self.module = copy.deepcopy(model).eval()
        self.updates = 0
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / warmups))

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            d = self.decay_fn(self.updates)
            msd = model.state_dict()
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


def make_model():
    m = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.BatchNorm2d(8),  # float buffers (running stats) + int buffer (num_batches_tracked)
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, 5),
    )
    return m


def main():
    torch.manual_seed(0)
    model = make_model()

    ema_new = ModelEMA(model, decay=0.9999, warmups=1000, start=0)
    ema_ref = ReferenceEMA(model, decay=0.9999, warmups=1000)

    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    for step in range(10):
        x = torch.randn(4, 3, 4, 4)
        loss = model(x).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        ema_new.update(model)
        ema_ref.update(model)

    new_sd = ema_new.module.state_dict()
    ref_sd = ema_ref.module.state_dict()
    assert set(new_sd) == set(ref_sd), 'state_dict keys diverged'

    worst = 0.0
    for k in ref_sd:
        if not ref_sd[k].dtype.is_floating_point:
            assert torch.equal(new_sd[k], ref_sd[k]), f'non-float buffer changed: {k}'
            continue
        diff = (new_sd[k] - ref_sd[k]).abs().max().item()
        worst = max(worst, diff)
        assert torch.allclose(new_sd[k], ref_sd[k], rtol=0, atol=1e-7), \
            f'{k}: max abs diff {diff}'

    print(f'OK: foreach EMA matches the reference loop after 10 updates '
          f'(worst abs diff {worst:.2e})')


if __name__ == '__main__':
    main()
