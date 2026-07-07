"""
A/B check for persistent_workers: set_epoch() from the main process must be
visible inside live DataLoader workers (the augmentation stop-policy reads
dataset.epoch there). Runs on CPU, no dataset or GPU needed:

    python3 tools/tests/test_persistent_epoch.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from engine.data.dataloader import DataLoader, BaseCollateFunction
from engine.data.dataset._dataset import DetDataset


class EpochEchoDataset(DetDataset):
    """Returns the epoch value as seen from inside the worker process."""

    def __init__(self, n=8):
        self.n = n
        self.transforms = None

    def __len__(self):
        return self.n

    def load_item(self, index):
        return torch.tensor([self.epoch]), {'epoch': torch.tensor([self.epoch])}


class EchoCollate(BaseCollateFunction):
    def __call__(self, items):
        # also echo the epoch the collate_fn (worker-side) sees
        return torch.stack([x[0] for x in items]), self.epoch


def main():
    dataset = EpochEchoDataset()
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
        persistent_workers=True,
        collate_fn=EchoCollate(),
        shuffle=None,
    )
    loader.shuffle = False

    for epoch in range(3):
        loader.set_epoch(epoch)
        for imgs, collate_epoch in loader:
            worker_epochs = set(imgs.flatten().tolist())
            assert worker_epochs == {epoch}, \
                f'epoch {epoch}: dataset inside worker saw {worker_epochs}'
            assert collate_epoch == epoch, \
                f'epoch {epoch}: collate_fn inside worker saw {collate_epoch}'

    print('OK: set_epoch propagates into live persistent workers '
          '(dataset.epoch and collate_fn.epoch both correct across 3 epochs)')


if __name__ == '__main__':
    main()
