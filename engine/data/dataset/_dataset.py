"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.utils.data as data

class DetDataset(data.Dataset):
    def __getitem__(self, index):
        img, target = self.load_item(index)
        if self.transforms is not None:
            img, target, _ = self.transforms(img, target, self)
        return img, target

    def load_item(self, index):
        raise NotImplementedError("Please implement this function to return item before `transforms`.")

    def set_epoch(self, epoch) -> None:
        # Shared-memory tensor: with persistent_workers the DataLoader workers
        # keep their pickled copy of the dataset alive across epochs, so a plain
        # attribute set in the main process would never reach them. A shared
        # tensor is seen by the live workers immediately.
        if not hasattr(self, '_shared_epoch'):
            self._shared_epoch = torch.tensor([int(epoch)], dtype=torch.int64).share_memory_()
        else:
            self._shared_epoch[0] = int(epoch)

    @property
    def epoch(self):
        return int(self._shared_epoch[0]) if hasattr(self, '_shared_epoch') else -1
