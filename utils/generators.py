from typing import Tuple

import torch

from network.augmentation import UpdatedAugmentStage


def pretrain_data_generator(
    images: torch.Tensor,  # bboxes or patches
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    perm = torch.randperm(len(images))
    images = images[perm]

    augment = UpdatedAugmentStage()
    augment = augment.to(device)

    size = len(images)
    start = 0
    while True:
        end = min(start + batch_size, size)
        if end - start > 1:
            x = images[start:end].to(device)
            xa = augment(x)
            xb = augment(x)
            yield xa, xb

        if end == size:
            start = 0
            perm = torch.randperm(len(images))
            images = images[perm]
        else:
            start = end
