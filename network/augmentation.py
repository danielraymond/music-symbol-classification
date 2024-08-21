import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from config.data_config import INPUT_SIZE
from custom_augmentation import transformations as custom_transformations


class AugmentStage(nn.Module):
    def __init__(self):
        super(AugmentStage, self).__init__()
        self.transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    INPUT_SIZE,
                    scale=(0.5, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomApply(
                    [custom_transformations.SaltAndPepperNoise()], p=0.4
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=23)], p=0.4
                ),
                transforms.RandomApply(
                    [custom_transformations.Fade(fade_factor=0.6)], p=0.4
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.stack([self.transforms(image) for image in x])


class UpdatedAugmentStage(nn.Module):
    def __init__(self):
        super(UpdatedAugmentStage, self).__init__()
        self.transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    INPUT_SIZE,
                    scale=(0.5, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomApply(
                    [custom_transformations.SaltAndPepperNoise()], p=0.4
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=23)], p=0.4
                ),
                transforms.RandomApply(
                    [custom_transformations.Fade(fade_factor=0.6)], p=0.4
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.stack([self.transforms(image) for image in x])


class MildAugmentStage(nn.Module):
    def __init__(self):
        super(MildAugmentStage, self).__init__()
        self.transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    INPUT_SIZE,
                    scale=(0.8, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=15)], p=0.5
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.stack([self.transforms(img) for img in x])
