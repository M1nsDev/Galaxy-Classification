import torch
from torchvision.transforms import v2, InterpolationMode
from torch.utils.data import Dataset
import random
import math

transform = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomRotation(30, interpolation=InterpolationMode.BILINEAR),
    v2.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
])

class AugmentedDataset(Dataset):
    def __init__(self, original_dataset, augmentation_factor=0.3, augment_classes=None):
        self.original_dataset = original_dataset
        self.augmentation_factor = augmentation_factor
        self.transform = transform
        self.augment_classes = augment_classes if augment_classes is not None else []

        self.num_originals = len(self.original_dataset)
        self.num_augmentations = math.ceil(self.augmentation_factor * self.num_originals)

        self._create_augmented_indices()

    def _create_augmented_indices(self):
        self.augmented_indices = []
        augment_count = 0

        while augment_count < self.num_augmentations:
            idx = random.randint(0, self.num_originals - 1)
            _, label = self.original_dataset[idx]
            if label in self.augment_classes:
                self.augmented_indices.append(idx)
                augment_count += 1

    def __len__(self):
        return self.num_originals + len(self.augmented_indices)

    def __getitem__(self, idx):
        if idx < self.num_originals:
            image, label = self.original_dataset[idx]
            image = v2.ToPILImage()(image)
        else:
            original_idx = self.augmented_indices[idx - self.num_originals]
            image, label = self.original_dataset[original_idx]
            image = v2.ToPILImage()(image)
            image = self.transform(image)
        
        image = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(image)
        return image, label