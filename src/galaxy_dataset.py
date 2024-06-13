from typing import Optional, cast
from collections.abc import Callable
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

Transformation = Callable[[torch.Tensor], torch.Tensor]


class GalaxyDataset(Dataset):
    def __init__(
        self,
        dataset_file: Path,
        transform_samples: Optional[Transformation] = None,
        transform_labels: Optional[Transformation] = None,
    ) -> None:
        super().__init__()
        self._dataset_file = dataset_file
        self._transform_samples = transform_samples
        self._transform_labels = transform_labels
        self._file_handler = h5py.File(self._dataset_file, "r")

    def __len__(self) -> int:
        return self._get_classes().shape[0]

    def __getitem__(self, index: int | list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        sorted_index, args = index, None
        if isinstance(index, list):
            sorted_index, args = torch.sort(torch.tensor(index))

        samples = torch.tensor(self._get_images()[sorted_index], dtype=torch.float32)
        labels = torch.tensor(self._get_classes()[sorted_index], dtype=torch.int64)

        if args is not None:
            sorted_args, _ = torch.sort(args)
            samples = samples[sorted_args]
            labels = labels[sorted_args]

        samples = torch.clamp(samples / 255.0, 0, 1)

        if self._transform_samples is not None:
            samples = self._transform_samples(samples)

        if self._transform_labels is not None:
            labels = self._transform_labels(labels)

        return samples, labels

    def _get_images(self) -> h5py.Dataset:
        return cast(h5py.Dataset, self._file_handler["images"])

    def _get_classes(self) -> h5py.Dataset:
        return cast(h5py.Dataset, self._file_handler["classes"])
