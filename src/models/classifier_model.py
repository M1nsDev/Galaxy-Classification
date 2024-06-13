from typing import Protocol
import torch


class _ConvBlockFactory(Protocol):
    def __call__(
        self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int
    ) -> torch.nn.Module: ...


class _ModelFrame(torch.nn.Module):
    def __init__(self, conv_block_factory: _ConvBlockFactory) -> None:
        super().__init__()
        conv_args = dict(kernel_size=5, pool_size=2)
        self.conv0 = conv_block_factory(in_channels=3, out_channels=16, **conv_args)
        self.conv1 = conv_block_factory(in_channels=16, out_channels=32, **conv_args)
        self.conv2 = conv_block_factory(in_channels=32, out_channels=64, **conv_args)
        self.flat = torch.nn.Flatten()
        self.drop = torch.nn.Dropout(0.5)
        self.clf = torch.nn.Linear(in_features=1600, out_features=10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv0(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flat(x)
        x = self.drop(x)
        return self.clf(x)


class GalaxyClassifierModel(_ModelFrame):
    def __init__(self) -> None:
        super().__init__(conv_block_factory=_dense_conv_block)


def _dense_conv_block(
    in_channels: int, out_channels: int, kernel_size: int, pool_size: int
) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=False,
        ),
        torch.nn.BatchNorm2d(num_features=out_channels),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=pool_size),
    )