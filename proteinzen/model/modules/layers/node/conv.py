"""
Convolutional operators on sequence embeddings

Largely inspired by https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
"""
import torch
from torch import nn
import torch.nn.functional as F

def swish(x):
    return x * torch.sigmoid(x)

class Downsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=0
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        pad = (0, 2)
        x = F.pad(
            x,
            pad,
            mode="constant",
            value=0
        )
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="nearest"
        )
        x = self.conv(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        dropout=0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if out_channels is None:
            out_channels = in_channels

        self.norm1 = nn.GroupNorm(
            num_groups=in_channels,
            num_channels=in_channels,
            eps=1e-6,
            affine=True
        )
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm2 = nn.GroupNorm(
            num_groups=out_channels,
            num_channels=out_channels,
            eps=1e-6,
            affine=True
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return x + h


class SequenceDownscaler(nn.Module):
    def __init__(
        self,
        n_layers,
        c_in,
        c_out,
        dropout=0.1
    ):
        super().__init__()
        assert c_in // c_out == 2 ** n_layers
        self.n_layers = n_layers

        resolutions = [int(c_in // (2**i)) for i in range(n_layers + 1)]

        self.blocks = nn.ModuleDict()
        for i in range(n_layers):
            self.blocks[f"resnet_{i}"] = ResNetBlock(
                in_channels=resolutions[i],
                out_channels=resolutions[i],
                dropout=dropout
            )
            self.blocks[f"downsample_{i}"] = Downsample(
                in_channels=resolutions[i],
                out_channels=resolutions[i+1],
            )
        self.resnet_out = ResNetBlock(
            c_out,
            c_out,
            dropout=dropout
        )

    def forward(self, seq_embedding, is_atom=None):
        x = seq_embedding.clone()
        if is_atom is not None:
            x[is_atom] = 0
        x = x.transpose(-1, -2)
        for i in range(self.n_layers):
            x = self.blocks[f"resnet_{i}"](x)
            x = self.blocks[f"downsample_{i}"](x)

        x = self.resnet_out(x)
        return x.transpose(-1, -2)


class SequenceUpscaler(nn.Module):
    def __init__(
        self,
        n_layers,
        c_in,
        c_out,
        dropout=0.1
    ):
        super().__init__()
        assert c_out // c_in == 2 ** n_layers
        self.n_layers = n_layers

        resolutions = [int(c_in * (2**i)) for i in range(n_layers + 1)]

        self.blocks = nn.ModuleDict()
        for i in range(n_layers):
            self.blocks[f"resnet_{i}"] = ResNetBlock(
                in_channels=resolutions[i],
                out_channels=resolutions[i],
                dropout=dropout
            )
            self.blocks[f"upsample_{i}"] = Upsample(
                in_channels=resolutions[i],
                out_channels=resolutions[i+1],
            )
        self.resnet_out = ResNetBlock(
            c_out,
            c_out,
            dropout=dropout
        )

    def forward(self, seq_embedding, seq_len, is_atom=None):
        x = seq_embedding.transpose(-1, -2)
        for i in range(self.n_layers):
            x = self.blocks[f"resnet_{i}"](x)
            x = self.blocks[f"upsample_{i}"](x)

        x = self.resnet_out(x)
        x = x.transpose(-1, -2)
        x = x[:, :seq_len]

        if is_atom is not None:
            x[is_atom] = 0
        return x