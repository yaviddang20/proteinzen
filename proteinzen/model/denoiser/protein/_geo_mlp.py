# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Factory functions for simple MLPs for multivector data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple, Union

import torch
from torch import nn

from gatr.layers.dropout import GradeDropout
from gatr.layers.linear import EquiLinear
from gatr.layers.mlp.geometric_bilinears import GeometricBilinear
from gatr.layers.mlp.nonlinearities import ScalarGatedNonlinearity

@dataclass
class MLPConfig:
    """Geometric MLP configuration.

    Parameters
    ----------
    mv_channels : iterable of int
        Number of multivector channels at each layer, from input to output
    s_channels : None or iterable of int
        If not None, sets the number of scalar channels at each layer, from input to output. Length
        needs to match mv_channels
    activation : {"relu", "sigmoid", "gelu"}
        Which (gated) activation function to use
    dropout_prob : float or None
        Dropout probability
    """

    mv_channels: Optional[List[int]] = None
    s_channels: Optional[List[int]] = None
    activation: str = "gelu"
    dropout_prob: Optional[float] = None
    output_init: Optional[str] = "default"

    def __post_init__(self):
        """Type checking / conversion."""
        if isinstance(self.dropout_prob, str) and self.dropout_prob.lower() in ["null", "none"]:
            self.dropout_prob = None

    @classmethod
    def cast(cls, config: Any) -> MLPConfig:
        """Casts an object as MLPConfig."""
        if isinstance(config, MLPConfig):
            return config
        if isinstance(config, Mapping):
            return cls(**config)
        raise ValueError(f"Can not cast {config} to {cls}")


class GeoMLP(nn.Module):
    """Geometric MLP.

    This is a core component of GATr's transformer blocks. It is similar to a regular MLP, except
    that it uses geometric bilinears (GP and equivariant join) in place of the first linear layer.

    Assumes input has shape `(..., channels[0], 16)`, output has shape `(..., channels[-1], 16)`,
    will create hidden layers with shape `(..., channel, 16)` for each additional entry in
    `channels`.

    Parameters
    ----------
    config: MLPConfig
        Configuration object
    """

    def __init__(
        self,
        config: MLPConfig,
    ) -> None:
        super().__init__()

        # Store settings
        self.config = config

        assert config.mv_channels is not None
        s_channels = (
            [None for _ in config.mv_channels] if config.s_channels is None else config.s_channels
        )

        layers: List[nn.Module] = []

        if len(config.mv_channels) >= 2:
            layers.append(
                GeometricBilinear(
                    in_mv_channels=config.mv_channels[0],
                    out_mv_channels=config.mv_channels[1],
                    in_s_channels=s_channels[0],
                    out_s_channels=s_channels[1],
                )
            )
            if config.dropout_prob is not None:
                layers.append(GradeDropout(config.dropout_prob))

            num_blocks = len(s_channels[2:])

            for b, (in_, out, in_s, out_s) in enumerate(zip(
                config.mv_channels[1:-1], config.mv_channels[2:], s_channels[1:-1], s_channels[2:]
            )):
                layers.append(ScalarGatedNonlinearity(config.activation))
                if b == num_blocks - 1:
                    layers.append(EquiLinear(in_, out, in_s_channels=in_s, out_s_channels=out_s, initialization=config.output_init))
                else:
                    layers.append(EquiLinear(in_, out, in_s_channels=in_s, out_s_channels=out_s))
                if config.dropout_prob is not None:
                    layers.append(GradeDropout(config.dropout_prob))

        self.layers = nn.ModuleList(layers)

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor, reference_mv: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Forward pass.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors.
        scalars : None or torch.Tensor with shape (..., in_s_channels)
            Optional input scalars.
        reference_mv : torch.Tensor with shape (..., 16)
            Reference multivector for equivariant join.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., out_mv_channels, 16)
            Output multivectors.
        outputs_s : None or torch.Tensor with shape (..., out_s_channels)
            Output scalars, if scalars are provided. Otherwise None.
        """

        mv, s = multivectors, scalars

        for i, layer in enumerate(self.layers):
            if i == 0:
                mv, s = layer(mv, scalars=s, reference_mv=reference_mv)
            else:
                mv, s = layer(mv, scalars=s)

        return mv, s
