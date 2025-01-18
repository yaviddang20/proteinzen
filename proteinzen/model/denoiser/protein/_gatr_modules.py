# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Self-attention layers."""

from dataclasses import dataclass, replace
from typing import Literal, Optional, Sequence, Tuple, Union, Callable, Mapping, Any
from warnings import warn

import torch
from einops import rearrange
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint as checkpoint_
from xformers.ops import AttentionBias

from gatr.layers.attention.attention import GeometricAttention
# from gatr.layers.attention.config import SelfAttentionConfig
from gatr.layers.attention.positional_encoding import ApplyRotaryPositionalEncoding
from gatr.layers.attention.qkv import MultiQueryQKVModule, QKVModule
from gatr.layers.dropout import GradeDropout
from gatr.layers.linear import EquiLinear
from gatr.layers.layer_norm import EquiLayerNorm
from gatr.primitives.bilinear import geometric_product
from gatr.primitives.dual import equivariant_join
from gatr.primitives.nonlinearities import gated_sigmoid
from gatr.utils.tensors import construct_reference_multivector

from proteinzen.model.modules.openfold.layers_v2 import Linear, LayerNorm, permute_final_dims, flatten_final_dims
from ._geo_mlp import MLPConfig, GeoMLP
from ._gatr_primitives import weighted_geometric_product, weighted_equivariant_join


def _flatten(tensor, start_dim, end_dim):
    """ Flatten dims of a tensor and return a view """
    new_shape = [*tensor.shape[:start_dim], -1, *tensor.shape[end_dim+1:]]
    return tensor.view(new_shape)


@dataclass
class SelfAttentionConfig:
    """Configuration for attention.

    Parameters
    ----------
    in_mv_channels : int
        Number of input multivector channels.
    out_mv_channels : int
        Number of output multivector channels.
    num_heads : int
        Number of attention heads.
    in_s_channels : int
        Input scalar channels. If None, no scalars are expected nor returned.
    out_s_channels : int
        Output scalar channels. If None, no scalars are expected nor returned.
    additional_qk_mv_channels : int
        Whether additional multivector features for the keys and queries will be provided.
    additional_qk_s_channels : int
        Whether additional scalar features for the keys and queries will be provided.
    normalizer : str
        Normalizer function to use in sdp_dist attention
    normalizer_eps : float
        Small umerical constant for stability in the normalizer in sdp_dist attention
    multi_query: bool
        Whether to do multi-query attention
    attention_type : {"scalar", "geometric", "sdp_dist"}
        Whether the attention mechanism is based on the scalar product or also the join.
    pos_encoding : bool
        Whether to apply rotary positional embeddings along the item dimension to the scalar keys
        and queries.
    pos_enc_base : int
        Base for the frequencies in the positional encoding.
    output_init : str
        Initialization scheme for final linear layer
    increase_hidden_channels : int
        Factor by which to increase the number of hidden channels (both multivectors and scalars)
    dropout_prob : float or None
        Dropout probability
    """

    multi_query: bool = True
    in_mv_channels: Optional[int] = None
    out_mv_channels: Optional[int] = None
    in_s_channels: Optional[int] = None
    out_s_channels: Optional[int] = None
    num_heads: int = 8
    additional_qk_mv_channels: int = 0
    additional_qk_s_channels: int = 0
    normalizer_eps: Optional[float] = 1e-3
    pos_encoding: bool = False
    pos_enc_base: int = 4096
    output_init: str = "default"
    checkpoint: bool = True
    increase_hidden_mv_channels: int = 2
    increase_hidden_s_channels: int = 2
    dropout_prob: Optional[float] = None

    def __post_init__(self):
        """Type checking / conversion."""
        if isinstance(self.dropout_prob, str) and self.dropout_prob.lower() in ["null", "none"]:
            self.dropout_prob = None

    @property
    def hidden_mv_channels(self) -> Optional[int]:
        """Returns the number of hidden multivector channels."""

        if self.in_mv_channels is None:
            return None

        return max(self.increase_hidden_mv_channels * self.in_mv_channels // self.num_heads, 1)

    @property
    def hidden_s_channels(self) -> Optional[int]:
        """Returns the number of hidden scalar channels."""

        if self.in_s_channels is None:
            return None

        hidden_s_channels = max(
            self.increase_hidden_s_channels * self.in_s_channels // self.num_heads, 4
        )

        # When using positional encoding, the number of scalar hidden channels needs to be even.
        # It also should not be too small.
        if self.pos_encoding:
            hidden_s_channels = (hidden_s_channels + 1) // 2 * 2
            hidden_s_channels = max(hidden_s_channels, 8)

        return hidden_s_channels

    @classmethod
    def cast(cls, config: Any):
        """Casts an object as SelfAttentionConfig."""
        if isinstance(config, SelfAttentionConfig):
            return config
        if isinstance(config, Mapping):
            return cls(**config)
        raise ValueError(f"Can not cast {config} to {cls}")



class EquiAdaLN(nn.Module):
    """ An equivariant adaptive LayerNorm inspired by AdaLN

    Args:
    """
    def __init__(
        self,
        mv_channels,
        scalar_channels,
        cond_mv_channels,
        cond_scalar_channels
    ):
        super().__init__()
        self.ln = EquiLayerNorm()

        self.cond_mv_gate = Linear(cond_scalar_channels, mv_channels)
        self.cond_scalars_gate = Linear(cond_scalar_channels, scalar_channels)
        self.lin_cond_no_bias = EquiLinear(
            in_mv_channels=cond_mv_channels,
            out_mv_channels=mv_channels,
            bias=False,
            in_s_channels=cond_scalar_channels,
            out_s_channels=scalar_channels
        )

    def forward(
        self,
        mv,
        s,
        cond_mv,
        cond_s
    ):
        mv, s = self.ln(mv, s)
        cond_mv, cond_s = self.ln(cond_mv, cond_s)

        gated_mv = gated_sigmoid(
            mv,
            gates=self.cond_mv_gate(cond_s)[..., None]
        )
        gated_s = torch.sigmoid(self.cond_scalars_gate(cond_s)) * s
        bias_mv, bias_s = self.lin_cond_no_bias(cond_mv, cond_s)

        return gated_mv + bias_mv, gated_s + bias_s


class TwoInputGeometricBilinear(nn.Module):
    """Geometric bilinear layer.

    Pin-equivariant map between multivector tensors that constructs new geometric features via
    geometric products and the equivariant join (based on a reference vector).

    Parameters
    ----------
    in_mv_channels : int
        Input multivector channels of `x`
    out_mv_channels : int
        Output multivector channels
    hidden_mv_channels : int or None
        Hidden MV channels. If None, uses out_mv_channels.
    in_s_channels : int or None
        Input scalar channels of `x`. If None, no scalars are expected nor returned.
    out_s_channels : int or None
        Output scalar channels. If None, no scalars are expected nor returned.
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        hidden_mv_channels: Optional[int] = None,
        in_s_channels: Optional[int] = None,
        out_s_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Default options
        if hidden_mv_channels is None:
            hidden_mv_channels = out_mv_channels

        out_mv_channels_each = hidden_mv_channels // 2
        assert (
            out_mv_channels_each * 2 == hidden_mv_channels
        ), "GeometricBilinear needs even channel number"

        # Linear projections for GP
        self.linear_left = EquiLinear(
            in_mv_channels,
            out_mv_channels_each,
            in_s_channels=in_s_channels,
            out_s_channels=None,
        )
        self.linear_right = EquiLinear(
            in_mv_channels,
            out_mv_channels_each,
            in_s_channels=in_s_channels,
            out_s_channels=None,
            initialization="almost_unit_scalar",
        )

        # Linear projections for join
        self.linear_join_left = EquiLinear(
            in_mv_channels, out_mv_channels_each, in_s_channels=in_s_channels, out_s_channels=None
        )
        self.linear_join_right = EquiLinear(
            in_mv_channels, out_mv_channels_each, in_s_channels=in_s_channels, out_s_channels=None
        )

        # Output linear projection
        self.linear_out = EquiLinear(
            hidden_mv_channels,
            out_mv_channels,
            2*in_s_channels if in_s_channels is not None else None,
            out_s_channels
        )

    def forward(
        self,
        multivectors_1: torch.Tensor,
        multivectors_2: torch.Tensor,
        scalars_1: torch.Tensor,
        scalars_2: torch.Tensor,
        reference_mv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors
        scalars : torch.Tensor with shape (..., in_s_channels)
            Input scalars
        reference_mv : torch.Tensor with shape (..., 16)
            Reference multivector for equivariant join.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., self.out_mv_channels, 16)
            Output multivectors
        output_s : None or torch.Tensor with shape (..., out_s_channels)
            Output scalars.
        """

        # GP
        left, _ = self.linear_left(multivectors_1, scalars=scalars_1)
        right, _ = self.linear_right(multivectors_2, scalars=scalars_2)
        gp_outputs = geometric_product(left, right)

        # Equivariant join
        left, _ = self.linear_join_left(multivectors_1, scalars=scalars_1)
        right, _ = self.linear_join_right(multivectors_2, scalars=scalars_2)
        join_outputs = equivariant_join(left, right, reference_mv)

        # Output linear
        outputs_mv = torch.cat((gp_outputs, join_outputs), dim=-2)
        outputs_mv, outputs_s = self.linear_out(
            outputs_mv,
            scalars=torch.cat([scalars_1, scalars_2], dim=-1)
        )

        return outputs_mv, outputs_s


class GeometricManyBodyProduct(nn.Module):
    def __init__(self,
                 config: SelfAttentionConfig):
        super().__init__()

        self.config = config
        self.mv_channels = config.hidden_mv_channels

        self.left_lin = EquiLinear(
            in_mv_channels=self.mv_channels,
            out_mv_channels=self.mv_channels,
            in_s_channels=config.hidden_s_channels
        )
        self.right_lin = EquiLinear(
            in_mv_channels=self.mv_channels,
            out_mv_channels=self.mv_channels,
            in_s_channels=config.hidden_s_channels
        )

        self.gp_weight = nn.Parameter(
            torch.zeros((self.mv_channels, 16, 16, 16))
        )
        self.join_weight = nn.Parameter(
            torch.zeros((self.mv_channels, 16, 16, 16))
        )

    def forward(self, o_mv, o_s, reference):
        x_mv, _ = self.left_lin(o_mv, o_s)
        y_mv, _ = self.right_lin(o_mv, o_s)

        out_mv = (
            weighted_geometric_product(self.gp_weight, x_mv, y_mv)
            + weighted_equivariant_join(self.join_weight, x_mv, y_mv, reference)
            + y_mv
        )
        return out_mv


class SelfAttentionPairBias(nn.Module):
    """Geometric self-attention layer.

    Constructs queries, keys, and values, computes attention, and projects linearly to outputs.

    Parameters
    ----------
    config : SelfAttentionConfig
        Attention configuration.
    """

    def __init__(self,
                 config: SelfAttentionConfig,
                 c_z,
                 many_body_expand=False,
                 inf=1e5
    ) -> None:
        super().__init__()

        # Store settings
        self.config = config
        self.inf = inf

        # QKV computation
        self.qkv_module = MultiQueryQKVModule(config) if config.multi_query else QKVModule(config)

        # Output projection
        self.out_linear = EquiLinear(
            in_mv_channels=config.hidden_mv_channels * config.num_heads,
            out_mv_channels=config.out_mv_channels,
            in_s_channels=(
                None
                if config.in_s_channels is None
                else config.hidden_s_channels * config.num_heads
            ),
            out_s_channels=config.out_s_channels,
            initialization=config.output_init,
        )

        # Optional positional encoding
        self.pos_encoding: nn.Module
        if config.pos_encoding:
            self.pos_encoding = ApplyRotaryPositionalEncoding(
                config.hidden_s_channels, item_dim=-2, base=config.pos_enc_base
            )
        else:
            self.pos_encoding = nn.Identity()

        # Attention
        self.attention = GeometricAttention(config)
        self.pair_bias = nn.Sequential(
            LayerNorm(c_z),
            Linear(c_z, config.num_heads, bias=False)
        )

        # Dropout
        self.dropout: Optional[nn.Module]
        if config.dropout_prob is not None:
            self.dropout = GradeDropout(config.dropout_prob)
        else:
            self.dropout = None

        self.many_body_expand = many_body_expand
        if self.many_body_expand:
            self.bilinear = TwoInputGeometricBilinear(
                in_mv_channels=config.hidden_mv_channels,
                out_mv_channels=config.hidden_mv_channels,
                in_s_channels=config.hidden_s_channels,
                out_s_channels=config.hidden_s_channels
            )
            self.mbp = GeometricManyBodyProduct(config)
        else:
            self.bilinear = None
            self.mbp = None

    def forward(
        self,
        multivectors: torch.Tensor,
        pair_scalars: torch.Tensor,
        additional_qk_features_mv: Optional[torch.Tensor] = None,
        scalars: Optional[torch.Tensor] = None,
        additional_qk_features_s: Optional[torch.Tensor] = None,
        attention_mask=None,
        reference_mv=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes forward pass on inputs with shape `(..., items, channels, 16)`.

        The result is the following:

        ```
        # For each head
        queries = linear_channels(inputs)
        keys = linear_channels(inputs)
        values = linear_channels(inputs)
        hidden = attention_items(queries, keys, values, biases=biases)
        head_output = linear_channels(hidden)

        # Combine results
        output = concatenate_heads head_output
        ```

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., num_items, channels_in, 16)
            Input multivectors.
        additional_qk_features_mv : None or torch.Tensor with shape
            (..., num_items, add_qk_mv_channels, 16)
            Additional Q/K features, multivector part.
        scalars : None or torch.Tensor with shape (..., num_items, num_items, in_scalars)
            Optional input scalars
        additional_qk_features_s : None or torch.Tensor with shape
            (..., num_items, add_qk_mv_channels, 16)
            Additional Q/K features, scalar part.
        scalars : None or torch.Tensor with shape (..., num_items, num_items, in_scalars)
            Optional input scalars
        attention_mask: None or torch.Tensor with shape (..., num_items, num_items)
            Optional attention mask

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., num_items, channels_out, 16)
            Output multivectors.
        output_scalars : torch.Tensor with shape (..., num_items, channels_out, out_scalars)
            Output scalars, if scalars are provided. Otherwise None.
        """
        # Compute Q, K, V
        q_mv, k_mv, v_mv, q_s, k_s, v_s = self.qkv_module(
            multivectors, scalars, additional_qk_features_mv, additional_qk_features_s
        )

        # Rotary positional encoding
        q_s = self.pos_encoding(q_s)
        k_s = self.pos_encoding(k_s)

        attn_bias = self.pair_bias(pair_scalars)
        if attention_mask is not None:
            attn_bias = attn_bias + (1 - attention_mask[..., None].float()) * self.inf
        attn_bias = permute_final_dims(attn_bias, (2, 0, 1))

        # Attention layer
        h_mv, h_s = self.attention(q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=attn_bias)

        if self.many_body_expand:
            assert reference_mv is not None
            h_mv, _h_s = self.bilinear(h_mv, v_mv, h_s, v_s, reference_mv[:, None])
            h_mv = self.mbp(h_mv, _h_s, reference_mv[:, None])

        h_mv = rearrange(
            h_mv, "... n_heads n_items hidden_channels x -> ... n_items (n_heads hidden_channels) x"
        )
        h_s = rearrange(
            h_s, "... n_heads n_items hidden_channels -> ... n_items (n_heads hidden_channels)"
        )

        # Transform linearly one more time
        outputs_mv, outputs_s = self.out_linear(h_mv, scalars=h_s)

        # Dropout
        if self.dropout is not None:
            outputs_mv, outputs_s = self.dropout(outputs_mv, outputs_s)

        return outputs_mv, outputs_s


class GATrPairBiasBlock(nn.Module):
    """Equivariant transformer block for GATr.

    This is the biggest building block of GATr.

    Inputs are first processed by a block consisting of LayerNorm, multi-head geometric
    self-attention, and a residual connection. Then the data is processed by a block consisting of
    another LayerNorm, an item-wise two-layer geometric MLP with GeLU activations, and another
    residual connection.

    Parameters
    ----------
    mv_channels : int
        Number of input and output multivector channels
    s_channels: int
        Number of input and output scalar channels
    attention: SelfAttentionConfig
        Attention configuration
    mlp: MLPConfig
        MLP configuration
    dropout_prob : float or None
        Dropout probability
    checkpoint : None or sequence of "mlp", "attention"
        Which components to apply gradient checkpointing to
    """

    def __init__(
        self,
        mv_channels: int,
        s_channels: int,
        z_channels: int,
        attention: SelfAttentionConfig,
        mlp: MLPConfig,
        dropout_prob: Optional[float] = None,
        checkpoint: Optional[Sequence[Literal["mlp", "attention"]]] = None,
        many_body_expand=False
    ) -> None:
        super().__init__()

        # Gradient checkpointing settings
        if checkpoint is not None:
            for key in checkpoint:
                assert key in ["mlp", "attention"]
        self._checkpoint_mlp = checkpoint is not None and "mlp" in checkpoint
        self._checkpoint_attn = checkpoint is not None and "attention" in checkpoint

        # Normalization layer (stateless, so we can use the same layer for both normalization
        # instances)
        self.norm = EquiLayerNorm()

        # Self-attention layer
        attention = replace(
            attention,
            in_mv_channels=mv_channels,
            out_mv_channels=mv_channels,
            in_s_channels=s_channels,
            out_s_channels=s_channels,
            output_init="small",
            dropout_prob=dropout_prob,
        )
        self.attention = SelfAttentionPairBias(
            attention,
            c_z=z_channels,
            many_body_expand=many_body_expand,
        )

        # MLP block
        mlp = replace(
            mlp,
            mv_channels=(mv_channels, 2 * mv_channels, mv_channels),
            s_channels=(s_channels, 2 * s_channels, s_channels),
            dropout_prob=dropout_prob,
        )
        self.mlp = GeoMLP(mlp)

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
        pair_scalars: torch.Tensor,
        reference_mv: Optional[torch.Tensor] = None,
        additional_qk_features_mv: Optional[torch.Tensor] = None,
        additional_qk_features_s: Optional[torch.Tensor] = None,
        attention_mask: Optional[Union[AttentionBias, torch.Tensor]] = None,
        node_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the transformer block.

        Inputs are first processed by a block consisting of LayerNorm, multi-head geometric
        self-attention, and a residual connection. Then the data is processed by a block consisting
        of another LayerNorm, an item-wise two-layer geometric MLP with GeLU activations, and
        another residual connection.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., items, channels, 16)
            Input multivectors.
        scalars : torch.Tensor with shape (..., s_channels)
            Input scalars.
        reference_mv : torch.Tensor with shape (..., 16) or None
            Reference multivector for the equivariant join operation in the MLP.
        additional_qk_features_mv : None or torch.Tensor with shape
            (..., num_items, add_qk_mv_channels, 16)
            Additional Q/K features, multivector part.
        additional_qk_features_s : None or torch.Tensor with shape
            (..., num_items, add_qk_mv_channels, 16)
            Additional Q/K features, scalar part.
        attention_mask: None or torch.Tensor or AttentionBias
            Optional attention mask.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., items, channels, 16).
            Output multivectors
        output_scalars : torch.Tensor with shape (..., s_channels)
            Output scalars
        """
        mv_mask = (node_mask[..., None, None] if node_mask is not None else 1)
        s_mask = (node_mask[..., None] if node_mask is not None else 1)

        # Attention block
        attn_kwargs = dict(
            multivectors=multivectors,
            scalars=scalars,
            pair_scalars=pair_scalars,
            additional_qk_features_mv=additional_qk_features_mv,
            additional_qk_features_s=additional_qk_features_s,
            attention_mask=attention_mask,
            reference_mv=reference_mv,
        )
        if self._checkpoint_attn:
            h_mv, h_s = checkpoint_(self._attention_block, use_reentrant=False, **attn_kwargs)
        else:
            h_mv, h_s = self._attention_block(**attn_kwargs)

        # Skip connection
        outputs_mv = multivectors + h_mv * mv_mask
        outputs_s = scalars + h_s * s_mask

        # MLP block
        mlp_kwargs = dict(multivectors=outputs_mv, scalars=outputs_s, reference_mv=reference_mv)
        if self._checkpoint_mlp:
            h_mv, h_s = checkpoint_(self._mlp_block, use_reentrant=False, **mlp_kwargs)
        else:
            h_mv, h_s = self._mlp_block(outputs_mv, scalars=outputs_s, reference_mv=reference_mv)

        # Skip connection
        outputs_mv = outputs_mv + h_mv * mv_mask
        outputs_s = outputs_s + h_s * s_mask

        return outputs_mv, outputs_s

    def _attention_block(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
        pair_scalars: torch.Tensor,
        additional_qk_features_mv: Optional[torch.Tensor] = None,
        additional_qk_features_s: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        reference_mv = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attention block."""

        h_mv, h_s = self.norm(multivectors, scalars=scalars)
        h_mv, h_s = self.attention(
            h_mv,
            scalars=h_s,
            pair_scalars=pair_scalars,
            additional_qk_features_mv=additional_qk_features_mv,
            additional_qk_features_s=additional_qk_features_s,
            attention_mask=attention_mask,
            reference_mv=reference_mv
        )
        return h_mv, h_s

    def _mlp_block(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
        reference_mv: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MLP block."""

        h_mv, h_s = self.norm(multivectors, scalars=scalars)
        h_mv, h_s = self.mlp(h_mv, scalars=h_s, reference_mv=reference_mv)
        return h_mv, h_s


class GATrPairBias(nn.Module):
    """GATr network for a data with a single token dimension.

    This, together with gatr.nets.axial_gatr.AxialGATr, is the main architecture proposed in our
    paper.

    It combines `num_blocks` GATr transformer blocks, each consisting of geometric self-attention
    layers, a geometric MLP, residual connections, and normalization layers. In addition, there
    are initial and final equivariant linear layers.

    Assumes input has shape `(..., items, in_channels, 16)`, output has shape
    `(..., items, out_channels, 16)`, will create hidden representations with shape
    `(..., items, hidden_channels, 16)`.

    Parameters
    ----------
    in_mv_channels : int
        Number of input multivector channels.
    out_mv_channels : int
        Number of output multivector channels.
    hidden_mv_channels : int
        Number of hidden multivector channels.
    in_s_channels : None or int
        If not None, sets the number of scalar input channels.
    out_s_channels : None or int
        If not None, sets the number of scalar output channels.
    hidden_s_channels : None or int
        If not None, sets the number of scalar hidden channels.
    attention: Dict
        Data for SelfAttentionConfig
    mlp: Dict
        Data for MLPConfig
    num_blocks : int
        Number of transformer blocks.
    checkpoint_blocks : bool
        Deprecated option to specify gradient checkpointing. Use `checkpoint=["block"]` instead
    dropout_prob : float or None
        Dropout probability
    checkpoint : None or sequence of "mlp", "attention", "block"
        Which components to apply gradient checkpointing to
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        hidden_mv_channels: int,
        in_s_channels: Optional[int],
        out_s_channels: Optional[int],
        hidden_s_channels: Optional[int],
        c_z: int,
        attention: SelfAttentionConfig,
        mlp: MLPConfig,
        num_blocks: int = 10,
        reinsert_mv_channels: Optional[Tuple[int]] = None,
        reinsert_s_channels: Optional[Tuple[int]] = None,
        checkpoint_blocks: bool = False,
        dropout_prob: Optional[float] = None,
        checkpoint: Union[
            None, Sequence[Literal["block"]], Sequence[Literal["mlp", "attention"]]
        ] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Gradient checkpointing settings
        if checkpoint_blocks:
            # The checkpoint_blocks keyword was deprecated in v1.4.0.
            if checkpoint is not None:
                raise ValueError(
                    "Both checkpoint_blocks and checkpoint were specified. Please only use"
                    "checkpoint."
                )
            warn(
                'The checkpoint_blocks keyword is deprecated since v1.4.0. Use checkpoint=["block"]'
                "instead.",
                category=DeprecationWarning,
            )
            checkpoint = ["block"]
        if checkpoint is not None:
            for key in checkpoint:
                assert key in ["block", "mlp", "attention"]
        if checkpoint is not None and "block" in checkpoint:
            self._checkpoint_blocks = True
            if "mlp" in checkpoint or "attention" in checkpoint:
                raise ValueError(
                    "Checkpointing both on the block level and the MLP / attention"
                    'level is not sensible. Please use either checkpoint=["block"] or '
                    f'checkpoint=["attention", "mlp"]. Found checkpoint={checkpoint}.'
                )
            checkpoint = None
        else:
            self._checkpoint_blocks = False

        self.linear_in = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=hidden_s_channels,
        )
        attention = replace(
            SelfAttentionConfig.cast(attention),  # convert duck typing to actual class
            additional_qk_mv_channels=(
                0 if reinsert_mv_channels is None else len(reinsert_mv_channels)
            ),
            additional_qk_s_channels=0 if reinsert_s_channels is None else len(reinsert_s_channels),
        )
        mlp = MLPConfig.cast(mlp)
        self.blocks = nn.ModuleList(
            [
                GATrPairBiasBlock(
                    mv_channels=hidden_mv_channels,
                    s_channels=hidden_s_channels,
                    z_channels=c_z,
                    attention=attention,
                    mlp=mlp,
                    dropout_prob=dropout_prob,
                    checkpoint=checkpoint,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = EquiLinear(
            hidden_mv_channels,
            out_mv_channels,
            in_s_channels=hidden_s_channels,
            out_s_channels=out_s_channels,
        )
        self._reinsert_s_channels = reinsert_s_channels
        self._reinsert_mv_channels = reinsert_mv_channels

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
        pair_scalars: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        join_reference: Union[Tensor, str] = "data",
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Forward pass of the network.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors.
        scalars : None or torch.Tensor with shape (..., in_s_channels)
            Optional input scalars.
        attention_mask: None or torch.Tensor with shape (..., num_items, num_items)
            Optional attention mask
        join_reference : Tensor with shape (..., 16) or {"data", "canonical"}
            Reference multivector for the equivariant joint operation. If "data", a
            reference multivector is constructed from the mean of the input multivectors. If
            "canonical", a constant canonical reference multivector is used instead.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., out_mv_channels, 16)
            Output multivectors.
        outputs_s : None or torch.Tensor with shape (..., out_s_channels)
            Output scalars, if scalars are provided. Otherwise None.
        """

        # Reference multivector and channels that will be re-inserted in any query / key computation
        reference_mv = construct_reference_multivector(join_reference, multivectors)
        additional_qk_features_mv, additional_qk_features_s = self._construct_reinserted_channels(
            multivectors, scalars
        )

        # Pass through the blocks
        h_mv, h_s = self.linear_in(multivectors, scalars=scalars)
        for block in self.blocks:
            if self._checkpoint_blocks:
                h_mv, h_s = checkpoint_(
                    block,
                    h_mv,
                    use_reentrant=False,
                    scalars=h_s,
                    pair_scalars=pair_scalars,
                    reference_mv=reference_mv,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    attention_mask=attention_mask,
                )
            else:
                h_mv, h_s = block(
                    h_mv,
                    scalars=h_s,
                    pair_scalars=pair_scalars,
                    reference_mv=reference_mv,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    attention_mask=attention_mask,
                )

        outputs_mv, outputs_s = self.linear_out(h_mv, scalars=h_s)

        return outputs_mv, outputs_s

    def _construct_reinserted_channels(self, multivectors, scalars):
        """Constructs input features that will be reinserted in every attention layer."""

        if self._reinsert_mv_channels is None:
            additional_qk_features_mv = None
        else:
            additional_qk_features_mv = multivectors[..., self._reinsert_mv_channels, :]

        if self._reinsert_s_channels is None:
            additional_qk_features_s = None
        else:
            assert scalars is not None
            additional_qk_features_s = scalars[..., self._reinsert_s_channels]

        return additional_qk_features_mv, additional_qk_features_s



class CrossAttnQKVModule(nn.Module):
    """Compute (multivector and scalar) queries, keys, and values via multi-head attention.

    Parameters
    ----------
    config: SelfAttentionConfig
        Attention configuration
    """

    def __init__(self, config: SelfAttentionConfig):
        super().__init__()
        self.q_linear = EquiLinear(
            in_mv_channels=config.in_mv_channels + config.additional_qk_mv_channels,
            out_mv_channels=config.hidden_mv_channels * config.num_heads,
            in_s_channels=config.in_s_channels + config.additional_qk_s_channels,
            out_s_channels=None
            if config.in_s_channels is None
            else config.hidden_s_channels * config.num_heads,
        )
        self.kv_linear = EquiLinear(
            in_mv_channels=config.in_mv_channels + config.additional_qk_mv_channels,
            out_mv_channels=2 * config.hidden_mv_channels * config.num_heads,
            in_s_channels=config.in_s_channels + config.additional_qk_s_channels,
            out_s_channels=None
            if config.in_s_channels is None
            else 2 * config.hidden_s_channels * config.num_heads,
        )
        self.config = config
        self.num_mv_channels_in = config.in_mv_channels + config.additional_qk_mv_channels
        self.num_s_channels_in = config.in_s_channels + config.additional_qk_s_channels

    def forward(
        self,
        inputs,
        scalars,
        to_queries,
        to_keys,
        additional_qk_features_mv=None,
        additional_qk_features_s=None,
    ):
        """Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Multivector inputs
        scalars : torch.Tensor
            Scalar inputs
        additional_qk_features_mv : None or torch.Tensor
            Additional multivector features that should be provided for the Q/K computation (e.g.
            positions of objects)
        additional_qk_features_s : None or torch.Tensor
            Additional scalar features that should be provided for the Q/K computation (e.g.
            object types)

        Returns
        -------
        q_mv : Tensor with shape (..., num_items_out, num_mv_channels_in, 16)
            Queries, multivector part.
        k_mv : Tensor with shape (..., num_items_in, num_mv_channels_in, 16)
            Keys, multivector part.
        v_mv : Tensor with shape (..., num_items_in, num_mv_channels_out, 16)
            Values, multivector part.
        q_s : Tensor with shape (..., heads, num_items_out, num_s_channels_in)
            Queries, scalar part.
        k_s : Tensor with shape (..., heads, num_items_in, num_s_channels_in)
            Keys, scalar part.
        v_s : Tensor with shape (..., heads, num_items_in, num_s_channels_out)
            Values, scalar part.
        """

        # Additional inputs
        if additional_qk_features_mv is not None:
            inputs = torch.cat((inputs, additional_qk_features_mv), dim=-2)
        if additional_qk_features_s is not None:
            scalars = torch.cat((scalars, additional_qk_features_s), dim=-1)

        q_inputs = to_queries(
            flatten_final_dims(inputs, 2)
        ).unflatten(-1, (self.num_mv_channels_in, 16))
        q_scalars = to_queries(scalars)

        q_mv, q_s = self.q_linear(
            q_inputs, q_scalars
        )  # (..., num_items, 3 * hidden_channels * num_heads, 16)
        q_mv = rearrange(
            q_mv,
            "... items (hidden num_heads) x -> ... num_heads items hidden x",
            num_heads=self.config.num_heads,
            hidden=self.config.hidden_mv_channels,
        )

        # Same, for optional scalar components
        if q_s is not None:
            q_s = rearrange(
                q_s,
                "... items (hidden num_heads) -> ... num_heads items hidden",
                num_heads=self.config.num_heads,
                hidden=self.config.hidden_s_channels,
            )
        else:
            q_s = None

        # kv_inputs = to_keys(
        #     flatten_final_dims(inputs, 2)
        # ).unflatten(-1, (self.num_mv_channels_in, 16))
        # kv_scalars = to_keys(scalars)

        kv_mv, kv_s = self.kv_linear(
            inputs, scalars
        )  # (..., num_items, 3 * hidden_channels * num_heads, 16)
        kv_mv_blocks = to_keys(
            flatten_final_dims(kv_mv, 2)
        ).unflatten(-1, (-1, 16))
        kv_s_blocks = to_keys(kv_s)
        kv_mv_blocks = rearrange(
            kv_mv_blocks,
            "... items (kv hidden num_heads) x -> kv ... num_heads items hidden x",
            num_heads=self.config.num_heads,
            hidden=self.config.hidden_mv_channels,
            kv=2,
        )
        k_mv_blocks, v_mv_blocks = kv_mv_blocks  # each: (..., num_heads, num_items, num_channels, 16)
        kv_mv_queries = rearrange(
            to_queries(
                flatten_final_dims(kv_mv, 2)
            ).unflatten(-1, (-1, 16)),
            "... items (kv hidden num_heads) x -> kv ... num_heads items hidden x",
            num_heads=self.config.num_heads,
            hidden=self.config.hidden_mv_channels,
            kv=2,
        )
        v_mv_queries = kv_mv_queries[1]

        # Same, for optional scalar components
        if kv_s is not None:
            kv_s_blocks = rearrange(
                kv_s_blocks,
                "... items (kv hidden num_heads) -> kv ... num_heads items hidden",
                num_heads=self.config.num_heads,
                hidden=self.config.hidden_s_channels,
                kv=2,
            )
            k_s_blocks, v_s_blocks = kv_s_blocks  # each: (..., num_heads, num_items, num_channels)

            kv_s_queries = rearrange(
                to_queries(kv_s),
                "... items (kv hidden num_heads) -> kv ... num_heads items hidden",
                num_heads=self.config.num_heads,
                hidden=self.config.hidden_s_channels,
                kv=2,
            )
            v_s_queries = kv_s_queries[1]
        else:
            k_s_blocks, v_s_blocks = None, None
            v_s_queries = None

        return q_mv, k_mv_blocks, v_mv_blocks, v_mv_queries, q_s, k_s_blocks, v_s_blocks, v_s_queries


class CrossAttentionPairBias(nn.Module):
    """Geometric self-attention layer.

    Constructs queries, keys, and values, computes attention, and projects linearly to outputs.

    Parameters
    ----------
    config : SelfAttentionConfig
        Attention configuration.
    """

    def __init__(self,
                 config: SelfAttentionConfig,
                 c_z,
                 many_body_expand=False,
                 inf=1e5
    ) -> None:
        super().__init__()

        # Store settings
        self.config = config
        self.inf = inf

        # QKV computation
        self.qkv_module = CrossAttnQKVModule(config)

        # Output projection
        self.out_linear = EquiLinear(
            in_mv_channels=config.hidden_mv_channels * config.num_heads,
            out_mv_channels=config.out_mv_channels,
            in_s_channels=(
                None
                if config.in_s_channels is None
                else config.hidden_s_channels * config.num_heads
            ),
            out_s_channels=config.out_s_channels,
            initialization=config.output_init,
        )

        # Optional positional encoding
        self.pos_encoding: nn.Module
        if config.pos_encoding:
            self.pos_encoding = ApplyRotaryPositionalEncoding(
                config.hidden_s_channels, item_dim=-2, base=config.pos_enc_base
            )
        else:
            self.pos_encoding = nn.Identity()

        # Attention
        self.attention = GeometricAttention(config)
        self.pair_bias = nn.Sequential(
            LayerNorm(c_z),
            Linear(c_z, config.num_heads, bias=False)
        )

        # Dropout
        self.dropout: Optional[nn.Module]
        if config.dropout_prob is not None:
            self.dropout = GradeDropout(config.dropout_prob)
        else:
            self.dropout = None

        self.many_body_expand = many_body_expand
        if self.many_body_expand:
            self.bilinear = TwoInputGeometricBilinear(
                in_mv_channels=config.hidden_mv_channels,
                out_mv_channels=config.hidden_mv_channels,
                in_s_channels=config.hidden_s_channels,
                out_s_channels=config.hidden_s_channels
            )
            self.mbp = GeometricManyBodyProduct(config)
        else:
            self.bilinear = None
            self.mbp = None

    def forward(
        self,
        multivectors: torch.Tensor,
        pair_scalars: torch.Tensor,
        to_queries: Callable,
        to_keys: Callable,
        additional_qk_features_mv: Optional[torch.Tensor] = None,
        scalars: Optional[torch.Tensor] = None,
        additional_qk_features_s: Optional[torch.Tensor] = None,
        attention_mask=None,
        reference_mv=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes forward pass on inputs with shape `(..., items, channels, 16)`.

        The result is the following:

        ```
        # For each head
        queries = linear_channels(inputs)
        keys = linear_channels(inputs)
        values = linear_channels(inputs)
        hidden = attention_items(queries, keys, values, biases=biases)
        head_output = linear_channels(hidden)

        # Combine results
        output = concatenate_heads head_output
        ```

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., num_items, channels_in, 16)
            Input multivectors.
        additional_qk_features_mv : None or torch.Tensor with shape
            (..., num_items, add_qk_mv_channels, 16)
            Additional Q/K features, multivector part.
        scalars : None or torch.Tensor with shape (..., num_items, num_items, in_scalars)
            Optional input scalars
        additional_qk_features_s : None or torch.Tensor with shape
            (..., num_items, add_qk_mv_channels, 16)
            Additional Q/K features, scalar part.
        scalars : None or torch.Tensor with shape (..., num_items, num_items, in_scalars)
            Optional input scalars
        attention_mask: None or torch.Tensor with shape (..., num_items, num_items)
            Optional attention mask

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., num_items, channels_out, 16)
            Output multivectors.
        output_scalars : torch.Tensor with shape (..., num_items, channels_out, out_scalars)
            Output scalars, if scalars are provided. Otherwise None.
        """
        # Compute Q, K, V
        q_mv, k_mv, v_mv, v_mv_queries, q_s, k_s, v_s, v_s_queries = self.qkv_module(
            multivectors,
            scalars,
            to_queries,
            to_keys,
            additional_qk_features_mv,
            additional_qk_features_s
        )

        # Rotary positional encoding
        q_s = self.pos_encoding(q_s)
        k_s = self.pos_encoding(k_s)

        attn_bias = self.pair_bias(pair_scalars)
        if attention_mask is not None:
            attn_bias = attn_bias + (1 - attention_mask[..., None].float()) * self.inf
        attn_bias = permute_final_dims(attn_bias, (2, 0, 1))

        # Attention layer
        h_mv, h_s = self.attention(q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=attn_bias)

        if self.many_body_expand:
            assert reference_mv is not None
            h_mv, _h_s = self.bilinear(h_mv, v_mv_queries, h_s, v_s_queries, reference_mv[:, None, None])
            h_mv = self.mbp(h_mv, _h_s, reference_mv[:, None, None])

        h_mv = rearrange(
            h_mv, "... n_heads n_items hidden_channels x -> ... n_items (n_heads hidden_channels) x"
        )
        h_s = rearrange(
            h_s, "... n_heads n_items hidden_channels -> ... n_items (n_heads hidden_channels)"
        )
        h_mv = _flatten(h_mv, 1, 2)
        h_s = _flatten(h_s, 1, 2)

        # Transform linearly one more time
        outputs_mv, outputs_s = self.out_linear(h_mv, scalars=h_s)

        # Dropout
        if self.dropout is not None:
            outputs_mv, outputs_s = self.dropout(outputs_mv, outputs_s)

        return outputs_mv, outputs_s


class SequenceBlockGATrPairBiasBlock(nn.Module):
    """Equivariant transformer block for GATr.

    This is the biggest building block of GATr.

    Inputs are first processed by a block consisting of LayerNorm, multi-head geometric
    self-attention, and a residual connection. Then the data is processed by a block consisting of
    another LayerNorm, an item-wise two-layer geometric MLP with GeLU activations, and another
    residual connection.

    Parameters
    ----------
    mv_channels : int
        Number of input and output multivector channels
    s_channels: int
        Number of input and output scalar channels
    attention: SelfAttentionConfig
        Attention configuration
    mlp: MLPConfig
        MLP configuration
    dropout_prob : float or None
        Dropout probability
    checkpoint : None or sequence of "mlp", "attention"
        Which components to apply gradient checkpointing to
    """

    def __init__(
        self,
        mv_channels: int,
        s_channels: int,
        z_channels: int,
        attention: SelfAttentionConfig,
        mlp: MLPConfig,
        dropout_prob: Optional[float] = None,
        checkpoint: Optional[Sequence[Literal["mlp", "attention"]]] = None,
        many_body_expand=False
    ) -> None:
        super().__init__()

        # Gradient checkpointing settings
        if checkpoint is not None:
            for key in checkpoint:
                assert key in ["mlp", "attention"]
        self._checkpoint_mlp = checkpoint is not None and "mlp" in checkpoint
        self._checkpoint_attn = checkpoint is not None and "attention" in checkpoint

        # Normalization layer (stateless, so we can use the same layer for both normalization
        # instances)
        self.norm = EquiLayerNorm()

        # Self-attention layer
        attention = replace(
            attention,
            in_mv_channels=mv_channels,
            out_mv_channels=mv_channels,
            in_s_channels=s_channels,
            out_s_channels=s_channels,
            output_init="small",
            dropout_prob=dropout_prob,
        )
        self.attention = CrossAttentionPairBias(
            attention,
            c_z=z_channels,
            many_body_expand=many_body_expand
        )

        # MLP block
        mlp = replace(
            mlp,
            mv_channels=(mv_channels, 2 * mv_channels, mv_channels),
            s_channels=(s_channels, 2 * s_channels, s_channels),
            dropout_prob=dropout_prob,
        )
        self.mlp = GeoMLP(mlp)

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
        pair_scalars: torch.Tensor,
        to_queries: Callable,
        to_keys: Callable,
        reference_mv: Optional[torch.Tensor] = None,
        additional_qk_features_mv: Optional[torch.Tensor] = None,
        additional_qk_features_s: Optional[torch.Tensor] = None,
        attention_mask: Optional[Union[AttentionBias, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the transformer block.

        Inputs are first processed by a block consisting of LayerNorm, multi-head geometric
        self-attention, and a residual connection. Then the data is processed by a block consisting
        of another LayerNorm, an item-wise two-layer geometric MLP with GeLU activations, and
        another residual connection.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., items, channels, 16)
            Input multivectors.
        scalars : torch.Tensor with shape (..., s_channels)
            Input scalars.
        reference_mv : torch.Tensor with shape (..., 16) or None
            Reference multivector for the equivariant join operation in the MLP.
        additional_qk_features_mv : None or torch.Tensor with shape
            (..., num_items, add_qk_mv_channels, 16)
            Additional Q/K features, multivector part.
        additional_qk_features_s : None or torch.Tensor with shape
            (..., num_items, add_qk_mv_channels, 16)
            Additional Q/K features, scalar part.
        attention_mask: None or torch.Tensor or AttentionBias
            Optional attention mask.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., items, channels, 16).
            Output multivectors
        output_scalars : torch.Tensor with shape (..., s_channels)
            Output scalars
        """

        # Attention block
        attn_kwargs = dict(
            multivectors=multivectors,
            scalars=scalars,
            to_queries=to_queries,
            to_keys=to_keys,
            pair_scalars=pair_scalars,
            additional_qk_features_mv=additional_qk_features_mv,
            additional_qk_features_s=additional_qk_features_s,
            attention_mask=attention_mask,
            reference_mv=reference_mv,
        )
        if self._checkpoint_attn:
            h_mv, h_s = checkpoint_(self._attention_block, use_reentrant=False, **attn_kwargs)
        else:
            h_mv, h_s = self._attention_block(**attn_kwargs)

        # Skip connection
        outputs_mv = multivectors + h_mv
        outputs_s = scalars + h_s

        # MLP block
        mlp_kwargs = dict(multivectors=outputs_mv, scalars=outputs_s, reference_mv=reference_mv)
        if self._checkpoint_mlp:
            h_mv, h_s = checkpoint_(self._mlp_block, use_reentrant=False, **mlp_kwargs)
        else:
            h_mv, h_s = self._mlp_block(outputs_mv, scalars=outputs_s, reference_mv=reference_mv)

        # Skip connection
        outputs_mv = outputs_mv + h_mv
        outputs_s = outputs_s + h_s

        return outputs_mv, outputs_s

    def _attention_block(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
        pair_scalars: torch.Tensor,
        to_queries: Callable,
        to_keys: Callable,
        additional_qk_features_mv: Optional[torch.Tensor] = None,
        additional_qk_features_s: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        reference_mv = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attention block."""

        h_mv, h_s = self.norm(multivectors, scalars=scalars)
        h_mv, h_s = self.attention(
            h_mv,
            scalars=h_s,
            pair_scalars=pair_scalars,
            to_queries=to_queries,
            to_keys=to_keys,
            additional_qk_features_mv=additional_qk_features_mv,
            additional_qk_features_s=additional_qk_features_s,
            attention_mask=attention_mask,
            reference_mv=reference_mv
        )
        return h_mv, h_s

    def _mlp_block(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
        reference_mv: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MLP block."""

        h_mv, h_s = self.norm(multivectors, scalars=scalars)
        h_mv, h_s = self.mlp(h_mv, scalars=h_s, reference_mv=reference_mv)
        return h_mv, h_s


class SequenceBlockGATrPairBias(nn.Module):
    """GATr network for a data with a single token dimension.

    This, together with gatr.nets.axial_gatr.AxialGATr, is the main architecture proposed in our
    paper.

    It combines `num_blocks` GATr transformer blocks, each consisting of geometric self-attention
    layers, a geometric MLP, residual connections, and normalization layers. In addition, there
    are initial and final equivariant linear layers.

    Assumes input has shape `(..., items, in_channels, 16)`, output has shape
    `(..., items, out_channels, 16)`, will create hidden representations with shape
    `(..., items, hidden_channels, 16)`.

    Parameters
    ----------
    in_mv_channels : int
        Number of input multivector channels.
    out_mv_channels : int
        Number of output multivector channels.
    hidden_mv_channels : int
        Number of hidden multivector channels.
    in_s_channels : None or int
        If not None, sets the number of scalar input channels.
    out_s_channels : None or int
        If not None, sets the number of scalar output channels.
    hidden_s_channels : None or int
        If not None, sets the number of scalar hidden channels.
    attention: Dict
        Data for SelfAttentionConfig
    mlp: Dict
        Data for MLPConfig
    num_blocks : int
        Number of transformer blocks.
    checkpoint_blocks : bool
        Deprecated option to specify gradient checkpointing. Use `checkpoint=["block"]` instead
    dropout_prob : float or None
        Dropout probability
    checkpoint : None or sequence of "mlp", "attention", "block"
        Which components to apply gradient checkpointing to
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        hidden_mv_channels: int,
        in_s_channels: Optional[int],
        out_s_channels: Optional[int],
        hidden_s_channels: Optional[int],
        c_z: int,
        attention: SelfAttentionConfig,
        mlp: MLPConfig,
        num_blocks: int = 10,
        reinsert_mv_channels: Optional[Tuple[int]] = None,
        reinsert_s_channels: Optional[Tuple[int]] = None,
        checkpoint_blocks: bool = False,
        dropout_prob: Optional[float] = None,
        checkpoint: Union[
            None, Sequence[Literal["block"]], Sequence[Literal["mlp", "attention"]]
        ] = None,
        many_body_expand=False,
        **kwargs,
    ) -> None:
        super().__init__()

        # Gradient checkpointing settings
        if checkpoint_blocks:
            # The checkpoint_blocks keyword was deprecated in v1.4.0.
            if checkpoint is not None:
                raise ValueError(
                    "Both checkpoint_blocks and checkpoint were specified. Please only use"
                    "checkpoint."
                )
            warn(
                'The checkpoint_blocks keyword is deprecated since v1.4.0. Use checkpoint=["block"]'
                "instead.",
                category=DeprecationWarning,
            )
            checkpoint = ["block"]
        if checkpoint is not None:
            for key in checkpoint:
                assert key in ["block", "mlp", "attention"]
        if checkpoint is not None and "block" in checkpoint:
            self._checkpoint_blocks = True
            if "mlp" in checkpoint or "attention" in checkpoint:
                raise ValueError(
                    "Checkpointing both on the block level and the MLP / attention"
                    'level is not sensible. Please use either checkpoint=["block"] or '
                    f'checkpoint=["attention", "mlp"]. Found checkpoint={checkpoint}.'
                )
            checkpoint = None
        else:
            self._checkpoint_blocks = False

        self.linear_in = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=hidden_s_channels,
        )
        attention = replace(
            SelfAttentionConfig.cast(attention),  # convert duck typing to actual class
            additional_qk_mv_channels=(
                0 if reinsert_mv_channels is None else len(reinsert_mv_channels)
            ),
            additional_qk_s_channels=0 if reinsert_s_channels is None else len(reinsert_s_channels),
        )
        mlp = MLPConfig.cast(mlp)
        self.blocks = nn.ModuleList(
            [
                SequenceBlockGATrPairBiasBlock(
                    mv_channels=hidden_mv_channels,
                    s_channels=hidden_s_channels,
                    z_channels=c_z,
                    attention=attention,
                    mlp=mlp,
                    dropout_prob=dropout_prob,
                    checkpoint=checkpoint,
                    many_body_expand=many_body_expand
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = EquiLinear(
            hidden_mv_channels,
            out_mv_channels,
            in_s_channels=hidden_s_channels,
            out_s_channels=out_s_channels,
        )
        self._reinsert_s_channels = reinsert_s_channels
        self._reinsert_mv_channels = reinsert_mv_channels

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
        pair_scalars: torch.Tensor,
        to_queries: Callable,
        to_keys: Callable,
        attention_mask: Optional[torch.Tensor] = None,
        join_reference: Union[Tensor, str] = "data",
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Forward pass of the network.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors.
        scalars : None or torch.Tensor with shape (..., in_s_channels)
            Optional input scalars.
        attention_mask: None or torch.Tensor with shape (..., num_items, num_items)
            Optional attention mask
        join_reference : Tensor with shape (..., 16) or {"data", "canonical"}
            Reference multivector for the equivariant joint operation. If "data", a
            reference multivector is constructed from the mean of the input multivectors. If
            "canonical", a constant canonical reference multivector is used instead.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., out_mv_channels, 16)
            Output multivectors.
        outputs_s : None or torch.Tensor with shape (..., out_s_channels)
            Output scalars, if scalars are provided. Otherwise None.
        """

        # Reference multivector and channels that will be re-inserted in any query / key computation
        reference_mv = construct_reference_multivector(join_reference, multivectors)
        additional_qk_features_mv, additional_qk_features_s = self._construct_reinserted_channels(
            multivectors, scalars
        )

        # Pass through the blocks
        h_mv, h_s = self.linear_in(multivectors, scalars=scalars)
        for block in self.blocks:
            if self._checkpoint_blocks:
                h_mv, h_s = checkpoint_(
                    block,
                    h_mv,
                    use_reentrant=False,
                    scalars=h_s,
                    pair_scalars=pair_scalars,
                    to_queries=to_queries,
                    to_keys=to_keys,
                    reference_mv=reference_mv,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    attention_mask=attention_mask,
                )
            else:
                h_mv, h_s = block(
                    h_mv,
                    scalars=h_s,
                    pair_scalars=pair_scalars,
                    to_queries=to_queries,
                    to_keys=to_keys,
                    reference_mv=reference_mv,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    attention_mask=attention_mask,
                )

        outputs_mv, outputs_s = self.linear_out(h_mv, scalars=h_s)

        return outputs_mv, outputs_s

    def _construct_reinserted_channels(self, multivectors, scalars):
        """Constructs input features that will be reinserted in every attention layer."""

        if self._reinsert_mv_channels is None:
            additional_qk_features_mv = None
        else:
            additional_qk_features_mv = multivectors[..., self._reinsert_mv_channels, :]

        if self._reinsert_s_channels is None:
            additional_qk_features_s = None
        else:
            assert scalars is not None
            additional_qk_features_s = scalars[..., self._reinsert_s_channels]

        return additional_qk_features_mv, additional_qk_features_s


class SequenceBlockConditionedGATrPairBiasBlock(nn.Module):
    """Equivariant transformer block for GATr.

    This is the biggest building block of GATr.

    Inputs are first processed by a block consisting of LayerNorm, multi-head geometric
    self-attention, and a residual connection. Then the data is processed by a block consisting of
    another LayerNorm, an item-wise two-layer geometric MLP with GeLU activations, and another
    residual connection.

    Parameters
    ----------
    mv_channels : int
        Number of input and output multivector channels
    s_channels: int
        Number of input and output scalar channels
    attention: SelfAttentionConfig
        Attention configuration
    mlp: MLPConfig
        MLP configuration
    dropout_prob : float or None
        Dropout probability
    checkpoint : None or sequence of "mlp", "attention"
        Which components to apply gradient checkpointing to
    """

    def __init__(
        self,
        mv_channels: int,
        s_channels: int,
        cond_mv_channels: int,
        cond_s_channels: int,
        z_channels: int,
        attention: SelfAttentionConfig,
        mlp: MLPConfig,
        dropout_prob: Optional[float] = None,
        checkpoint: Optional[Sequence[Literal["mlp", "attention"]]] = None,
        many_body_expand=False,
    ) -> None:
        super().__init__()

        # Gradient checkpointing settings
        if checkpoint is not None:
            for key in checkpoint:
                assert key in ["mlp", "attention"]
        self._checkpoint_mlp = checkpoint is not None and "mlp" in checkpoint
        self._checkpoint_attn = checkpoint is not None and "attention" in checkpoint

        # Norms
        self.norm_attn = EquiAdaLN(
            mv_channels=mv_channels,
            scalar_channels=s_channels,
            cond_mv_channels=cond_mv_channels,
            cond_scalar_channels=cond_s_channels
        )
        self.norm_mlp = EquiAdaLN(
            mv_channels=mv_channels,
            scalar_channels=s_channels,
            cond_mv_channels=cond_mv_channels,
            cond_scalar_channels=cond_s_channels
        )

        # Self-attention layer
        attention = replace(
            attention,
            in_mv_channels=mv_channels,
            out_mv_channels=mv_channels,
            in_s_channels=s_channels,
            out_s_channels=s_channels,
            output_init="small",
            dropout_prob=dropout_prob,
        )
        self.attention = CrossAttentionPairBias(
            attention,
            c_z=z_channels,
            many_body_expand=many_body_expand
        )

        # MLP block
        mlp = replace(
            mlp,
            mv_channels=(mv_channels, 2 * mv_channels, mv_channels),
            s_channels=(s_channels, 2 * s_channels, s_channels),
            dropout_prob=dropout_prob,
        )
        self.mlp = GeoMLP(mlp)

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
        cond_multivectors: torch.Tensor,
        cond_scalars: torch.Tensor,
        pair_scalars: torch.Tensor,
        to_queries: Callable,
        to_keys: Callable,
        reference_mv: Optional[torch.Tensor] = None,
        additional_qk_features_mv: Optional[torch.Tensor] = None,
        additional_qk_features_s: Optional[torch.Tensor] = None,
        attention_mask: Optional[Union[AttentionBias, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the transformer block.

        Inputs are first processed by a block consisting of LayerNorm, multi-head geometric
        self-attention, and a residual connection. Then the data is processed by a block consisting
        of another LayerNorm, an item-wise two-layer geometric MLP with GeLU activations, and
        another residual connection.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., items, channels, 16)
            Input multivectors.
        scalars : torch.Tensor with shape (..., s_channels)
            Input scalars.
        reference_mv : torch.Tensor with shape (..., 16) or None
            Reference multivector for the equivariant join operation in the MLP.
        additional_qk_features_mv : None or torch.Tensor with shape
            (..., num_items, add_qk_mv_channels, 16)
            Additional Q/K features, multivector part.
        additional_qk_features_s : None or torch.Tensor with shape
            (..., num_items, add_qk_mv_channels, 16)
            Additional Q/K features, scalar part.
        attention_mask: None or torch.Tensor or AttentionBias
            Optional attention mask.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., items, channels, 16).
            Output multivectors
        output_scalars : torch.Tensor with shape (..., s_channels)
            Output scalars
        """

        # Attention block
        attn_kwargs = dict(
            multivectors=multivectors,
            scalars=scalars,
            cond_multivectors=cond_multivectors,
            cond_scalars=cond_scalars,
            to_queries=to_queries,
            to_keys=to_keys,
            pair_scalars=pair_scalars,
            additional_qk_features_mv=additional_qk_features_mv,
            additional_qk_features_s=additional_qk_features_s,
            attention_mask=attention_mask,
            reference_mv=reference_mv
        )
        if self._checkpoint_attn:
            h_mv, h_s = checkpoint_(self._attention_block, use_reentrant=False, **attn_kwargs)
        else:
            h_mv, h_s = self._attention_block(**attn_kwargs)

        # Skip connection
        outputs_mv = multivectors + h_mv
        outputs_s = scalars + h_s

        # MLP block
        mlp_kwargs = dict(
            multivectors=outputs_mv,
            scalars=outputs_s,
            cond_multivectors=cond_multivectors,
            cond_scalars=cond_scalars,
            reference_mv=reference_mv
        )
        if self._checkpoint_mlp:
            h_mv, h_s = checkpoint_(self._mlp_block, use_reentrant=False, **mlp_kwargs)
        else:
            h_mv, h_s = self._mlp_block(**mlp_kwargs)

        # Skip connection
        outputs_mv = outputs_mv + h_mv
        outputs_s = outputs_s + h_s

        return outputs_mv, outputs_s

    def _attention_block(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
        cond_multivectors: torch.Tensor,
        cond_scalars: torch.Tensor,
        pair_scalars: torch.Tensor,
        to_queries: Callable,
        to_keys: Callable,
        additional_qk_features_mv: Optional[torch.Tensor] = None,
        additional_qk_features_s: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        reference_mv = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attention block."""

        h_mv, h_s = self.norm_attn(
            multivectors,
            scalars,
            cond_multivectors,
            cond_scalars,
        )
        h_mv, h_s = self.attention(
            h_mv,
            scalars=h_s,
            pair_scalars=pair_scalars,
            to_queries=to_queries,
            to_keys=to_keys,
            additional_qk_features_mv=additional_qk_features_mv,
            additional_qk_features_s=additional_qk_features_s,
            attention_mask=attention_mask,
            reference_mv=reference_mv
        )
        return h_mv, h_s

    def _mlp_block(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
        cond_multivectors: torch.Tensor,
        cond_scalars: torch.Tensor,
        reference_mv: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MLP block."""

        h_mv, h_s = self.norm_mlp(
            multivectors,
            scalars,
            cond_multivectors,
            cond_scalars,
        )
        h_mv, h_s = self.mlp(h_mv, scalars=h_s, reference_mv=reference_mv)
        return h_mv, h_s


class SequenceBlockConditionedGATrPairBias(nn.Module):
    """GATr network for a data with a single token dimension.

    This, together with gatr.nets.axial_gatr.AxialGATr, is the main architecture proposed in our
    paper.

    It combines `num_blocks` GATr transformer blocks, each consisting of geometric self-attention
    layers, a geometric MLP, residual connections, and normalization layers. In addition, there
    are initial and final equivariant linear layers.

    Assumes input has shape `(..., items, in_channels, 16)`, output has shape
    `(..., items, out_channels, 16)`, will create hidden representations with shape
    `(..., items, hidden_channels, 16)`.

    Parameters
    ----------
    in_mv_channels : int
        Number of input multivector channels.
    out_mv_channels : int
        Number of output multivector channels.
    hidden_mv_channels : int
        Number of hidden multivector channels.
    in_s_channels : None or int
        If not None, sets the number of scalar input channels.
    out_s_channels : None or int
        If not None, sets the number of scalar output channels.
    hidden_s_channels : None or int
        If not None, sets the number of scalar hidden channels.
    attention: Dict
        Data for SelfAttentionConfig
    mlp: Dict
        Data for MLPConfig
    num_blocks : int
        Number of transformer blocks.
    checkpoint_blocks : bool
        Deprecated option to specify gradient checkpointing. Use `checkpoint=["block"]` instead
    dropout_prob : float or None
        Dropout probability
    checkpoint : None or sequence of "mlp", "attention", "block"
        Which components to apply gradient checkpointing to
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        hidden_mv_channels: int,
        in_s_channels: Optional[int],
        out_s_channels: Optional[int],
        hidden_s_channels: Optional[int],
        c_z: int,
        cond_mv_channels: int,
        cond_s_channels: int,
        attention: SelfAttentionConfig,
        mlp: MLPConfig,
        num_blocks: int = 10,
        reinsert_mv_channels: Optional[Tuple[int]] = None,
        reinsert_s_channels: Optional[Tuple[int]] = None,
        checkpoint_blocks: bool = False,
        dropout_prob: Optional[float] = None,
        checkpoint: Union[
            None, Sequence[Literal["block"]], Sequence[Literal["mlp", "attention"]]
        ] = None,
        many_body_expand=False,
        **kwargs,
    ) -> None:
        super().__init__()

        # Gradient checkpointing settings
        if checkpoint_blocks:
            # The checkpoint_blocks keyword was deprecated in v1.4.0.
            if checkpoint is not None:
                raise ValueError(
                    "Both checkpoint_blocks and checkpoint were specified. Please only use"
                    "checkpoint."
                )
            warn(
                'The checkpoint_blocks keyword is deprecated since v1.4.0. Use checkpoint=["block"]'
                "instead.",
                category=DeprecationWarning,
            )
            checkpoint = ["block"]
        if checkpoint is not None:
            for key in checkpoint:
                assert key in ["block", "mlp", "attention"]
        if checkpoint is not None and "block" in checkpoint:
            self._checkpoint_blocks = True
            if "mlp" in checkpoint or "attention" in checkpoint:
                raise ValueError(
                    "Checkpointing both on the block level and the MLP / attention"
                    'level is not sensible. Please use either checkpoint=["block"] or '
                    f'checkpoint=["attention", "mlp"]. Found checkpoint={checkpoint}.'
                )
            checkpoint = None
        else:
            self._checkpoint_blocks = False

        self.linear_in = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=hidden_s_channels,
        )
        attention = replace(
            SelfAttentionConfig.cast(attention),  # convert duck typing to actual class
            additional_qk_mv_channels=(
                0 if reinsert_mv_channels is None else len(reinsert_mv_channels)
            ),
            additional_qk_s_channels=0 if reinsert_s_channels is None else len(reinsert_s_channels),
        )
        mlp = MLPConfig.cast(mlp)
        self.blocks = nn.ModuleList(
            [
                SequenceBlockConditionedGATrPairBiasBlock(
                    mv_channels=hidden_mv_channels,
                    s_channels=hidden_s_channels,
                    cond_mv_channels=cond_mv_channels,
                    cond_s_channels=cond_s_channels,
                    z_channels=c_z,
                    attention=attention,
                    mlp=mlp,
                    dropout_prob=dropout_prob,
                    checkpoint=checkpoint,
                    many_body_expand=many_body_expand
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = EquiLinear(
            hidden_mv_channels,
            out_mv_channels,
            in_s_channels=hidden_s_channels,
            out_s_channels=out_s_channels,
        )
        self._reinsert_s_channels = reinsert_s_channels
        self._reinsert_mv_channels = reinsert_mv_channels

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
        pair_scalars: torch.Tensor,
        cond_mv: torch.Tensor,
        cond_s: torch.Tensor,
        to_queries: Callable,
        to_keys: Callable,
        attention_mask: Optional[torch.Tensor] = None,
        join_reference: Union[Tensor, str] = "data",
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Forward pass of the network.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors.
        scalars : None or torch.Tensor with shape (..., in_s_channels)
            Optional input scalars.
        attention_mask: None or torch.Tensor with shape (..., num_items, num_items)
            Optional attention mask
        join_reference : Tensor with shape (..., 16) or {"data", "canonical"}
            Reference multivector for the equivariant joint operation. If "data", a
            reference multivector is constructed from the mean of the input multivectors. If
            "canonical", a constant canonical reference multivector is used instead.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., out_mv_channels, 16)
            Output multivectors.
        outputs_s : None or torch.Tensor with shape (..., out_s_channels)
            Output scalars, if scalars are provided. Otherwise None.
        """

        # Reference multivector and channels that will be re-inserted in any query / key computation
        reference_mv = construct_reference_multivector(join_reference, multivectors)
        additional_qk_features_mv, additional_qk_features_s = self._construct_reinserted_channels(
            multivectors, scalars
        )

        # Pass through the blocks
        h_mv, h_s = self.linear_in(multivectors, scalars=scalars)
        for block in self.blocks:
            if self._checkpoint_blocks:
                h_mv, h_s = checkpoint_(
                    block,
                    h_mv,
                    use_reentrant=False,
                    scalars=h_s,
                    pair_scalars=pair_scalars,
                    cond_multivectors=cond_mv,
                    cond_scalars=cond_s,
                    to_queries=to_queries,
                    to_keys=to_keys,
                    reference_mv=reference_mv,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    attention_mask=attention_mask,
                )
            else:
                h_mv, h_s = block(
                    h_mv,
                    scalars=h_s,
                    pair_scalars=pair_scalars,
                    cond_multivectors=cond_mv,
                    cond_scalars=cond_s,
                    to_queries=to_queries,
                    to_keys=to_keys,
                    reference_mv=reference_mv,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    attention_mask=attention_mask,
                )

        outputs_mv, outputs_s = self.linear_out(h_mv, scalars=h_s)

        return outputs_mv, outputs_s

    def _construct_reinserted_channels(self, multivectors, scalars):
        """Constructs input features that will be reinserted in every attention layer."""

        if self._reinsert_mv_channels is None:
            additional_qk_features_mv = None
        else:
            additional_qk_features_mv = multivectors[..., self._reinsert_mv_channels, :]

        if self._reinsert_s_channels is None:
            additional_qk_features_s = None
        else:
            assert scalars is not None
            additional_qk_features_s = scalars[..., self._reinsert_s_channels]

        return additional_qk_features_mv, additional_qk_features_s