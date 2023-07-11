""" Sequence design via diffusing sidechain densities

Parts adapted from https://github.com/NVIDIA/DeepLearningExamples/blob/master/DGLPyTorch/DrugDiscovery/SE3Transformer/se3_transformer/model/layers/attention.py """

from typing import Dict, Optional, Union

import dgl
import numpy as np
import torch
from dgl import DGLGraph
from dgl.ops import edge_softmax
from se3_transformer.model.basis import get_basis, update_basis_with_fused
from se3_transformer.model.fiber import Fiber
from se3_transformer.model.layers.convolution import ConvSE3, ConvSE3FuseLevel
from se3_transformer.model.layers.linear import LinearSE3
from se3_transformer.model.layers.norm import NormSE3
from se3_transformer.runtime.utils import (aggregate_residual, degree_to_dim,
                                           unfuse_features)
from torch import Tensor, nn
from torch.cuda.nvtx import range as nvtx_range

from ligbinddiff.utils.type_l import (int_to_str_key, str_to_int_key,
                                      type_l_add, type_l_apply, type_l_cat,
                                      type_l_mult, type_l_partial_cat)


def is_bad(t):
    return torch.isnan(t).any() or not torch.isfinite(t).any()


class FeedForwardSE3(nn.Module):
    def __init__(self, fiber_in, fiber_hidden):
        super().__init__()
        self.W_in = LinearSE3(fiber_in, fiber_hidden)
        self.nonlin = NormSE3(fiber_hidden)
        self.W_out = LinearSE3(fiber_hidden, fiber_in)

    def forward(self, features):
        return self.W_out(self.nonlin(self.W_in(features)))


class AttentionSE3(nn.Module):
    """ Multi-headed sparse graph self-attention (SE(3)-equivariant) """

    def __init__(

            self,
            num_heads: int,
            key_fiber: Fiber,
            value_fiber: Fiber
    ):
        """
        :param num_heads:     Number of attention heads
        :param key_fiber:     Fiber for the keys (and also for the queries)
        :param value_fiber:   Fiber for the values
        """
        super().__init__()
        self.num_heads = num_heads
        self.key_fiber = key_fiber
        self.value_fiber = value_fiber

    def forward(
            self,
            value: Union[Tensor, Dict[str, Tensor]],  # edge features (may be fused)
            key: Union[Tensor, Dict[str, Tensor]],  # edge features (may be fused)
            query: Dict[str, Tensor],  # node features
            graph: DGLGraph
    ):
        with nvtx_range('AttentionSE3'):
            with nvtx_range('reshape keys and queries'):
                if isinstance(key, Tensor):
                    # case where features of all types are fused
                    key = key.reshape(key.shape[0], self.num_heads, -1)
                    # need to reshape queries that way to keep the same layout as keys
                    out = torch.cat([query[str(d)] for d in self.key_fiber.degrees], dim=-1)
                    query = out.reshape(list(query.values())[0].shape[0], self.num_heads, -1)
                else:
                    # features are not fused, need to fuse and reshape them
                    key = self.key_fiber.to_attention_heads(key, self.num_heads)
                    query = self.key_fiber.to_attention_heads(query, self.num_heads)

            with nvtx_range('attention dot product + softmax'):
                # Compute attention weights (softmax of inner product between key and query)
                edge_weights = dgl.ops.e_dot_v(graph, key, query).squeeze(-1)
                edge_weights = edge_weights / np.sqrt(self.key_fiber.num_features)
                edge_prelogits = edge_weights
                edge_weights = edge_softmax(graph, edge_weights)
                edge_weights = edge_weights[..., None, None]

            with nvtx_range('weighted sum'):
                if isinstance(value, Tensor):
                    # features of all types are fused
                    v = value.view(value.shape[0], self.num_heads, -1, value.shape[-1])
                    weights = edge_weights * v
                    feat_out = dgl.ops.copy_e_sum(graph, weights)
                    feat_out = feat_out.view(feat_out.shape[0], -1, feat_out.shape[-1])  # merge heads
                    out = unfuse_features(feat_out, self.value_fiber.degrees)
                else:
                    out = {}
                    for degree, channels in self.value_fiber:
                        v = value[str(degree)].view(-1, self.num_heads, channels // self.num_heads,
                                                    degree_to_dim(degree))
                        weights = edge_weights * v
                        res = dgl.ops.copy_e_sum(graph, weights)
                        out[str(degree)] = res.view(-1, channels, degree_to_dim(degree))  # merge heads

                return out, edge_prelogits


class AttentionBlockSE3(nn.Module):
    """ Multi-headed sparse graph self-attention block with skip connection, linear projection (SE(3)-equivariant) """

    def __init__(
            self,
            fiber_in: Fiber,
            fiber_out: Fiber,
            fiber_edge: Fiber,
            num_heads: int = 4,
            channels_div: int = 2,
            use_layer_norm: bool = False,
            max_degree: int = 4,
            fuse_level: ConvSE3FuseLevel = ConvSE3FuseLevel.FULL,
            low_memory: bool = False,
            **kwargs
    ):
        """
        :param fiber_in:         Fiber describing the input features
        :param fiber_out:        Fiber describing the output features
        :param fiber_edge:       Fiber describing the edge features (node distances excluded)
        :param num_heads:        Number of attention heads
        :param channels_div:     Divide the channels by this integer for computing values
        :param use_layer_norm:   Apply layer normalization between MLP layers
        :param max_degree:       Maximum degree used in the bases computation
        :param fuse_level:       Maximum fuse level to use in TFN convolutions
        """
        super().__init__()
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.fiber_edge = fiber_edge
        fiber_edge_plus = fiber_edge + Fiber({0:1})
        # value_fiber has same structure as fiber_out but #channels divided by 'channels_div'
        value_fiber = Fiber([(degree, channels // channels_div) for degree, channels in fiber_out])
        # key_query_fiber has the same structure as fiber_out, but only degrees which are in in_fiber
        # (queries are merely projected, hence degrees have to match input)
        key_query_fiber = Fiber([(fe.degree, fe.channels) for fe in value_fiber if fe.degree in fiber_in.degrees])

        self.to_key_value = ConvSE3(fiber_in, value_fiber + key_query_fiber, pool=False, fiber_edge=fiber_edge,
                                    use_layer_norm=use_layer_norm, max_degree=max_degree, fuse_level=fuse_level,
                                    allow_fused_output=True, low_memory=low_memory)
        self.to_query = LinearSE3(fiber_in, key_query_fiber)
        self.attention = AttentionSE3(num_heads, key_query_fiber, value_fiber)
        self.project_node = LinearSE3(value_fiber + fiber_in, fiber_out)
        self.project_edge = LinearSE3(fiber_edge_plus + Fiber({0: num_heads}), fiber_edge_plus)

    def forward(
            self,
            node_features: Dict[str, Tensor],
            edge_features: Dict[str, Tensor],
            graph: DGLGraph,
            basis: Dict[str, Tensor]
    ):
        with nvtx_range('AttentionBlockSE3'):
            with nvtx_range('keys / values'):
                fused_key_value = self.to_key_value(node_features, edge_features, graph, basis)
                key, value = self._get_key_value_from_fused(fused_key_value)

            with nvtx_range('queries'):
                query = self.to_query(node_features)

            z, edge_preupdate = self.attention(value, key, query, graph)
            z_concat = aggregate_residual(node_features, z, 'cat')
            edge_preupdate = type_l_partial_cat(edge_features, {'0': edge_preupdate.unsqueeze(-1)})
            edge_update = self.project_edge(edge_preupdate)
            return self.project_node(z_concat), edge_update

    def _get_key_value_from_fused(self, fused_key_value):
        # Extract keys and queries features from fused features
        if isinstance(fused_key_value, Tensor):
            # Previous layer was a fully fused convolution
            value, key = torch.chunk(fused_key_value, chunks=2, dim=-2)
        else:
            key, value = {}, {}
            for degree, feat in fused_key_value.items():
                if int(degree) in self.fiber_in.degrees:
                    value[degree], key[degree] = torch.chunk(feat, chunks=2, dim=-2)
                else:
                    value[degree] = feat

        return key, value


class DensityUpdate(nn.Module):
    """ Layer to perform a gated update on the density """
    def __init__(self, fiber_node, fiber_density, fiber_edge, max_degree=5):
        super().__init__()
        # print(fiber_node, fiber_density)
        self.update = ConvSE3(fiber_in=fiber_node,
                              fiber_out=fiber_density,
                              fiber_edge=fiber_edge,
                              self_interaction=True,
                              max_degree=max_degree)
        self.gate = ConvSE3(fiber_in=fiber_node,
                            fiber_out=fiber_density,
                            fiber_edge=fiber_edge,
                            self_interaction=True,
                            max_degree=max_degree)
        self.transition = LinearSE3(fiber_density, fiber_density)
        self.update_norm = NormSE3(fiber_density)

    def forward(self, node_features, density_features, edge_features, graph, basis):
        update = self.update(node_features, edge_features, graph, basis)
        update = self.update_norm(update)
        update = self.transition(update)

        # gate = self.gate(node_features, edge_features, graph, basis)
        # gate = type_l_apply(torch.sigmoid, gate)

        # update = type_l_mult(gate, update)

        return type_l_add(density_features, update)


class DensityDenoisingLayer(nn.Module):
    """ Denoising layer on sidechain densities """
    def __init__(self,
                 fiber_node,
                 fiber_density,
                 fiber_edge,
                 max_degree=5):
        """
        Args
        ----
        """
        super().__init__()
        self.attention = AttentionBlockSE3(
            fiber_in=fiber_node,
            fiber_out=fiber_node,
            fiber_edge=fiber_edge,
            max_degree=max_degree
        )
        self.density_update = DensityUpdate(
            fiber_node=fiber_node,
            fiber_density=fiber_density,
            fiber_edge=fiber_edge,
            max_degree=max_degree
        )
        fiber_edge_p1 = fiber_edge + Fiber({0:1})
        self.norm_nodes_att = NormSE3(fiber_node)
        self.norm_edges_att = NormSE3(fiber_edge_p1)

        self.ff_node = FeedForwardSE3(fiber_node, fiber_node * 2)
        self.ff_edge = FeedForwardSE3(fiber_edge_p1, fiber_edge_p1 * 2)
        self.norm_nodes_ff = NormSE3(fiber_node)
        self.norm_edges_ff = NormSE3(fiber_edge_p1)

    def forward(
            self,
            density_features: Dict[str, Tensor],
            node_features: Dict[str, Tensor],
            edge_features: Dict[str, Tensor],
            graph: DGLGraph,
            basis: Dict[str, Tensor]
    ):
        node_update, edge_update = self.attention(node_features, edge_features, graph, basis)
        node_features = type_l_add(node_features, node_update)
        edge_features = type_l_add(edge_features, edge_update)
        node_features = self.norm_nodes_att(node_features)
        edge_features = self.norm_edges_att(edge_features)

        node_features = self.norm_nodes_ff(
            type_l_add(
                self.ff_node(node_features),
                node_features
                )
        )
        edge_features = self.norm_edges_ff(
            type_l_add(
                self.ff_edge(edge_features),
                edge_features
                )
        )

        density_features = self.density_update(node_features, density_features, edge_features, graph, basis)
        return density_features, node_features, edge_features


class SequenceHead(nn.Module):
    """ Layer to predict AA identity from density """
    def __init__(self, fiber_density, fiber_edge, h_dim, num_aa=20):
        super().__init__()
        fiber_out = Fiber({0: h_dim})
        self.collapse = ConvSE3(fiber_in=fiber_density,
                                fiber_out=fiber_out,
                                fiber_edge=fiber_edge,
                                self_interaction=True)
        self.norm = nn.LayerNorm(h_dim)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, num_aa)
        )

    def forward(self, density_features, edge_features, graph, basis):
        density_scalars = self.collapse(density_features, edge_features, graph, basis)
        density_scalars = density_scalars['0'].squeeze(-1)
        # density_scalars = self.norm(density_scalars)
        prelogits = self.mlp(density_scalars)
        logits = torch.log_softmax(prelogits, dim=-1)
        return logits


class Atom91Head(nn.Module):
    """ Layer to predict sidechain atom positions from density """
    def __init__(self,
                 fiber_density,
                 fiber_edge,
                 num_aa=20,
                 num_layers=3):
        super().__init__()
        fiber_aa = Fiber({0: num_aa})
        fiber_out = Fiber({1: 91})
        self.transition = LinearSE3(fiber_density+fiber_aa, fiber_density)
        self.transition_norm = NormSE3(fiber_density)
        self.collapse = ConvSE3(fiber_in=fiber_density,
                                fiber_out=fiber_out,
                                fiber_edge=fiber_edge,
                                self_interaction=True)
        self.collapse_norm = NormSE3(fiber_out)
        self.refine = nn.ModuleList([
            nn.Sequential(
                LinearSE3(fiber_out, fiber_out*2),
                NormSE3(fiber_out*2),
                LinearSE3(fiber_out*2, fiber_out)
            )
            for _ in range(num_layers)
        ])

    def forward(self, density_features, sequence_features, edge_features, graph, basis):
        in_features = type_l_partial_cat(density_features, sequence_features)
        transition = self.transition(in_features)
        norm = self.transition_norm(transition)
        atoms = self.collapse(norm, edge_features, graph, basis)
        atoms = self.collapse_norm(atoms)
        for layer in self.refine:
            atoms = type_l_add(atoms, layer(atoms))

        return atoms


# adapted from https://github.com/jmclong/random-fourier-features-pytorch/blob/main/rff/layers.py
class RBF(nn.Module):
    """ Damped random Fourier Feature encoding layer """
    def __init__(self, n_basis=64):
        super().__init__()
        kappa = torch.randn((n_basis,))
        self.register_buffer('kappa', kappa)

    def forward(self, ts):
        tp = 2 * np.pi * ts * self.kappa
        return torch.cat([torch.cos(tp), torch.sin(tp)], dim=-1)


def get_populated_edge_features(relative_pos: Tensor, edge_features: Optional[Dict[str, Tensor]] = None):
    """ Add relative positions to existing edge features """
    edge_features = edge_features.copy() if edge_features else {}
    r = relative_pos.norm(dim=-1, keepdim=True)
    if '0' in edge_features:
        edge_features['0'] = torch.cat([edge_features['0'], r[..., None]], dim=-2)
    else:
        edge_features['0'] = r[..., None]

    return edge_features


class DensityDenoiser(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 fiber_in,
                 fiber_node,
                 fiber_density,
                 fiber_edge,
                 scalar_h_dim=128,
                 h_time=64,
                 n_layers=4,
                 low_memory=False,
                 tensor_cores=False):
        super().__init__()
        self.max_degree = max(
            *fiber_in.degrees,
            *fiber_node.degrees,
            *fiber_density.degrees,
            *fiber_edge.degrees,
        )
        self.tensor_cores = tensor_cores
        self.low_memory = low_memory
        if low_memory:
            self.fuse_level = ConvSE3FuseLevel.NONE
        else:
            # Fully fused convolutions when using Tensor Cores (and not low memory mode)
            self.fuse_level = ConvSE3FuseLevel.FULL if tensor_cores else ConvSE3FuseLevel.PARTIAL
        self.density_n_max = max(fiber_density.degrees)

        # generate fiber in structure with appended time
        self.h_time = h_time
        fiber_in_with_time = {
            degree: num_channels
            for degree, num_channels in fiber_in.items()
        }
        fiber_in_with_time[0] = fiber_in_with_time[0] + h_time
        fiber_in_with_time = Fiber(fiber_in_with_time)
        self.time_rbf = RBF(n_basis=h_time//2)
        self.time_mlp = nn.Sequential(
            nn.Linear(h_time, scalar_h_dim),
            nn.ReLU(),
            nn.Linear(scalar_h_dim, h_time)
        )

        # we need this to project to higher fiber dimensionality
        self.embed_node = ConvSE3(fiber_in=fiber_in_with_time,
                                  fiber_out=fiber_node,
                                  fiber_edge=fiber_edge,
                                  self_interaction=True,
                                  max_degree=self.max_degree)
        # add rel_pos mag
        fiber_edge_plus = fiber_edge + Fiber({0:1})
        self.embed_edge = LinearSE3(fiber_edge_plus, fiber_edge_plus)
        self.denoiser = nn.ModuleList([
            DensityDenoisingLayer(fiber_node=fiber_node,
                                  fiber_density=fiber_density,
                                  fiber_edge=fiber_edge,
                                  max_degree=self.max_degree)
            for _ in range(n_layers)
        ])

        self.coalesce = ConvSE3(fiber_in=fiber_density,
                                fiber_out=fiber_density,
                                fiber_edge=fiber_edge,
                                self_interaction=True,
                                max_degree=self.max_degree)
        self.coalesce_gate = ConvSE3(fiber_in=fiber_node,
                                     fiber_out=fiber_density,
                                     fiber_edge=fiber_edge,
                                     self_interaction=True,
                                     max_degree=self.max_degree)
        self.coalesce_norm = NormSE3(fiber_density)

        self.output_refine = nn.Sequential(
            LinearSE3(fiber_density, fiber_density*2),
            NormSE3(fiber_density*2),
            LinearSE3(fiber_density*2, fiber_density*2),
            NormSE3(fiber_density*2),
            LinearSE3(fiber_density*2, fiber_density)
        )
        self.output_gate = nn.Sequential(
            LinearSE3(fiber_density, fiber_density*2),
            NormSE3(fiber_density*2),
            LinearSE3(fiber_density*2, fiber_density*2),
            NormSE3(fiber_density*2),
            LinearSE3(fiber_density*2, fiber_density)
        )

        self.seq_head = SequenceHead(fiber_density, fiber_edge, scalar_h_dim)
        self.atom91 = Atom91Head(fiber_density, fiber_edge)

    def forward(self, node_features, edge_features, density_features, ts, graph, basis=None):
        # print("Input")
        # print({k: is_bad(v) for k, v in node_features.items()})
        # print({k: is_bad(v) for k, v in edge_features.items()})
        # print({k: is_bad(v) for k, v in density_features.items()})

        ## setup basis and features
        # Compute bases in case they weren't precomputed as part of the data loading
        basis = basis or get_basis(graph.edata['rel_pos'], max_degree=self.max_degree, compute_gradients=False,
                                   amp=torch.is_autocast_enabled())

        # Add fused bases (per output degree, per input degree, and fully fused) to the dict
        basis = update_basis_with_fused(basis, self.max_degree,
                                        use_pad_trick=self.tensor_cores and not self.low_memory,
                                        fully_fused=self.fuse_level == ConvSE3FuseLevel.FULL)


        ## add time embedding
        fourier_time = self.time_rbf(ts)  # (h_time,)
        node_scalars = node_features[0]  # n_res x n_channel x 1
        preshape = list(node_scalars.shape[:-2])
        fourier_time = fourier_time.view([1 for _ in preshape] + [self.h_time]).expand(
            preshape + [-1]
        )
        embedded_time = self.time_mlp(fourier_time)
        embedded_time = {0: embedded_time.unsqueeze(-1)}
        node_features = type_l_partial_cat(node_features, embedded_time)

        ## se3 transformer uses str keys
        node_features = int_to_str_key(node_features)
        density_features = int_to_str_key(density_features)
        edge_features = int_to_str_key(edge_features)

        edge_features = get_populated_edge_features(graph.edata['rel_pos'], edge_features)

        # print("Start")
        # print({k: is_bad(v) for k, v in node_features.items()})
        # print({k: is_bad(v) for k, v in edge_features.items()})
        # print({k: is_bad(v) for k, v in density_features.items()})

        ## denoising
        f_D = density_features
        f_E = self.embed_edge(edge_features)
        f_V = self.embed_node(node_features, f_E, graph, basis)

        # print("Embed")
        # print({k: is_bad(v) for k, v in node_features.items()})
        # print({k: is_bad(v) for k, v in edge_features.items()})
        # print({k: is_bad(v) for k, v in density_features.items()})

        for layer in self.denoiser:
            f_D, f_V, f_E = layer(f_D, f_V, f_E, graph, basis)

        # print("Layers")
        # print({k: is_bad(v) for k, v in f_D.items()})
        # print({k: is_bad(v) for k, v in f_V.items()})
        # print({k: is_bad(v) for k, v in f_E.items()})

        f_D_update = self.coalesce(f_D, f_E, graph, basis)
        f_D_update = self.coalesce_norm(f_D_update)
        f_D_gate = self.coalesce_gate(f_V, f_E, graph, basis)
        f_D_gate = type_l_apply(torch.sigmoid, f_D_gate)
        f_D_update = type_l_mult(f_D_update, f_D_gate)
        f_D = type_l_add(f_D, f_D_update)

        f_D_update = self.output_refine(f_D)
        f_D_gate = self.output_gate(f_D)
        f_D_gate = type_l_apply(torch.sigmoid, f_D_gate)
        f_D_update = type_l_mult(f_D_update, f_D_gate)
        f_D = type_l_add(f_D, f_D_update)
        # print("Final conv")
        # print({k: is_bad(v) for k, v in f_D.items()})

        ## aux heads
        seq_logits = self.seq_head(f_D, f_E, graph, basis)
        seq_features = {'0': seq_logits.unsqueeze(-1)}
        atom91 = self.atom91(f_D, seq_features, f_E, graph, basis)

        return str_to_int_key(f_D), seq_logits, atom91['1']
