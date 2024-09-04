from typing import Union, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import e3nn
from e3nn import o3
from e3nn.nn import NormActivation, Gate, BatchNorm
import torch_geometric.utils as pygu

from proteinzen.model.modules.layers.edge.tfn import FasterTensorProduct


## modified from diffdock https://github.com/gcorso/DiffDock/blob/main/models/tensor_layers.py

ACTIVATIONS = {
    'relu': nn.ReLU,
    'silu': nn.SiLU
}

def irrep_to_size(irrep):
    irreps = irrep.split(' + ')
    size = 0
    for ir in irreps:
        m, (l, p) = ir.split('x')
        size += int(m) * (2 * int(l) + 1)
    return size


def tp_scatter_simple(tp, fc_layer, node_attr, edge_index, edge_attr, edge_sh,
                      out_nodes=None, reduce='mean', edge_weight=1.0):
    """
    Perform TensorProduct + scatter operation, aka graph convolution.

    This function is only for edge_groups == 1. For multiple edge groups, and for larger graphs,
    use tp_scatter_multigroup instead.
    """

    assert isinstance(edge_attr, torch.Tensor), \
        "This function is only for a single edge group, so edge_attr must be a tensor and not a list."

    _device = node_attr.device
    _dtype = node_attr.dtype
    edge_dst, edge_src = edge_index
    out_irreps = fc_layer(edge_attr).to(_device).to(_dtype)
    out_irreps.mul_(edge_weight)
    tp = tp(node_attr[edge_dst], edge_sh, out_irreps)
    out_nodes = out_nodes or node_attr.shape[0]
    out = pygu.scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)
    return out


def tp_scatter_multigroup(tp: o3.TensorProduct, fc_layer: Union[nn.Module, nn.ModuleDict],
                          node_attr: torch.Tensor, edge_index: torch.Tensor,
                          edge_attr_groups: Dict[str, torch.Tensor], edge_sh: torch.Tensor,
                          out_nodes=None, reduce='mean', edge_weight=1.0):
    """
    Perform TensorProduct + scatter operation, aka graph convolution.

    To keep the peak memory usage reasonably low, this function does not concatenate the edge_attr_groups.
    Rather, we sum the output of the tensor product for each edge group, and then divide by the number of edges

    Parameters
    ----------
    tp: o3.TensorProduct
    fc_layer: nn.Module, or nn.ModuleList
        If a list, must be the same length as edge_attr_groups
    node_attr: torch.Tensor
    edge_index: torch.Tensor of shape (2, num_edges)
        Indicates the source and destination nodes of each edge
    edge_attr_groups: List[torch.Tensor]
        List of tensors, with shape (X_i, num_edge_attributes). Each tensor is a different group of edge attributes
        X may be different for each tensor, although sum(X_i) must be equal to edge_index.shape[1]
    edge_sh: torch.Tensor
        Spherical harmonics for the edges (see o3.spherical_harmonics)
    out_nodes:
        Number of output nodes
    reduce: str
        'mean' or 'sum'. Reduce function for scatter.
    edge_weight : float or torch.Tensor
        Edge weights. If a tensor, must be the same shape as `edge_index`

    Returns
    -------
    torch.Tensor
        Result of the graph convolution
    """

    # assert isinstance(edge_attr_groups, list), "This function is only for a list of edge groups"
    assert reduce in {"mean", "sum"}, "Only 'mean' and 'sum' are supported for reduce"
    # It would be possible to support mul/min/max but that would require more work and more code,
    # so only going to do it if it's needed.

    _device = node_attr.device
    _dtype = node_attr.dtype
    edge_dst, edge_src = edge_index
    edge_attr_lengths = {
        key: _edge_attr.shape[0]
        for key, _edge_attr in edge_attr_groups.items()
    }
    total_rows = sum(edge_attr_lengths.values())
    assert total_rows == edge_index.shape[1], "Sum of edge_attr_groups must be equal to edge_index.shape[1] but got " + str((total_rows, edge_index.shape[1]))
    edge_weight_is_indexable = hasattr(edge_weight, '__getitem__')

    out_nodes = out_nodes or node_attr.shape[0]
    total_output_dim = sum([x.dim for x in tp.irreps_out])
    final_out = torch.zeros((out_nodes, total_output_dim), device=_device, dtype=_dtype)
    div_factors = torch.zeros(out_nodes, device=_device, dtype=_dtype)

    cur_start = 0
    for edge_group_key in edge_attr_groups.keys():
        cur_length = edge_attr_lengths[edge_group_key]
        cur_end = cur_start + cur_length
        cur_edge_range = slice(cur_start, cur_end)
        cur_edge_src, cur_edge_dst = edge_src[cur_edge_range], edge_dst[cur_edge_range]

        cur_fc = fc_layer[edge_group_key] if isinstance(fc_layer, nn.ModuleDict) else fc_layer
        cur_out_irreps = cur_fc(edge_attr_groups[edge_group_key])
        if edge_weight_is_indexable:
            cur_out_irreps.mul_(edge_weight[cur_edge_range])
        else:
            cur_out_irreps.mul_(edge_weight)

        summand = tp(node_attr[cur_edge_dst, :], edge_sh[cur_edge_range, :], cur_out_irreps)
        # We take a simple sum, and then add up the count of edges which contribute,
        # so that we can take the mean later.
        final_out += pygu.scatter(summand, cur_edge_src, dim=0, dim_size=out_nodes, reduce="sum")
        div_factors += torch.bincount(cur_edge_src, minlength=out_nodes)

        cur_start = cur_end

        del cur_out_irreps, summand

    if reduce == 'mean':
        div_factors = torch.clamp(div_factors, torch.finfo(_dtype).eps)
        final_out = final_out / div_factors[:, None]

    return final_out


def FCBlock(in_dim, hidden_dim, out_dim, layers, dropout, activation='relu'):
    activation = ACTIVATIONS[activation]
    assert layers >= 2
    sequential = [nn.Linear(in_dim, hidden_dim), activation(), nn.Dropout(dropout)]
    for i in range(layers - 2):
        sequential += [nn.Linear(hidden_dim, hidden_dim), activation(), nn.Dropout(dropout)]
    sequential += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*sequential)


class TensorProductConvLayer(torch.nn.Module):
    def __init__(self,
                 in_irreps,
                 sh_irreps,
                 out_irreps,
                 n_edge_features,
                 residual=True,
                 batch_norm=True,
                 dropout=0.0,
                 hidden_features=None,
                 faster=False,
                 edge_groups=[],
                 tp_weights_layers=2,
                 activation='relu',
                 depthwise=False):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        self.num_edge_groups = len(edge_groups)
        self.edge_groups = edge_groups
        self.out_size = irrep_to_size(out_irreps)
        self.depthwise = depthwise
        if hidden_features is None:
            hidden_features = n_edge_features

        if depthwise:
            in_irreps = o3.Irreps(in_irreps)
            sh_irreps = o3.Irreps(sh_irreps)
            out_irreps = o3.Irreps(out_irreps)

            irreps_mid = []
            instructions = []
            for i, (mul, ir_in) in enumerate(in_irreps):
                for j, (_, ir_edge) in enumerate(sh_irreps):
                    for ir_out in ir_in * ir_edge:
                        if ir_out in out_irreps:
                            k = len(irreps_mid)
                            irreps_mid.append((mul, ir_out))
                            instructions.append((i, j, k, "uvu", True))

            # We sort the output irreps of the tensor product so that we can simplify them
            # when they are provided to the second o3.Linear
            irreps_mid = o3.Irreps(irreps_mid)
            irreps_mid, p, _ = irreps_mid.sort()

            # Permute the output indexes of the instructions to match the sorted irreps:
            instructions = [
                (i_in1, i_in2, p[i_out], mode, train)
                for i_in1, i_in2, i_out, mode, train in instructions
            ]

            self.tp = o3.TensorProduct(
                in_irreps,
                sh_irreps,
                irreps_mid,
                instructions,
                shared_weights=False,
                internal_weights=False,
            )

            self.linear_2 = o3.Linear(
                # irreps_mid has uncoallesed irreps because of the uvu instructions,
                # but there's no reason to treat them seperately for the Linear
                # Note that normalization of o3.Linear changes if irreps are coallesed
                # (likely for the better)
                irreps_in=irreps_mid.simplify(),
                irreps_out=out_irreps,
                internal_weights=True,
                shared_weights=True,
            )

        else:
            if faster:
                print("Faster Tensor Product")
                self.tp = FasterTensorProduct(in_irreps, sh_irreps, out_irreps)
                # self.tp = torch.compile(self.tp)
            else:
                self.tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        if self.num_edge_groups == 0:
            self.fc = FCBlock(n_edge_features, hidden_features, self.tp.weight_numel, tp_weights_layers, dropout, activation)
        else:
            self.fc = nn.ModuleDict(
                {
                    key: FCBlock(n_edge_features, hidden_features, self.tp.weight_numel, tp_weights_layers, dropout, activation)
                    for key in edge_groups
                }
            )

        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean', edge_weight=1.0):
        if edge_index.shape[1] == 0 and node_attr.shape[0] == 0:
            raise ValueError("No edges and no nodes")

        _dtype = node_attr.dtype
        if edge_index.shape[1] == 0:
            out = torch.zeros((node_attr.shape[0], self.out_size), dtype=_dtype, device=node_attr.device)
        else:
            if self.num_edge_groups == 0:
                out = tp_scatter_simple(self.tp, self.fc, node_attr, edge_index, edge_attr, edge_sh,
                                        out_nodes, reduce, edge_weight)
            else:
                out = tp_scatter_multigroup(self.tp, self.fc, node_attr, edge_index, edge_attr, edge_sh,
                                            out_nodes, reduce, edge_weight)

            if self.depthwise:
                out = self.linear_2(out)

            if self.batch_norm:
                out = self.batch_norm(out)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        out = out.to(_dtype)
        return out


class TensorProductAggLayer(torch.nn.Module):
    def __init__(self,
                 in_irreps,
                 sh_irreps,
                 out_irreps,
                 n_edge_features,
                 residual=True,
                 batch_norm=True,
                 dropout=0.0,
                 hidden_features=None,
                 faster=False,
                 tp_weights_layers=2,
                 activation='relu',
                 depthwise=False):
        super(TensorProductAggLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        self.out_size = irrep_to_size(out_irreps)
        self.depthwise = depthwise
        if hidden_features is None:
            hidden_features = n_edge_features

        if depthwise:
            in_irreps = o3.Irreps(in_irreps)
            sh_irreps = o3.Irreps(sh_irreps)
            out_irreps = o3.Irreps(out_irreps)

            irreps_mid = []
            instructions = []
            for i, (mul, ir_in) in enumerate(in_irreps):
                for j, (_, ir_edge) in enumerate(sh_irreps):
                    for ir_out in ir_in * ir_edge:
                        if ir_out in out_irreps:
                            k = len(irreps_mid)
                            irreps_mid.append((mul, ir_out))
                            instructions.append((i, j, k, "uvu", True))

            # We sort the output irreps of the tensor product so that we can simplify them
            # when they are provided to the second o3.Linear
            irreps_mid = o3.Irreps(irreps_mid)
            irreps_mid, p, _ = irreps_mid.sort()

            # Permute the output indexes of the instructions to match the sorted irreps:
            instructions = [
                (i_in1, i_in2, p[i_out], mode, train)
                for i_in1, i_in2, i_out, mode, train in instructions
            ]

            self.tp = o3.TensorProduct(
                in_irreps,
                sh_irreps,
                irreps_mid,
                instructions,
                shared_weights=False,
                internal_weights=False,
            )

            self.linear_2 = o3.Linear(
                # irreps_mid has uncoallesed irreps because of the uvu instructions,
                # but there's no reason to treat them seperately for the Linear
                # Note that normalization of o3.Linear changes if irreps are coallesed
                # (likely for the better)
                irreps_in=irreps_mid.simplify(),
                irreps_out=out_irreps,
                internal_weights=True,
                shared_weights=True,
            )

        else:
            if faster:
                print("Faster Tensor Product")
                self.tp = FasterTensorProduct(in_irreps, sh_irreps, out_irreps)
            else:
                self.tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = FCBlock(n_edge_features, hidden_features, self.tp.weight_numel, tp_weights_layers, dropout, activation)
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None


    def forward(self,
                dst_node_attr,
                agg_node_attr,
                agg_index,
                edge_attr,
                edge_sh,
                out_nodes=None,
                reduce='mean',
                edge_weight=1.0):
        _device = agg_node_attr.device
        _dtype = agg_node_attr.dtype
        out_irreps = self.fc(edge_attr).to(_device).to(_dtype)
        out_irreps.mul_(edge_weight)
        tp = self.tp(agg_node_attr, edge_sh, out_irreps)
        out_nodes = out_nodes or dst_node_attr.shape[0]
        out = pygu.scatter(tp, agg_index, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.depthwise:
            out = self.linear_2(out)

        if self.batch_norm:
            out = self.batch_norm(out)

        if self.residual:
            padded = F.pad(dst_node_attr, (0, out.shape[-1] - dst_node_attr.shape[-1]))
            out = out + padded

        out = out.to(_dtype)
        return out


class TensorProductBroadcastLayer(torch.nn.Module):
    def __init__(self,
                 in_irreps,
                 sh_irreps,
                 out_irreps,
                 n_edge_features,
                 residual=True,
                 batch_norm=True,
                 dropout=0.0,
                 hidden_features=None,
                 faster=False,
                 tp_weights_layers=2,
                 activation='relu',
                 depthwise=False):
        super(TensorProductBroadcastLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        self.out_size = irrep_to_size(out_irreps)
        self.depthwise = depthwise
        if hidden_features is None:
            hidden_features = n_edge_features

        if depthwise:
            in_irreps = o3.Irreps(in_irreps)
            sh_irreps = o3.Irreps(sh_irreps)
            out_irreps = o3.Irreps(out_irreps)

            irreps_mid = []
            instructions = []
            for i, (mul, ir_in) in enumerate(in_irreps):
                for j, (_, ir_edge) in enumerate(sh_irreps):
                    for ir_out in ir_in * ir_edge:
                        if ir_out in out_irreps:
                            k = len(irreps_mid)
                            irreps_mid.append((mul, ir_out))
                            instructions.append((i, j, k, "uvu", True))

            # We sort the output irreps of the tensor product so that we can simplify them
            # when they are provided to the second o3.Linear
            irreps_mid = o3.Irreps(irreps_mid)
            irreps_mid, p, _ = irreps_mid.sort()

            # Permute the output indexes of the instructions to match the sorted irreps:
            instructions = [
                (i_in1, i_in2, p[i_out], mode, train)
                for i_in1, i_in2, i_out, mode, train in instructions
            ]

            self.tp = o3.TensorProduct(
                in_irreps,
                sh_irreps,
                irreps_mid,
                instructions,
                shared_weights=False,
                internal_weights=False,
            )

            self.linear_2 = o3.Linear(
                # irreps_mid has uncoallesed irreps because of the uvu instructions,
                # but there's no reason to treat them seperately for the Linear
                # Note that normalization of o3.Linear changes if irreps are coallesed
                # (likely for the better)
                irreps_in=irreps_mid.simplify(),
                irreps_out=out_irreps,
                internal_weights=True,
                shared_weights=True,
            )

        else:
            if faster:
                print("Faster Tensor Product")
                self.tp = FasterTensorProduct(in_irreps, sh_irreps, out_irreps)
            else:
                self.tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = FCBlock(n_edge_features, hidden_features, self.tp.weight_numel, tp_weights_layers, dropout, activation)
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None


    def forward(self,
                src_node_attr,
                bcast_node_attr,
                bcast_index,
                edge_attr,
                edge_sh,
                edge_weight=1.0):
        _device = src_node_attr.device
        _dtype = src_node_attr.dtype
        out_irreps = self.fc(edge_attr).to(_device).to(_dtype)
        out_irreps.mul_(edge_weight)
        out = self.tp(src_node_attr[bcast_index], edge_sh, out_irreps)

        if self.depthwise:
            out = self.linear_2(out)

        if self.batch_norm:
            out = self.batch_norm(out)

        if self.residual:
            padded = F.pad(bcast_node_attr, (0, out.shape[-1] - bcast_node_attr.shape[-1]))
            out = out + padded

        out = out.to(_dtype)
        return out



# class SeparableS2Activation(nn.Module):
#     def __init__(self, irreps, grid_res=10):
#         super().__init__()
#         self.irreps = irreps
#         self.num_scalars = self.irreps.count("1e")
#         self.scalar_act = torch.nn.SiLU()
#
#         s2_irreps = o3.Irreps((1, (l, 1)) for l in irreps.ls)
#         self.s2_act     = e3nn.nn.S2Activation(irreps=s2_irreps, act=self.scalar_act, grid_res)
#
#         multiplicities = [irreps.count(o3.Irrep((l, 1))) for l in irreps.ls]
#         assert np.all(np.array(multiplicities) == multiplicities[0])
#         self.multiplicity = multiplicities[0]
#
#     def forward(self, input_tensors):
#         input_components = [input_tensors[..., s] for s in self.irreps.sort().slices()]
#
#         input_scalars = input_components[0]
#         output_scalars = self.scalar_act(input_scalars)
#
#         input_spherical_tensors = torch.cat(
#             [
#                 component.view(-1, self.multiplicity, 2*l+1)
#                 for component, l in zip(input_components, self.irreps.ls)
#             ], dim=-1
#         )
#         output_spherical_tensors = self.s2_act(input_spherical_tensors)
#         output_components = output_spherical_tensors.split(
#             [2*l+1 for l in self.irreps.ls],
#             dim=-1
#         )
#         output_tensors = torch.cat(
#             [component.view(-1, self.multiplicity) for component in output_components],
#             dim=-1
#         )
#         outputs = torch.cat(
#             [output_scalars, output_tensors[..., self.num_scalars:]],
#             dim=-1
#         )
#         return outputs
#
#
# class FeedForward(nn.Module):
#     def __init__(self, in_irreps, h_multiplicity, out_irreps, bypass=True):
#         super().__init__()
#         self.in_irreps = in_irreps
#         self.h_irreps = h_multiplicity
#
#         ls = sorted(set(in_irreps.ls) + set(out_irreps.ls))
#         h_irreps = o3.Irreps([(h_multiplicity, (l, 1)) for l in ls])
#
#         self.lin1 = o3.Linear(in_irreps, h_irreps)
#         self.lin2 = o3.Linear(h_irreps, h_irreps)
#         self.lin3 = o3.Linear(h_irreps, out_irreps)
#         self.act = SeparableS2Activation(h_irreps, grid_res=18)
#
#         if bypass:
#             self.bypass = o3.Linear(in_irreps, out_irreps)
#         self.norm = e3nn.nn.BatchNorm(out_irreps)
#
#     def forward(self, features):
#         out = self.lin1(features)
#         out = self.act(out)
#         out = self.lin2(out)
#         out = self.act(out)
#         out = self.lin3(out)
#
#         if hasattr(self, "bypass"):
#             out = self.bypass(features) + out
#
#         return self.norm(out)


class EdgeUpdate(nn.Module):
    def __init__(self,
                 feat_irreps,
                 h_edge):
        super().__init__()
        self.feat_irreps = feat_irreps
        self.out_irreps = o3.Irreps([(h_edge, (0, 1))])

        self.lin = o3.Linear(
            feat_irreps,
            self.out_irreps
        )
        self.fc = nn.Sequential(
            nn.Linear(h_edge * 3, h_edge),
            nn.ReLU(),
            nn.Linear(h_edge, h_edge),
            nn.ReLU(),
            nn.Linear(h_edge, h_edge),
        )
        self.ln = nn.LayerNorm(h_edge)

    def forward(self,
                atom_features: torch.Tensor,
                edge_features: torch.Tensor,
                edge_index: torch.Tensor):

        atom_scalars = self.lin(atom_features)

        edge_dst, edge_src = edge_index
        edge_in = torch.cat([
            atom_scalars[edge_dst],
            atom_scalars[edge_src],
            edge_features
        ], dim=-1)
        edge_features = self.ln(edge_features + self.fc(edge_in))
        return edge_features


class TensorConvLayer(nn.Module):
    def __init__(self,
                 in_irreps,
                 sh_irreps,
                 out_irreps,
                 h_edge,
                 vpa=False
        ):
        super().__init__()
        self.in_irreps = in_irreps
        self.sh_irreps = sh_irreps
        self.out_irreps = out_irreps

        self.tp = FasterTensorProduct(
            in_irreps,
            sh_irreps,
            out_irreps,
            shared_weights=False
        )
        self.fc = nn.Sequential(
            nn.Linear(h_edge, h_edge),
            nn.ReLU(),
            nn.Linear(h_edge, self.tp.weight_numel)
        )
        self.norm = BatchNorm(out_irreps)
        self.vpa = vpa

    def forward(self,
                atom_features: torch.Tensor,
                edge_features: torch.Tensor,
                edge_sh,
                edge_index: torch.Tensor):

        edge_dst, edge_src = edge_index
        tp = self.tp(
            atom_features[edge_dst],
            edge_sh,
            self.fc(edge_features)
        )

        if self.vpa:
            out = pygu.scatter(tp, edge_src, dim=0, dim_size=atom_features.shape[0], reduce="sum")
            norm = pygu.scatter(torch.ones_like(edge_src), edge_src, dim=0, dim_size=atom_features.shape[0], reduce="sum")
            norm[norm < 1] = 1
            out = out / torch.sqrt(norm)
        else:
            out = pygu.scatter(tp, edge_src, dim=0, dim_size=atom_features.shape[0], reduce="mean")

        padded = F.pad(atom_features, (0, out.shape[-1] - atom_features.shape[-1]))
        out = out + padded
        out = self.norm(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, in_irreps, h_irreps, out_irreps, bypass=True):
        super().__init__()
        self.in_irreps = in_irreps
        self.h_irreps = h_irreps

        self.lin1 = o3.Linear(in_irreps, h_irreps)
        self.lin2 = o3.Linear(h_irreps, h_irreps)
        self.lin3 = o3.Linear(h_irreps, out_irreps)

        self.act = NormActivation(h_irreps, scalar_nonlinearity=torch.sigmoid)

        if bypass:
            self.bypass = o3.Linear(in_irreps, out_irreps)
        self.norm = BatchNorm(out_irreps)

    def forward(self, features):
        out = self.lin1(features)
        out = self.act(out)
        out = self.lin2(out)
        out = self.act(out)
        out = self.lin3(out)

        if hasattr(self, "bypass"):
            out = self.bypass(features) + out

        return self.norm(out)


def separate_heads_dim(irreps, irreps_tensor, num_heads):
    slices = (irreps * num_heads).slices()
    tensor_slices = [irreps_tensor[..., s] for s in slices]
    head_slices = [t.view(t.shape[:-1] + [num_heads, -1]) for t in tensor_slices]
    return torch.cat(head_slices, dim=-1)

def fuse_heads_dim(irreps, irreps_tensor, num_heads):
    slices = irreps.slices()
    head_slices = [irreps_tensor[..., s] for s in slices]
    tensor_slices = [t.view(t.shape[:-2] + [-1]) for t in head_slices]
    return torch.cat(tensor_slices, dim=-1)


class SE3AttentionUpdate(nn.Module):
    def __init__(self,
                 feat_irreps,
                 sh_irreps,
                 h_edge,
                 qk_irreps,
                 v_irreps,
                 num_heads=4):
        super().__init__()

        self.feat_irreps = feat_irreps
        self.sh_irreps = sh_irreps
        self.num_heads = num_heads

        self.qk_irreps = qk_irreps
        self.v_irreps = v_irreps

        self.k_tp = FasterTensorProduct(
            feat_irreps,
            sh_irreps,
            self.qk_irreps * num_heads,
            shared_weights=False
        )
        self.v_tp = FasterTensorProduct(
            feat_irreps,
            sh_irreps,
            self.v_irreps * num_heads,
            shared_weights=False
        )

        self.q_lin = o3.Linear(feat_irreps, self.qk_irreps * num_heads)
        self.k_fc = nn.Sequential(
            nn.Linear(h_edge, h_edge),
            nn.ReLU(),
            nn.Linear(h_edge, self.kv_tp.weight_numel)
        )
        self.k_fc = nn.Sequential(
            nn.Linear(h_edge, h_edge),
            nn.ReLU(),
            nn.Linear(h_edge, self.kv_tp.weight_numel)
        )
        self.lin_self = o3.Linear(feat_irreps, feat_irreps)
        self.lin_out = o3.Linear(v_irreps * num_heads, feat_irreps)
        self.norm = BatchNorm(feat_irreps)

    def forward(self,
                tfn_features: torch.Tensor,
                edge_features: torch.Tensor,
                edge_sh,
                edge_index: torch.Tensor):

        edge_dst, edge_src = edge_index

        q = self.q_lin(tfn_features)
        q = separate_heads_dim(self.qk_irreps, q, self.num_heads)
        k = self.tp(
            tfn_features[edge_dst],
            edge_sh,
            self.k_fc(edge_features)
        )
        k = separate_heads_dim(self.qk_irreps, k, self.num_heads)
        v = self.tp(
            tfn_features[edge_dst],
            edge_sh,
            self.v_fc(edge_features)
        )
        v = separate_heads_dim(self.v_irreps, v, self.num_heads)

        attn = torch.einsum("...j,...j->...", q, k)
        attn = pygu.softmax(attn, edge_src, dim=0)

        update = pygu.scatter(
            v * attn[..., None],
            edge_src,
            dim=0,
            dim_size=tfn_features.shape[0]
        )
        update = fuse_heads_dim(self.v_irreps, update, self.num_heads)
        update = self.lin_out(update)
        self_update = self.lin_self(tfn_features)

        return self.norm(self_update + update)