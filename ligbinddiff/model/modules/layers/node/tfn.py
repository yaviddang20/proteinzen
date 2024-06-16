import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import e3nn
from e3nn import o3
from e3nn.nn import NormActivation, Gate, BatchNorm
import torch_geometric.utils as pygu

from ligbinddiff.model.modules.layers.edge.tfn import FasterTensorProduct

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
                 feat_irreps,
                 sh_irreps,
                 h_edge,
                 update_edge=False):
        super().__init__()
        self.feat_irreps = feat_irreps
        self.sh_irreps = sh_irreps

        self.tp = FasterTensorProduct(
            feat_irreps,
            sh_irreps,
            feat_irreps,
            shared_weights=False
        )
        self.fc = nn.Sequential(
            nn.Linear(h_edge, h_edge),
            nn.ReLU(),
            nn.Linear(h_edge, self.tp.weight_numel)
        )
        self.norm = BatchNorm(feat_irreps)

        self.update_edge = update_edge
        if update_edge:
            self.edge_update = EdgeUpdate(
                feat_irreps=feat_irreps,
                h_edge=h_edge
            )

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

        out = pygu.scatter(tp, edge_src, dim=0, dim_size=atom_features.shape[0],reduce="mean")
        padded = F.pad(atom_features, (0, out.shape[-1] - atom_features.shape[-1]))
        out = out + padded
        out = self.norm(out)

        if self.update_edge:
            edge_features = self.edge_update(
                out,
                edge_features,
                edge_index
            )

        return out, edge_features


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
    head_slices = [t.view(t.shape[:-1] + [num_heads, -1]) for t in tensor_slices])
    return torch.cat(head_slices, dim=-1)

def fuse_heads_dim(irreps, irreps_tensor, num_heads):
    slices = irreps.slices()
    head_slices = [irreps_tensor[..., s] for s in slices]
    tensor_slices = [t.view(t.shape[:-2] + [-1]) for t in head_slices])
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