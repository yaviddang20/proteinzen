import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_cluster import knn, knn_graph, radius_graph
from torch_scatter import scatter_mean
import torch_geometric.utils as pygu

from e3nn import o3
from e3nn.nn import NormActivation, Gate, BatchNorm

from ligbinddiff.model.modules.common import GaussianRandomFourierBasis
from ligbinddiff.model.modules.layers.edge.tfn import FasterTensorProduct
from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings, _rbf
from ligbinddiff.utils.openfold import rigid_utils as ru


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

        out = scatter_mean(tp, edge_src, dim=0, dim_size=atom_features.shape[0])
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


