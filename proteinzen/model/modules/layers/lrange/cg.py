import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_cluster import knn, knn_graph, radius_graph
from torch_scatter import scatter_mean
import torch_geometric.utils as pygu

from e3nn import o3
from e3nn.nn import NormActivation, Gate, BatchNorm

from proteinzen.model.modules.common import GaussianRandomFourierBasis
from proteinzen.model.modules.layers.edge.tfn import FasterTensorProduct
from proteinzen.data.datasets.featurize.common import _edge_positional_embeddings, _rbf
from proteinzen.data.datasets.featurize.sidechain import _dihedrals, _ideal_virtual_Cb
from proteinzen.model.modules.openfold.layers import Linear, flatten_final_dims, ipa_point_weights_init_
from proteinzen.model.modules.layers.node.tfn import SE3AttentionUpdate, EdgeUpdate
from proteinzen.model.modules.layers.node.frame_tfn_cross import TFN2FrameCrossIPA, Frame2TFNCrossAttentionUpdate
from proteinzen.utils.openfold.rigid_utils import Rigid

class CoarseGrainUpdate(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        c_hidden,
        no_heads,
        no_qk_points,
        no_v_points,
        feat_irreps,
        sh_irreps,
        qk_irreps,
        v_irreps,
        num_rbf=16
    ):
        super().__init__()
        self.sh_irreps = sh_irreps
        self.num_rbf = num_rbf
        self.frame2tfn_edge_update = nn.Sequential(
            nn.Linear(c_z + num_rbf, c_z * 2),
            nn.ReLU(),
            nn.Linear(2*c_z, c_z)
        )
        self.frame2tfn_edge_ln = nn.LayerNorm(c_z)

        self.frame_to_tfn = Frame2TFNCrossAttentionUpdate(
            c_s,
            feat_irreps,
            sh_irreps,
            c_z,
            qk_irreps,
            v_irreps,
            num_heads=no_heads
        )
        self.tfn_to_tfn = SE3AttentionUpdate(
            feat_irreps,
            sh_irreps,
            c_z,
            qk_irreps,
            v_irreps,
            num_heads=no_heads
        )
        self.tfn_to_frame = TFN2FrameCrossIPA(
            c_s,
            c_z,
            c_hidden,
            no_heads,
            no_qk_points,
            no_v_points,
            feat_irreps,
        )
        self.edge_transition = EdgeUpdate(feat_irreps, c_z)


    def _gen_edge_features(self, frame_x, tfn_x, frame2tfn, tfn2tfn, tfn2frame, eps=1e-8):
        frame_dst, tfn_src = frame2tfn
        tfn_dst, frame_src = tfn2frame

        frame2tfn_edge_dist_vecs = frame_x[frame_dst] - tfn_x[tfn_src]
        frame2tfn_edge_dists = torch.linalg.vector_norm(frame2tfn_edge_dist_vecs + eps, dim=-1)

        frame2tfn_edge_rbf = _rbf(frame2tfn_edge_dists, D_min=0.0, D_max=20.0, D_count=self.num_rbf, device=frame2tfn.device)
        frame2tfn_edge_features = frame2tfn_edge_rbf
        frame2tfn_edge_sh = o3.spherical_harmonics(self.sh_irreps, frame2tfn_edge_dist_vecs, normalize=True, normalization='component')

        tfn2tfn_edge_dist_vecs = tfn_x[tfn2tfn[0]] - tfn_x[tfn2tfn[1]]
        tfn2tfn_edge_dists = torch.linalg.vector_norm(tfn2tfn_edge_dist_vecs + eps, dim=-1)

        tfn2tfn_edge_rbf = _rbf(tfn2tfn_edge_dists, D_min=0.0, D_max=20.0, D_count=self.num_rbf, device=tfn2tfn.device)
        tfn2tfn_edge_features = tfn2tfn_edge_rbf
        tfn2tfn_edge_sh = o3.spherical_harmonics(self.sh_irreps, tfn2tfn_edge_dist_vecs, normalize=True, normalization='component')

        tfn2frame_edge_dist_vecs = fn_x[tfn_dst] - frame_x[frame_src]
        tfn2frame_edge_dists = torch.linalg.vector_norm(tfn2frame_edge_dist_vecs + eps, dim=-1)

        tfn2frame_edge_rbf = _rbf(tfn2frame_edge_dists, D_min=0.0, D_max=20.0, D_count=self.num_rbf, device=tfn2frame.device)
        tfn2frame_edge_features = tfn2frame_edge_rbf
        tfn2frame_edge_sh = o3.spherical_harmonics(self.sh_irreps, tfn2frame_edge_dist_vecs, normalize=True, normalization='component')

        return (
            frame2tfn_edge_features,
            frame2tfn_edge_sh,
            tfn2tfn_edge_features,
            tfn2tfn_edge_sh,
            tfn2frame_edge_features,
            tfn2frame_edge_sh
        )


    def forward(self,
                frame_features,
                tfn_features,
                frame2tfn_edge_features,
                tfn2tfn_edge_features,
                tfn2frame_edge_features,
                rigids,
                frame2tfn_edge_index,
                tfn2tfn_edge_index,
                tfn2frame_edge_index,
                res_mask
                ):

        tfn_x = pygu.scatter(
            rigids.get_trans(),
            frame2tfn_edge_index[0],
            dim=0,
            dim_size=tfn_features.shape[0],
            reduce='mean'
        )

        (
            frame2tfn_edge_update,
            frame2tfn_edge_sh,
            tfn2tfn_edge_update,
            tfn2tfn_edge_sh,
            tfn2frame_edge_update,
            tfn2frame_edge_sh
        ) = self._gen_edge_features(
            rigids.get_trans(),
            tfn_x,
            frame2tfn_edge_index,
            tfn2tfn_edge_index,
            tfn2frame_edge_index
        )





