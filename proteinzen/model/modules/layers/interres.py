import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.utils as pygu

from proteinzen.model.modules.layers.node.attention import GraphInvariantPointAttention
from proteinzen.model.modules.layers.edge.sitewise import EdgeTransition
from proteinzen.model.modules.openfold.layers import Linear
from proteinzen.utils.openfold import rigid_utils as ru

from proteinzen.stoch_interp.interpolate.so3_utils import rotmat_to_rotvec

# inspired by Chroma
def average_rotations(R,
                      R_probs,
                      index,
                      num_nodes,
                      dither_eps=1e-4):
    # Average rotation via SVD
    R_avg_unc = pygu.scatter(
        R * R_probs[..., None],
        index,
        dim=0,
        dim_size=num_nodes,
        reduce='sum')
    R_avg_unc = R_avg_unc + dither_eps * torch.randn_like(R_avg_unc)
    U, S, Vh = torch.linalg.svd(R_avg_unc, full_matrices=True)
    R_avg = U @ Vh

    # Enforce that matrix is rotation matrix
    d = torch.linalg.det(R_avg)
    d_expand = F.pad(d[..., None, None], (2, 0), value=1.0)
    Vh = Vh * d_expand
    R_avg = U @ Vh
    return R_avg

# from https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Unit_quaternions
def rotvec_to_quat(rotvec, eps=1e-8):
    theta = torch.linalg.vector_norm(rotvec + eps, dim=-1)
    omega = rotvec / theta[..., None]
    quat = torch.cat(
        [torch.cos(theta/2)[..., None], torch.sin(theta/2)[..., None] * omega], dim=-1
    )
    return quat


def rigid_quat_compose(rigid1, rigid2):
    # we manually compose this since we need to keep the rotations as quats
    rigid1_quats = rigid1.get_rots().get_quats()
    rigid2_quats = rigid2.get_rots().get_quats()
    quats = ru.quat_multiply(rigid1_quats, rigid2_quats)
    trans = rigid1.get_rots().apply(rigid2.get_trans()) + rigid1.get_trans()
    out_rigids = ru.Rigid(
        rots=ru.Rotation(quats=quats),
        trans=trans
    )
    return out_rigids


class OneParamPairwiseEquilibrate(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_points,
                 num_v_points,
                 num_equilibration_steps=3
                 ):
        super().__init__()

        self.ipa = GraphInvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=num_heads,
            no_qk_points=num_qk_points,
            no_v_points=num_v_points,
        )
        self.ln = nn.LayerNorm(c_s)
        self.edge_update = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z
        )
        self.pairwise_update = Linear(c_z, 6, init='final')
        self.logits = nn.Linear(c_z, 1)
        self.num_equilibration_steps = num_equilibration_steps

    def forward(self,
                node_features,
                edge_features,
                edge_index,
                rigids: ru.Rigid,
                node_mask
                ):
        node_update = self.ipa(
            node_features,
            edge_features,
            edge_index,
            rigids,
            node_mask)
        node_features = self.ln(node_features + node_update * node_mask[..., None])
        edge_features = self.edge_update(
            node_features,
            edge_features,
            edge_index
        )
        pairwise_q_update_vec = self.pairwise_update(edge_features)
        pairwise_logits = self.logits(edge_features)
        pairwise_weights = pygu.softmax(
            pairwise_logits,
            edge_index[1],
            num_nodes=node_features.shape[0],
            dim=0
        )

        src_rigids = rigids[edge_index[1]]
        dst_rigids = rigids[edge_index[0]]
        inv_dst_rigids = dst_rigids.invert()

        # we manually compose this since we need to keep the rotations as quats
        transforms_from_dst = rigid_quat_compose(inv_dst_rigids, src_rigids)
        pred_pairwise_transforms = transforms_from_dst.compose_q_update_vec(pairwise_q_update_vec)
        rigids = self.equilibrate(rigids, pred_pairwise_transforms, pairwise_weights, edge_index)

        return node_features, rigids

    def equilibrate(self, rigids, pairwise_transforms, pairwise_weights, edge_index):
        rots = rigids.get_rots()
        trans = rigids.get_trans()
        for _ in range(self.num_equilibration_steps):
            pairwise_rots = pairwise_transforms.get_rots()
            dst_rots = rots[edge_index[0]]
            rot_src_pred_from_dst = dst_rots.compose_r(pairwise_rots)
            rot_mat_consensus = average_rotations(
                rot_src_pred_from_dst.get_rot_mats(),
                pairwise_weights,
                edge_index[1],
                num_nodes=rigids.shape[0]
            )
            rots = ru.Rotation(rot_mats=rot_mat_consensus)

            pairwise_trans = pairwise_transforms.get_trans()
            trans_src_pred_from_dst = trans[edge_index[0]] + dst_rots.apply(pairwise_trans)
            trans = pygu.scatter(
                trans_src_pred_from_dst * pairwise_weights,
                edge_index[1],
                dim_size=rigids.shape[0],
                reduce='sum'
            )

        # a roundabout but numerically stable(?) way to convert rotation matrices to quats
        rotvecs = rotmat_to_rotvec(rots.get_rot_mats())
        quats = rotvec_to_quat(rotvecs)

        rigids = ru.Rigid(
            rots=ru.Rotation(quats=quats),
            trans=trans
        )

        return rigids
