import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_cluster import knn, knn_graph, radius_graph
from torch_scatter import scatter_mean
import torch_geometric.utils as pygu

from e3nn import o3
from e3nn.nn import NormActivation, Gate, BatchNorm

from ligbinddiff.model.modules.layers.node.mpnn import IPMP

from ligbinddiff.model.modules.common import GaussianRandomFourierBasis
from ligbinddiff.model.modules.layers.edge.tfn import FasterTensorProduct
from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings, _rbf
from ligbinddiff.data.datasets.featurize.sidechain import _dihedrals, _ideal_virtual_Cb
from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.utils.atom_reps import letter_to_num
from ligbinddiff.utils.atom_reps import atom14_atomic_row, atom14_atomic_period, restype_3to1, atom_to_atomic_period, atom_to_atomic_row


def atom14_atom_type_embedding(seq, letter2num, max_period=None, max_row=None):
    device = seq.device
    if max_period is None:
        max_period = max(atom_to_atomic_period.values())
    if max_row is None:
        max_row = max(atom_to_atomic_row.values())
    one_hot_period = torch.eye(max_period+1, device=device)
    one_hot_row = torch.eye(max_row+1, device=device)

    # add one to 1 index, then add 1 for X
    num_aa = max(letter2num.values()) + 2

    period_store = torch.zeros(
        num_aa,
        14,
        device=device
    )
    row_store = torch.zeros(
        num_aa,
        14,
        device=device
    )
    for aa, periods in atom14_atomic_period.items():
        aa_idx = letter_to_num[restype_3to1[aa]]
        period_store[aa_idx] = torch.as_tensor(periods, device=device)
    for aa, row in atom14_atomic_row.items():
        aa_idx = letter_to_num[restype_3to1[aa]]
        row_store[aa_idx] = torch.as_tensor(row, device=device)

    atom_periods = period_store[seq]
    atom_rows = row_store[seq]
    period_embedding = one_hot_period[atom_periods.long()]
    row_embedding = one_hot_row[atom_rows.long()]
    atom_type_embedding = torch.cat([period_embedding, row_embedding], dim=-1)  # n_nodes x 14 x n_period+n_row

    return atom_type_embedding  # n_atoms x 14 x n_period+n_row



def full_cross_sidechain_graph(atom14_mask,
                               invariant_points_mask,
                               res_edge_index):
                               #edge_subset_select):
    device = atom14_mask.device
    contains_invariant_points = invariant_points_mask.any(dim=-1)
    atom_mask = atom14_mask.clone()
    atom_mask[contains_invariant_points, 4:] = False
    atom_mask[invariant_points_mask] = True

    # # add self-edges to allow for attention within blocks
    # res_edge_index, res_edge_features = pygu.add_remaining_self_loops(
    #     res_edge_index,
    #     res_edge_features,
    #     fill_value=0.
    # )

    num_atoms = torch.sum(atom_mask)
    atom_idx = torch.arange(num_atoms, device=device)
    atom14_idx = torch.zeros_like(atom14_mask).long()
    atom14_idx[atom_mask] = atom_idx
    # res_idx = torch.arange(atom14_mask.shape[0], device=device)[..., None].expand(-1, 14)
    # atom_res_idx = res_idx[atom14_mask]

    src = res_edge_index[1]
    dst = res_edge_index[0]

    atom_src_idx = atom14_idx[src][..., None].expand(-1, -1, 14)
    atom_dst_idx = atom14_idx[dst][..., None, :].expand(-1, 14, -1)
    cross_atom_idx = torch.stack([
        atom_dst_idx, atom_src_idx
    ], dim=-1)
    atom_src_mask = atom_mask[src][..., None].expand(-1, -1, 14)
    atom_dst_mask = atom_mask[dst][..., None, :].expand(-1, 14, -1)
    cross_atom_mask = atom_src_mask & atom_dst_mask
    cross_atom_edges = cross_atom_idx[cross_atom_mask].transpose(-1, -2)

    ca_idx = torch.zeros_like(atom14_mask).bool()
    ca_idx[:, 1] = True
    ca_idx = ca_idx & atom_mask
    ca_select = ca_idx[atom_mask]
    invariant_points_select = invariant_points_mask[atom_mask]

    res_edge_id = torch.arange(res_edge_index.shape[1], device=device)
    atom_edge_to_res_edge = res_edge_id[..., None, None].expand(-1, 14, 14)[cross_atom_mask]

    return (
        atom_mask,
        ca_select,
        invariant_points_select,
        cross_atom_edges,
        atom_edge_to_res_edge
    )


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

        # self.tp = o3.FullyConnectedTensorProduct(
        #     feat_irreps,
        #     sh_irreps,
        #     feat_irreps,
        #     shared_weights=False
        # )
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


class SE3Attention(nn.Module):
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
        self.lin_self = o3.Linear(feat_irreps, feat_irreps)

        self.q_lin = o3.Linear(feat_irreps, feat_irreps)
        self.k_fc = nn.Sequential(
            nn.Linear(h_edge, h_edge),
            nn.ReLU(),
            nn.Linear(h_edge, self.tp.weight_numel)
        )
        self.v_fc = nn.Sequential(
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

        atom_q = self.q_lin(atom_features)[edge_src]
        atom_k = self.tp(
            atom_features[edge_dst],
            edge_sh,
            self.k_fc(edge_features)
        )
        atom_v = self.tp(
            atom_features[edge_dst],
            edge_sh,
            self.v_fc(edge_features)
        )
        atom_attn = torch.einsum("...j,...j->...", atom_q, atom_k)
        atom_attn = pygu.softmax(atom_attn, edge_src, dim=0)

        atom_update = pygu.scatter(
            atom_v * atom_attn[..., None],
            edge_src,
            dim=0,
            dim_size=atom_features.shape[0]
        )
        atom_features = atom_features + atom_update
        atom_features = self.norm(atom_features)

        if self.update_edge:
            edge_features = self.edge_update(
                atom_features,
                edge_features,
                edge_index
            )

        return atom_features, edge_features


class UpdateLayer(nn.Module):
    def __init__(self,
                 feat_irreps,
                 sh_irreps,
                 h_edge,
                 update_edge=False):
        super().__init__()
        self.conv = TensorConvLayer(
            feat_irreps=feat_irreps,
            sh_irreps=sh_irreps,
            h_edge=h_edge,
            update_edge=update_edge
        )
        # self.conv = SE3Attention(
        #     feat_irreps=feat_irreps,
        #     sh_irreps=sh_irreps,
        #     h_edge=h_edge,
        #     update_edge=update_edge
        # )
        self.ffn = FeedForward(
            feat_irreps,
            feat_irreps,
            feat_irreps,
            bypass=True,
        )
        self.update_invariant_points = o3.Linear(
            feat_irreps,
            o3.Irreps([(1, (1, -1))])
        )

    def forward(self,
                atom_features,
                atom_coords,
                atom_edge_features,
                edge_sh,
                atom_edge_index,
                invariant_point_select):
        atom_features, atom_edge_features = torch.utils.checkpoint.checkpoint(
            self.conv,
            atom_features,
            atom_edge_features,
            edge_sh,
            atom_edge_index,
            use_reentrant=False
        )
        atom_features = self.ffn(atom_features)

        invariant_points = atom_features[invariant_point_select]
        invariant_point_trans = self.update_invariant_points(invariant_points)
        invariant_point_update = torch.zeros_like(atom_coords)
        invariant_point_update[invariant_point_select] = invariant_point_trans
        atom_coords = atom_coords + invariant_point_update

        return atom_features, atom_coords, atom_edge_features


class NodeUpdate(nn.Module):
    def __init__(self, atom_irreps, c_node):
        super().__init__()
        self.update = o3.Linear(
            atom_irreps,
            o3.Irreps([(c_node, (0, 1))])
        )
        self.ln = nn.LayerNorm(c_node)

    def forward(self, atom_features, node_features, ca_select, atom_mask):
        node_update = self.update(atom_features[ca_select])
        node_features = node_features.clone()
        node_features[atom_mask[:, 1].bool()] += node_update
        node_features = self.ln(node_features)

        return node_features

class CAlphaUpdate(nn.Module):
    def __init__(self, atom_irreps, c_node):
        super().__init__()
        self.num_scalars = atom_irreps.count("0e")
        self.num_vectors = atom_irreps.count("1o")
        self.update_scalars = nn.Linear(
            c_node,
            self.num_scalars
        )
        self.update_vectors = nn.Linear(
            c_node,
            self.num_vectors * 3
        )
        self.bn = BatchNorm(atom_irreps)

    def forward(self, atom_features, node_features, rigids, ca_select, atom_mask):
        scalar_update = self.update_scalars(node_features)
        local_vector_update = self.update_vectors(node_features).view(-1, self.num_vectors, 3)
        global_vector_update = rigids[..., None].apply(local_vector_update)
        irrep_update = torch.cat([scalar_update, global_vector_update.view(-1, self.num_vectors * 3)], dim=-1)
        atom_features = atom_features.clone()
        atom_features[ca_select] += irrep_update[atom_mask[:, 1].bool()]
        atom_features = self.bn(atom_features)

        return atom_features


class EmbedNode(nn.Module):
    def __init__(self,
                 in_irreps,
                 out_irreps,
                 sh_irreps,
                 h_edge):
        super().__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps

        self.tp = o3.FullyConnectedTensorProduct(
            in_irreps,
            sh_irreps,
            out_irreps,
            shared_weights=False
        )
        self.fc = nn.Sequential(
            nn.Linear(h_edge, 2*h_edge),
            nn.ReLU(),
            nn.Linear(2*h_edge, self.tp.weight_numel)
        )
        self.norm = BatchNorm(out_irreps)

        self.bypass = o3.Linear(in_irreps, out_irreps)


    def forward(self,
                node_features: torch.Tensor,
                edge_features: torch.Tensor,
                edge_sh,
                edge_index: torch.Tensor):
        edge_dst, edge_src = edge_index
        tp = self.tp(
            node_features[edge_dst],
            edge_sh,
            self.fc(edge_features)
        )

        out = scatter_mean(tp, edge_src, dim=0, dim_size=node_features.shape[0])
        out = out + self.bypass(node_features)
        out = self.norm(out)
        return out


class PairwiseEmbedding(nn.Module):
    def __init__(self,
                 num_rbf=16,
                 num_pos_embed=16,
                 dist_clip=10,
                 scale=1,
                 D_min=2.0,
                 D_max=22.0,
                 num_aa=20,
                 classic_mode=False):
        super().__init__()
        self.dist_clip = dist_clip
        self.num_aa = num_aa + 1
        self.num_rbf = num_rbf
        self.num_pos_embed = num_pos_embed

        self.scale = scale
        self.D_min = D_min
        self.D_max = D_max

        self.classic_mode = classic_mode

        if not classic_mode:
            self.out_dim = (
                2                       # mask bits
                + self.num_aa * 2       # src/dst seq
                + (5 * 5) * num_rbf     # bb x bb dist
                + num_pos_embed         # rel pos embed
                + (4 * 3) * 2           # src/dst bb coords
            )
        else:
            self.out_dim = (
                + (5 * 5) * num_rbf     # bb x bb dist
                + num_pos_embed         # rel pos embed
            )

    def forward(self, data, edge_index, eps=1e-8):
        res_data = data['residue']
        num_edges = edge_index.shape[1]
        noising_mask = res_data['noising_mask']

        dst = edge_index[0]
        src = edge_index[1]
        noised_src = noising_mask[src]
        noised_dst = noising_mask[dst]
        unnoised_edges = ~(noised_src | noised_dst)
        # TODO: fix this inconsistency
        if "rigids_1" in res_data:
            rigids = ru.Rigid.from_tensor_7(data['residue'].rigids_1)
        else:
            rigids = ru.Rigid.from_tensor_7(data['residue'].rigids_0)

        if not self.classic_mode:

            edge_features = [
                noised_dst.float()[..., None],
                noised_src.float()[..., None]
            ]  # 2

            seq = res_data['seq']
            src_seq = F.one_hot(seq[src], num_classes=self.num_aa)
            dst_seq = F.one_hot(seq[dst], num_classes=self.num_aa)
            edge_features.append(src_seq * ~noised_src[..., None])  # 21
            edge_features.append(dst_seq * ~noised_dst[..., None])  # 21
        else:
            edge_features = []

        atom14 = res_data["atom14_gt_positions"]
        bb = atom14[..., :4, :]
        virtual_Cb = _ideal_virtual_Cb(bb)
        bb = torch.cat([bb, virtual_Cb[..., None, :]], dim=-2)

        ## bb edge distances
        edge_bb_src = bb[src]
        edge_bb_dst = bb[dst]
        edge_bb_dists = torch.linalg.vector_norm(
            edge_bb_src[..., None, :] - edge_bb_dst[..., None, :, :] + eps,
            dim=-1)
        edge_bb_dists = edge_bb_dists.view(edge_index.shape[1], -1, 1)
        edge_rbf = _rbf(edge_bb_dists, D_min=self.D_min, D_max=self.D_max, D_count=self.num_rbf, device=edge_index.device)
        edge_rbf = edge_rbf.view(num_edges, -1)
        edge_features.append(edge_rbf)  # 25 * 16

        ## edge rel pos embedding
        edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=self.num_pos_embed, device=edge_index.device)
        edge_features.append(edge_dist_rel_pos)  # 16

        if not self.classic_mode:
            ## pairwise sidechain atom distances
            src_bb_local = rigids[..., None].invert_apply(bb)[src]
            dst_bb_in_src_local = rigids[src, None].invert_apply(bb[dst])
            edge_features.append(
                src_bb_local.view(num_edges, -1)
            )  # 42
            edge_features.append(
                dst_bb_in_src_local.view(num_edges, -1)
            )  # 42

        # total 596
        return torch.cat(edge_features, dim=-1).float()



class AtomicEncoder(nn.Module):
    def __init__(self,
                 atom_s=16,
                 atom_v=8,
                 c_s=128,
                 c_z=128,
                 c_out=128,
                 num_points=4,
                 feat_lmax=1,
                 sh_lmax=1,
                 in_v=0,
                 num_rbf=64,
                 num_pos_embed=16,
                 num_layers=4,
                 n_aa=20,
                 scalarize_wrt_frames=True,
                 k=30,
                 classic_mode=False
                 ):
        super().__init__()
        self.n_aa = n_aa + 1
        self.atom_s = atom_s
        self.atom_v = atom_v
        self.c_out = c_out
        self.num_points = num_points
        self.classic_mode = classic_mode
        max_period = max(atom_to_atomic_period.values())
        max_row = max(atom_to_atomic_row.values())
        atom_in_s = max_period + max_row + 2

        self.in_irreps = o3.Irreps([
            (atom_in_s, (0, 1)),
            (in_v, (1, -1))
        ])
        self.feat_irreps = o3.Irreps(f"{atom_s}x0e+{atom_v}x1o")
        self.sh_irreps = o3.Irreps("1x0e+1x1o")
        self.num_rbf = num_rbf
        self.num_pos_embed = num_pos_embed
        self.k = k

        self.atom_embed = EmbedNode(self.in_irreps, self.feat_irreps, self.sh_irreps, num_rbf)

        if classic_mode:
            self.c_atomic = 0
        else:
            self.c_atomic = (
                + self.n_aa  # seq identity
                + (4 * 3)  # bb atoms
                + 1  # mask position
            )
        c_s_in=6

        self.embed_node = nn.Sequential(
            nn.Linear(c_s_in + self.c_atomic, 2*c_s),
            nn.ReLU(),
            nn.Linear(2*c_s, 2*c_s),
            nn.ReLU(),
            nn.Linear(2*c_s, c_s),
            nn.LayerNorm(c_s),
        )
        num_aa=20
        self.init_edge_embed = PairwiseEmbedding(
            num_rbf=self.num_rbf,
            num_pos_embed=self.num_pos_embed,
            num_aa=num_aa,
            classic_mode=classic_mode
        )
        self.embed_edge = nn.Sequential(
            nn.Linear(self.init_edge_embed.out_dim, 2*c_z),
            nn.ReLU(),
            nn.Linear(2*c_z, 2*c_z),
            nn.ReLU(),
            nn.Linear(2*c_z, c_z),
            nn.LayerNorm(c_z),
        )

        self.tc = nn.ModuleList(
            [
                UpdateLayer(
                    self.feat_irreps,
                    self.sh_irreps,
                    num_rbf,
                    update_edge=False
                )
                for _ in range(num_layers)
            ]
        )
        self.ipmp = nn.ModuleList(
            [
                IPMP(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_s
                )
                for _ in range(num_layers)
            ]
        )
        self.node_update = nn.ModuleList(
            [
                NodeUpdate(
                    atom_irreps=self.feat_irreps,
                    c_node=c_s,
                )
                for _ in range(num_layers)
            ]
        )
        self.ca_update = nn.ModuleList(
            [
                CAlphaUpdate(
                    atom_irreps=self.feat_irreps,
                    c_node=c_s,
                )
                for _ in range(num_layers)
            ]
        )

        self.invariant_point_feat_init = nn.Parameter(
            torch.randn((self.num_points, atom_in_s))
        )
        self.invariant_point_init = nn.Parameter(
            torch.randn((self.num_points, 3))
        )

        # self.scalarize_wrt_frames = scalarize_wrt_frames
        # if scalarize_wrt_frames:
        #     self.output_mu = nn.Linear(self.feat_irreps.dim, c_out)
        #     self.output_logvar = nn.Linear(self.feat_irreps.dim, c_out)
        # else:
        #     self.output_mu = o3.Linear(self.feat_irreps, o3.Irreps((c_out, (0, 1))))
        #     self.output_logvar = o3.Linear(self.feat_irreps, o3.Irreps((c_out, (0, 1))))
        self.output_mu = nn.Linear(c_s, c_out)
        self.output_logvar = nn.Linear(c_s, c_out)

    @torch.no_grad
    def _prep_features(self, graph, eps=1e-8):
        res_data = graph['residue']
        mgm_mask = res_data['mlm_mask']

        # edge graph
        X_ca = res_data['x'].float()
        res_mask = res_data['res_mask']
        masked_X_ca = X_ca.clone()
        masked_X_ca[~res_mask] = torch.inf
        edge_index = knn_graph(masked_X_ca, self.k, graph['residue'].batch, loop=True)

        atom14_features = atom14_atom_type_embedding(
            res_data['seq'],
            letter_to_num
        ).float()
        atom14_features[..., 4:, :] *= mgm_mask[..., None, None]
        # add invariant point initial features
        atom14_features[~mgm_mask, 4:4+self.num_points, :] = self.invariant_point_feat_init[None].expand(
            (~mgm_mask).sum(), -1, -1)

        atom14_coords = res_data['atom14_gt_positions'].float()
        atom14_mask = res_data['atom14_gt_exists'].bool()
        atom14_coords[..., 4:, :] *= mgm_mask[..., None, None]
        atom14_mask[..., 4:] *= mgm_mask[..., None]
        # add invariant points
        # TODO: fix this inconsistency
        if "rigids_1" in res_data:
            rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_1)
        else:
            rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_0)
        invariant_points_local = self.invariant_point_init[None].expand(atom14_features.shape[0], -1, -1)
        invariant_points_global = rigids[..., None].apply(invariant_points_local)
        atom14_coords[~mgm_mask, 4:4+self.num_points, :] = invariant_points_global[~mgm_mask]

        invariant_points_mask = torch.zeros_like(atom14_mask).bool()
        invariant_points_mask[mgm_mask, 4:4+self.num_points] = True
        invariant_points_mask[~res_mask] = False

        (
            all_atom_mask,
            ca_select,
            invariant_points_select,
            cross_atom_edges,
            atom_edge_to_res_edge
        ) = full_cross_sidechain_graph(
            atom14_mask,
            invariant_points_mask,
            edge_index
        )
        all_atom_features = atom14_features[all_atom_mask]
        all_atom_coords = atom14_coords[all_atom_mask]

        return all_atom_features, all_atom_coords, cross_atom_edges, ca_select, invariant_points_select

    def _prep_ipmp_features(self, graph):
        res_data = graph['residue']
        res_mask = res_data['res_mask']

        # node features
        X_ca = res_data['x'].float()
        bb = res_data['bb'].float()
        dihedrals = _dihedrals(bb)

        mask = res_data['mlm_mask']
        # TODO: fix this inconsistency
        if "rigids_1" in res_data:
            rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_1)
        else:
            rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_0)

        if not self.classic_mode:
            seq = F.one_hot(res_data['seq'], num_classes=self.n_aa).to(mask.device)
            masked_seq = seq * mask[..., None]

            node_scalars = torch.cat(
                [
                    dihedrals,
                    mask.float()[..., None],
                    masked_seq
                ],
            dim=-1)
            node_vectors = rigids[..., None].invert_apply(bb)
            # node_vectors = node_vectors * mask[..., None, None]
            node_features = torch.cat(
                [node_scalars, node_vectors.view([node_scalars.shape[0], -1])],
                dim=-1
            ).float()
            node_features = node_features * res_mask[..., None]
        else:
            node_features = dihedrals * res_mask[..., None]

        # edge graph
        masked_X_ca = X_ca.clone()
        masked_X_ca[~res_mask] = torch.inf
        edge_index = knn_graph(masked_X_ca, self.k, graph['residue'].batch)

        # # edge features
        # virtual_Cb = _ideal_virtual_Cb(bb)
        # bb = torch.cat([bb, virtual_Cb[..., None, :]], dim=-2)
        src = edge_index[1]
        dst = edge_index[0]

        # ## edge distances
        # edge_bb_src = bb[src]
        # edge_bb_dst = bb[dst]
        # edge_bb_dists = torch.linalg.vector_norm(
        #     edge_bb_src[..., None, :] - edge_bb_dst[..., None, :, :] + eps,
        #     dim=-1)
        # edge_bb_dists = edge_bb_dists.view(edge_index.shape[1], -1, 1)
        # edge_rbf = _rbf(edge_bb_dists, D_min=2.0, D_max=22.0, D_count=self.num_rbf, device=edge_index.device)
        # edge_rbf = edge_rbf.view(edge_index.shape[1], -1)
        # ## edge rel pos embedding
        # edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=self.num_pos_embed, device=edge_index.device)
        edge_features = self.init_edge_embed(graph, edge_index)
        # technically this shouldn't be necessary but just to be safe
        edge_mask = res_mask[src] & res_mask[dst]
        edge_features = edge_features * edge_mask[..., None]

        # edge_features = torch.cat([edge_rbf, edge_dist_rel_pos, local_edge_dir_feats], dim=-1)
        # edge_features = torch.cat([edge_rbf, edge_dist_rel_pos], dim=-1)

        return node_features, edge_features, edge_index, rigids

    def _gen_edge_features(self, all_atom_coords, cross_atom_edges, ca_select, eps=1e-8, dist_thresh=6.0):
        src = cross_atom_edges[1]
        dst = cross_atom_edges[0]
        edge_contains_ca = ca_select[src] | ca_select[dst]

        # edge features
        edge_features = []

        edge_dist_vecs = all_atom_coords[dst] - all_atom_coords[src]
        edge_dists = torch.linalg.vector_norm(edge_dist_vecs + eps, dim=-1)

        # remove edges which are below the threshold
        edge_filter = edge_dists < dist_thresh
        # include edges_which connect to CA, since this is what connects the atom graph to the residue graph
        edge_filter = edge_filter | edge_contains_ca
        edge_dists = edge_dists[edge_filter]
        edge_dist_vecs = edge_dist_vecs[edge_filter]
        filtered_edge_index = cross_atom_edges[:, edge_filter]

        edge_rbf = _rbf(edge_dists, D_min=0.0, D_max=dist_thresh, D_count=self.num_rbf, device=cross_atom_edges.device)
        edge_rbf = edge_rbf.view(filtered_edge_index.shape[1], -1)
        edge_features.append(edge_rbf)
        edge_features = torch.cat(edge_features, dim=-1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_dist_vecs, normalize=True, normalization='component')

        return edge_features, edge_sh, filtered_edge_index

    def forward(self, graph):
        res_mask = graph['residue']['res_mask']
        atom14_mask = graph['residue']['atom14_gt_exists']

        atom_features, atom_coords, cross_atom_edges, ca_select, invariant_points_select = self._prep_features(graph)
        edge_features, edge_sh, edge_index = self._gen_edge_features(atom_coords, cross_atom_edges, ca_select)

        atom_features = self.atom_embed(atom_features, edge_features, edge_sh, edge_index)

        node_features, ipmp_edge_features, ipmp_edge_index, rigids = self._prep_ipmp_features(graph)
        node_features = self.embed_node(node_features)
        ipmp_edge_features = self.embed_edge(ipmp_edge_features)

        for ca_update, tc, ipmp, node_update in zip(self.ca_update, self.tc, self.ipmp, self.node_update):
            atom_features = ca_update(atom_features, node_features, rigids, ca_select, atom14_mask)
            atom_features, atom_coords, edge_features = tc(
                atom_features,
                atom_coords,
                edge_features,
                edge_sh,
                edge_index,
                invariant_points_select)
            node_features = node_update(atom_features, node_features, ca_select, atom14_mask)
            node_features, ipmp_edge_features = ipmp(
                s=node_features,
                z=ipmp_edge_features,
                edge_index=ipmp_edge_index,
                r=rigids,
                mask=res_mask
            )
            edge_features, edge_sh, edge_index = self._gen_edge_features(atom_coords, cross_atom_edges, ca_select)

        latent_mu = self.output_mu(node_features)
        latent_logvar = self.output_logvar(node_features)

        out_dict = {}
        out_dict['latent_mu'] = latent_mu
        out_dict['latent_logvar'] = latent_logvar
        return out_dict
