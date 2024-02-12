import torch
from torch import nn
import torch.nn.functional as F

from ligbinddiff.data.datasets.featurize.sidechain import _ideal_virtual_Cb, _rbf, _edge_positional_embeddings
from ligbinddiff.utils.openfold import rigid_utils as ru


class PairwiseAtomicEmbedding(nn.Module):
    def __init__(self,
                 num_rbf=16,
                 dist_clip=10,
                 num_aa=20):
        super().__init__()
        self.dist_clip = dist_clip
        self.num_aa = num_aa + 1
        self.num_rbf = num_rbf

        # TODO: compute this based on input params
        self.out_dim = 596

    def forward(self, data, edge_index, eps=1e-8):
        res_data = data['residue']
        num_edges = edge_index.shape[1]
        noising_mask = res_data['noising_mask']

        dst = edge_index[0]
        src = edge_index[1]
        noised_src = noising_mask[src]
        noised_dst = noising_mask[dst]
        unnoised_edges = ~(noised_src | noised_dst)

        atom14 = res_data["atom14_gt_positions"]
        rigids = ru.Rigid.from_tensor_7(res_data['rigids_1'])

        edge_features = [
            noised_dst.float()[..., None],
            noised_src.float()[..., None]
        ]  # 2

        seq = res_data['seq']
        src_seq = F.one_hot(seq[src], num_classes=self.num_aa)
        dst_seq = F.one_hot(seq[dst], num_classes=self.num_aa)
        edge_features.append(src_seq * ~noised_src[..., None])  # 21
        edge_features.append(dst_seq * ~noised_dst[..., None])  # 21

        bb = atom14[..., :3, :]
        virtual_Cb = _ideal_virtual_Cb(bb)
        bb = torch.cat([bb, virtual_Cb], dim=-2)

        ## bb edge distances
        edge_bb_src = bb[src]
        edge_bb_dst = bb[dst]
        edge_bb_dists = torch.linalg.vector_norm(
            edge_bb_src[..., None, :] - edge_bb_dst[..., None, :, :] + eps,
            dim=-1)
        edge_bb_dists = edge_bb_dists.view(edge_index.shape[1], -1, 1)
        edge_rbf = _rbf(edge_bb_dists, D_min=2.0, D_max=22.0, D_count=self.num_rbf, device=edge_index.device)
        edge_rbf = edge_rbf.view(num_edges, -1)
        edge_features.append(edge_rbf)  # 256

        ## edge rel pos embedding
        edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=self.num_pos_embed, device=edge_index.device)
        edge_features.append(edge_dist_rel_pos)  # 16

        ## pairwise sidechain atom distances
        src_atom14_local = rigids[..., None].invert_apply(atom14)[src]
        dst_atom14_in_src_local = rigids[src][..., None].invert_apply(atom14)
        edge_features.append(
            src_atom14_local.view(num_edges, -1) *  ~noised_src[..., None]
        )  # 42
        edge_features.append(
            dst_atom14_in_src_local.view(num_edges, -1) *  ~noised_dst[..., None]
        )  # 42

        # TODO: clip by some upper bound e.g. 10 A?
        atom14_pairwise_dist = torch.linalg.vector_norm(
            src_atom14_local[..., None, :] - dst_atom14_in_src_local[..., None, :, :] + eps,
            dim=-1
        )
        edge_features.append(
            atom14_pairwise_dist.view(num_edges, -1) * unnoised_edges[..., -1]
        )  # 196

        # total 596
        return torch.cat(edge_features, dim=-1)
