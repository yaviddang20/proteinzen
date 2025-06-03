import torch
from torch import nn
import torch.nn.functional as F

from proteinzen.data.datasets.featurize.common import _rbf, _edge_positional_embeddings
from proteinzen.data.datasets.featurize.sidechain import _ideal_virtual_Cb
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.utils.framediff.all_atom import compute_backbone


class PairwiseAtomicEmbedding(nn.Module):
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
        self.classic_mode = classic_mode

        self.scale = scale
        self.D_min = D_min
        self.D_max = D_max

        if classic_mode:
            self.out_dim = (
                + (5 * 5) * num_rbf     # bb x bb dist
                + num_pos_embed         # rel pos embed
            )
        else:
            self.out_dim = (
                2                       # mask bits
                + self.num_aa * 2       # src/dst seq
                + (5 * 5) * num_rbf     # bb x bb dist
                + num_pos_embed         # rel pos embed
                + (14 * 3) * 2          # src/dst atom14
                + 14 * 14               # src/dst atom14 pairwise dist
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

        atom14 = res_data["atom14_gt_positions"]
        # TODO: fix this inconsistency
        if "rigids_1" in res_data:
            rigids = ru.Rigid.from_tensor_7(data['residue'].rigids_1)
        else:
            rigids = ru.Rigid.from_tensor_7(data['residue'].rigids_0)

        if self.classic_mode:
            edge_features = []
        else:
            edge_features = [
                noised_dst.float()[..., None],
                noised_src.float()[..., None]
            ]  # 2

            seq = res_data['seq']
            src_seq = F.one_hot(seq[src], num_classes=self.num_aa)
            dst_seq = F.one_hot(seq[dst], num_classes=self.num_aa)
            edge_features.append(src_seq * ~noised_src[..., None])  # 21
            edge_features.append(dst_seq * ~noised_dst[..., None])  # 21

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
        edge_features.append(edge_rbf)  # 256

        ## edge rel pos embedding
        edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=self.num_pos_embed, device=edge_index.device)
        edge_features.append(edge_dist_rel_pos)  # 16

        if not self.classic_mode:
            ## pairwise sidechain atom distances
            src_atom14_local = rigids[..., None].invert_apply(atom14)[src]
            dst_atom14_in_src_local = rigids[src, None].invert_apply(atom14[dst])

            src_atom14_local[..., 4:, :] = src_atom14_local[..., 4:, :] * ~noised_src[..., None, None]
            dst_atom14_in_src_local[..., 4:, :] = dst_atom14_in_src_local[..., 4:, :] * ~noised_dst[..., None, None]

            edge_features.append(
                src_atom14_local.view(num_edges, -1)
            )  # 42
            edge_features.append(
                dst_atom14_in_src_local.view(num_edges, -1)
            )  # 42

            # TODO: clip by some upper bound e.g. 10 A?
            atom14_pairwise_dist = torch.linalg.vector_norm(
                src_atom14_local[..., None, :] - dst_atom14_in_src_local[..., None, :, :] + eps,
                dim=-1
            )
            edge_features.append(
                atom14_pairwise_dist.view(num_edges, -1) * unnoised_edges[..., None]
            )  # 196

        # total 596
        return torch.cat(edge_features, dim=-1).float()


class MLMPairwiseAtomicEmbedding(nn.Module):
    def __init__(self,
                 num_rbf=64,
                 num_pos_embed=64,
                 dist_clip=10,
                 scale=0.1,
                 D_min=0.0,
                 D_max=20.0,
                 num_aa=20):
        super().__init__()
        self.dist_clip = dist_clip
        self.num_aa = num_aa + 1
        self.num_rbf = num_rbf
        self.num_pos_embed = num_pos_embed

        self.scale = scale
        self.D_min = D_min
        self.D_max = D_max

        # TODO: compute this based on input params
        self.out_dim = (
            2                       # mask bits
            + self.num_aa * 2       # src/dst seq
            + num_rbf               # CA x CA dist
            + num_pos_embed         # rel pos embed
            + (14 * 3) * 2          # src/dst atom14
            + 14 * 14               # src/dst atom14 pairwise dist
        )

    def forward(self, data, rigids, edge_index, eps=1e-8):
        res_data = data['residue']
        num_edges = edge_index.shape[1]
        noising_mask = res_data['noising_mask']

        dst = edge_index[0]
        src = edge_index[1]
        noised_src = noising_mask[src]
        noised_dst = noising_mask[dst]
        unnoised_edges = ~(noised_src | noised_dst)

        atom14 = res_data["atom14_gt_positions"] * self.scale

        edge_features = [
            noised_dst.float()[..., None],
            noised_src.float()[..., None]
        ]  # 2

        seq = res_data['seq']
        src_seq = F.one_hot(seq[src], num_classes=self.num_aa)
        dst_seq = F.one_hot(seq[dst], num_classes=self.num_aa)
        edge_features.append(src_seq * ~noised_src[..., None])  # 21
        edge_features.append(dst_seq * ~noised_dst[..., None])  # 21

        X_ca = atom14[..., 1, :]
        bb = compute_backbone(
            rigids,
            psi_torsions=torch.tensor([[0., 1.]], device=rigids.device).expand(res_data.num_nodes, -1)
        )[-1]
        # mask O, since we don't necessarily know where it is
        bb[..., 4, :] = 0.
        atom14 = atom14 * (~noising_mask[..., None, None]) + bb * noising_mask[..., None, None]

        ## bb edge distances
        edge_dists = torch.linalg.vector_norm(
            X_ca[dst] - X_ca[src], dim=-1
        )
        edge_rbf = _rbf(edge_dists, D_min=self.D_min, D_max=self.D_max, D_count=self.num_rbf, device=edge_index.device)
        edge_rbf = edge_rbf.view(num_edges, -1)
        edge_features.append(edge_rbf)  # 64

        ## edge rel pos embedding
        edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=self.num_pos_embed, device=edge_index.device)
        edge_features.append(edge_dist_rel_pos)  # 64

        ## pairwise sidechain atom distances
        src_atom14_local = rigids[..., None].invert_apply(atom14)[src]
        dst_atom14_in_src_local = rigids[src, None].invert_apply(atom14[dst])
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
        return torch.cat(edge_features, dim=-1).float()



class SelfConditionPairwiseAtomicEmbedding(nn.Module):
    def __init__(self,
                 num_rbf=64,
                 num_pos_embed=64,
                 dist_clip=10,
                 scale=0.1,
                 D_min=0.0,
                 D_max=20.0,
                 num_aa=20):
        super().__init__()
        self.dist_clip = dist_clip
        self.num_aa = num_aa + 1
        self.num_rbf = num_rbf
        self.num_pos_embed = num_pos_embed

        self.scale = scale
        self.D_min = D_min
        self.D_max = D_max

        self.out_dim = (
            + 1                     # self condition bit
            + self.num_aa * 2       # src/dst seq
            + num_rbf               # CA x CA dist
            # + num_pos_embed         # rel pos embed
            + (14 * 3) * 2          # src/dst atom14
            + 14 * 14               # src/dst atom14 pairwise dist
        )


    def forward(self, self_condition, edge_index, eps=1e-8):
        num_edges = edge_index.shape[1]
        dst = edge_index[0]
        src = edge_index[1]

        atom14 = self_condition["decoded_atom14"] * self.scale
        rigids = self_condition['final_rigids'].scale_translation(self.scale)

        edge_features = [
            torch.ones((num_edges, 1), device=atom14.device)
        ]

        seq = self_condition['decoded_seq_logits'].argmax(dim=-1)
        src_seq = F.one_hot(seq[src], num_classes=self.num_aa)
        dst_seq = F.one_hot(seq[dst], num_classes=self.num_aa)
        edge_features.append(src_seq)  # 21
        edge_features.append(dst_seq)  # 21

        X_ca = atom14[..., 1, :]

        ## bb edge distances
        edge_dists = torch.linalg.vector_norm(
            X_ca[dst] - X_ca[src], dim=-1
        )
        edge_rbf = _rbf(edge_dists, D_min=self.D_min, D_max=self.D_max, D_count=self.num_rbf, device=edge_index.device)
        edge_rbf = edge_rbf.view(num_edges, -1)
        edge_features.append(edge_rbf)  # 64

        # ## edge rel pos embedding
        # edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=self.num_pos_embed, device=edge_index.device)
        # edge_features.append(edge_dist_rel_pos)  # 64

        ## pairwise sidechain atom distances
        src_atom14_local = rigids[..., None].invert_apply(atom14)[src]
        dst_atom14_in_src_local = rigids[src, None].invert_apply(atom14[dst])
        edge_features.append(
            src_atom14_local.view(num_edges, -1)
        )  # 42
        edge_features.append(
            dst_atom14_in_src_local.view(num_edges, -1)
        )  # 42

        # TODO: clip by some upper bound e.g. 10 A?
        atom14_pairwise_dist = torch.linalg.vector_norm(
            src_atom14_local[..., None, :] - dst_atom14_in_src_local[..., None, :, :] + eps,
            dim=-1
        )
        edge_features.append(
            atom14_pairwise_dist.view(num_edges, -1)
        )  # 196

        # total 594
        return torch.cat(edge_features, dim=-1).float()
