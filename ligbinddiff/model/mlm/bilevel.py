import torch
from torch import nn
import torch.nn.functional as F
from torch_cluster import knn_graph, radius_graph
import torch_geometric.utils as pygu


from ligbinddiff.model.modules.layers.node.mpnn import IPMP
from ligbinddiff.model.modules.layers.bilevel.attn import BilevelInvariantPointGraphAttention
from ligbinddiff.model.modules.openfold.frames import  StructureModuleTransition
from ligbinddiff.model.modules.layers.edge.sitewise import EdgeTransition
from ligbinddiff.data.datasets.featurize.sidechain import _dihedrals, _ideal_virtual_Cb, atom14_atom_type_embedding
from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings, _rbf

from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.utils.framediff.all_atom import compute_all_atom14
from ligbinddiff.utils.atom_reps import restype_1to3, atom91_start_end, letter_to_num
from ligbinddiff.utils.atom_reps import atom14_to_atom91, atom14_residue_bonds, atom14_atomic_row, atom14_atomic_period, restype_3to1, atom_to_atomic_period, atom_to_atomic_row
from ligbinddiff.data.openfold.residue_constants import restypes


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


class PairwiseEmbedding(nn.Module):
    def __init__(self,
                 num_rbf=16,
                 num_pos_embed=16,
                 dist_clip=10,
                 scale=1,
                 D_min=2.0,
                 D_max=22.0,
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
            + (4 * 4) * num_rbf     # bb x bb dist
            + num_pos_embed         # rel pos embed
            + (4 * 3) * 2           # src/dst bb coords
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

        ## pairwise sidechain atom distances
        src_bb_local = rigids[..., None].invert_apply(bb)[src]
        dst_bb_in_src_local = rigids[src, None].invert_apply(bb[dst])
        edge_features.append(
            src_bb_local.view(num_edges, -1) *  ~noised_src[..., None]
        )  # 42
        edge_features.append(
            dst_bb_in_src_local.view(num_edges, -1) *  ~noised_dst[..., None]
        )  # 42

        # total 596
        return torch.cat(edge_features, dim=-1).float()



class BilevelGraphAttnUpdateLayer(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_atom,
                 num_bilevel_points=8,
                 num_ipmp_points=8,
                 num_heads=4,
                 num_rbf_atom=16,
                 dropout=0.,
                 edge_dropout=0.):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.p_dropout = dropout

        self.bilevel_attn = BilevelInvariantPointGraphAttention(
            c_s=c_s,
            c_z=c_z,
            c_atom=c_atom,
            num_points=num_bilevel_points,
            num_heads=num_heads,
            num_rbf_atom=num_rbf_atom,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.ipmp = IPMP(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_s,
            no_points=num_ipmp_points,
            dropout=self.p_dropout,
            edge_dropout=edge_dropout
        )
        self.node_transition = StructureModuleTransition(
            c_s,
            dropout=dropout
        )
        self.atom_transition = StructureModuleTransition(
            c_atom,
            dropout=dropout
        )
        self.edge_transition = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z,
            dropout=dropout
        )

    def forward(self,
                node_features,
                rigids,
                edge_features,
                edge_index,
                # edge_subset_select,
                atom14_features,
                atom14_coords,
                atom14_mask,
                mgm_mask,
                res_mask):
        # dropout_edge_index, dropout_edge_mask = dropout_edge(edge_index, p=self.p_dropout, training=self.training)
        # dropout_edge_features = edge_features[dropout_edge_mask]
        node_features, atom14_features = self.bilevel_attn(
            node_features,
            rigids,
            edge_features,
            edge_index,
            # edge_subset_select,
            atom14_features,
            atom14_coords,
            atom14_mask,
            mgm_mask,
            res_mask)
        node_features = self.node_transition(node_features) * res_mask[..., None]
        edge_features = self.edge_transition(node_features, edge_features, edge_index)
        atom14_features = self.atom_transition(atom14_features) * atom14_mask[..., None]
        # node_features, edge_features = self.ipmp(
        #     s=node_features,
        #     z=edge_features,
        #     edge_index=edge_index,
        #     r=rigids,
        #     mask=res_mask
        # )

        return node_features, edge_features, atom14_features


class BilevelGraphAttnEncoder(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_atom=16,
                 c_s_in=6,
                 num_heads=4,
                 num_bilevel_points=8,
                 num_ipmp_points=8,
                 num_rbf=16,
                 num_pos_embed=16,
                 num_layers=4,
                 r_atomic=8,
                 k_residue=30,
                 num_aa=20,
                 dropout=0.1,
                 edge_dropout=0.2):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.num_rbf = num_rbf
        self.num_pos_embed = num_pos_embed
        # atoms_per_res = 5
        # self.c_z_in = (
        #     num_rbf * (atoms_per_res ** 2)  # bb x bb distances
        #     # + atoms_per_res * 3  # src CA to dst bb direction unit vectors
        #     + num_pos_embed  # rel pos embed
        # )
        self.num_aa = num_aa + 1
        self.c_atomic = (
            + self.num_aa  # seq identity
            + (4 * 3)  # bb atoms
            + 1  # mask position
        )

        self.embed_node = nn.Sequential(
            nn.Linear(c_s_in + self.c_atomic, 2*c_s),
            nn.ReLU(),
            nn.Linear(2*c_s, 2*c_s),
            nn.ReLU(),
            nn.Linear(2*c_s, c_s),
            nn.LayerNorm(c_s),
        )
        self.init_edge_embed = PairwiseEmbedding(
            num_rbf=self.num_rbf,
            num_pos_embed=self.num_pos_embed,
            num_aa=num_aa
        )
        self.embed_edge = nn.Sequential(
            nn.Linear(self.init_edge_embed.out_dim, 2*c_z),
            nn.ReLU(),
            nn.Linear(2*c_z, 2*c_z),
            nn.ReLU(),
            nn.Linear(2*c_z, c_z),
            nn.LayerNorm(c_z),
        )
        atom_in = max(atom_to_atomic_period.values()) + max(atom_to_atomic_row.values()) + 2
        self.embed_atom = nn.Sequential(
            nn.Linear(atom_in, 2*c_atom),
            nn.ReLU(),
            nn.Linear(2*c_atom, 2*c_atom),
            nn.ReLU(),
            nn.Linear(2*c_atom, c_atom),
            nn.LayerNorm(c_atom),
        )

        self.update = nn.ModuleList([
            BilevelGraphAttnUpdateLayer(
                c_s=c_s,
                c_z=c_z,
                c_atom=c_atom,
                num_rbf_atom=num_rbf,
                num_heads=num_heads,
                num_bilevel_points=num_bilevel_points,
                num_ipmp_points=num_ipmp_points,
                dropout=dropout,
                edge_dropout=edge_dropout,
            )
            for _ in range(num_layers)
        ])
        self.output_mu = nn.Linear(c_s, c_s)
        self.output_logvar = nn.Linear(c_s, c_s)

        self.r_atomic = r_atomic
        self.k_residue = k_residue

    @torch.no_grad()
    def _prep_features(self, graph, eps=1e-8):
        res_data = graph['residue']
        res_mask = res_data['res_mask']

        # node features
        X_ca = res_data['x'].float()
        bb = res_data['bb'].float()
        dihedrals = _dihedrals(bb)
        X_cb = _ideal_virtual_Cb(bb)
        # bb = torch.cat([bb, X_cb[..., None, :]], dim=-2)

        mgm_mask = res_data['mlm_mask']
        noising_mask = res_data['noising_mask']

        seq = F.one_hot(res_data['seq'], num_classes=self.num_aa).to(mgm_mask.device)
        masked_seq = seq * mgm_mask[..., None]

        node_scalars = torch.cat(
            [
                dihedrals,
                mgm_mask.float()[..., None],
                masked_seq
            ],
        dim=-1)
        # TODO: fix this inconsistency
        if "rigids_1" in res_data:
            rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_1)
        else:
            rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_0)
        node_vectors = rigids[..., None].invert_apply(res_data['atom37'])
        # node_vectors = torch.cat([
        #     node_vectors[..., :3, :], X_cb[..., None, :]],
        # dim=-2)
        # TODO, WARNING: this is data leakage because 4 is real Cb
        # i'm using this to test a theory
        node_vectors = node_vectors[..., :4, :]
        # node_vectors = rigids[..., None].invert_apply(bb)
        # node_vectors = node_vectors * mask[..., None, None]
        node_features = torch.cat(
            [node_scalars, node_vectors.view([node_scalars.shape[0], -1])],
            dim=-1
        ).float()
        node_features = node_features * res_mask[..., None]

        # edge graph
        masked_X_ca = X_ca.clone()
        masked_X_ca[~res_mask] = torch.inf
        # masked_X_cb = X_cb.clone()
        # masked_X_cb[~res_mask] = torch.inf
        # there's a better way of doing this but im lazy
        edge_index = knn_graph(masked_X_ca, self.k_residue, graph['residue'].batch)
        edge_subset_select = torch.ones(edge_index.shape[1], device=edge_index.device).bool()
        # k_atomic_edge_index = radius_graph(
        #     masked_X_ca,
        #     self.r_atomic,
        #     graph['residue'].batch,
        #     max_num_neighbors=self.k_residue)
        # k_residue_edge_index = knn_graph(masked_X_ca, self.k_residue, graph['residue'].batch)
        # k_atomic_mask = torch.ones(k_atomic_edge_index.shape[-1], device=k_atomic_edge_index.device)
        # k_residue_mask = torch.zeros(k_residue_edge_index.shape[-1], device=k_residue_edge_index.device)

        # edge_index, edge_subset_select = pygu.coalesce(
        #     torch.cat([k_atomic_edge_index, k_residue_edge_index], dim=-1),
        #     torch.cat([k_atomic_mask, k_residue_mask], dim=-1),
        #     sort_by_row=False
        # )
        # edge_subset_select = edge_subset_select.bool()

        # # edge features
        # virtual_Cb = _ideal_virtual_Cb(bb)
        # bb = torch.cat([bb, virtual_Cb[..., None, :]], dim=-2)
        src = edge_index[1]
        dst = edge_index[0]

        # ## edge distances
        edge_features = self.init_edge_embed(graph, edge_index)
        # technically this shouldn't be necessary but just to be safe
        edge_mask = res_mask[src] & res_mask[dst]
        edge_features = edge_features * edge_mask[..., None]

        atom14_coords = res_data['atom14_gt_positions'].float()
        atom14_mask = res_data['atom14_gt_exists'].bool()
        atom14_coords[..., 4:, :] *= mgm_mask[..., None, None]
        atom14_mask[..., 4:] = mgm_mask[..., None]

        atom14_features = atom14_atom_type_embedding(
            res_data['seq'],
            letter_to_num).float()
        atom14_features[..., 4:, :] *= mgm_mask[..., None, None]

        # edge_features = torch.cat([edge_rbf, edge_dist_rel_pos, local_edge_dir_feats], dim=-1)
        # edge_features = torch.cat([edge_rbf, edge_dist_rel_pos], dim=-1)

        return node_features, edge_features, edge_index, edge_subset_select, atom14_features, atom14_coords, atom14_mask


    def forward(self, graph, eps=1e-8):
        ## prep features
        res_data = graph['residue']
        res_mask = res_data['res_mask']
        # TODO: fix this inconsistency
        if "rigids_1" in res_data:
            rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_1)
        else:
            rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_0)
        noising_mask = res_data['noising_mask']

        (
            node_features,
            edge_features,
            edge_index,
            edge_subset_select,
            atom14_features,
            atom14_coords,
            atom14_mask
        ) = self._prep_features(graph, eps=eps)

        node_features = self.embed_node(node_features)
        edge_features = self.embed_edge(edge_features)
        atom14_features = self.embed_atom(atom14_features)

        for layer in self.update:
            node_features, edge_features, atom14_features  = layer(
                node_features,
                rigids,
                edge_features,
                edge_index,
                # edge_subset_select,
                atom14_features,
                atom14_coords,
                atom14_mask,
                noising_mask,
                res_mask)

        latent_mu = self.output_mu(node_features)
        latent_logvar = self.output_logvar(node_features)

        out_dict = {}
        out_dict['latent_mu'] = latent_mu
        out_dict['latent_logvar'] = latent_logvar

        return out_dict

