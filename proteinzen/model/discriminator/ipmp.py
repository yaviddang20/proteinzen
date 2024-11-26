import torch
from torch import nn
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.utils import dropout_edge, scatter

from proteinzen.model.modules.layers.node.mpnn import IPMP
from proteinzen.model.modules.layers.edge.embed import PairwiseAtomicEmbedding
from proteinzen.data.datasets.featurize.sidechain import _dihedrals, _ideal_virtual_Cb
from proteinzen.data.datasets.featurize.common import _edge_positional_embeddings, _rbf
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.model.modules.openfold.layers import Linear


class IPMPUpdateLayer(nn.Module):
    def __init__(self, c_s, c_z, c_hidden, dropout=0.):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.p_dropout = dropout

        self.ipmp = IPMP(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
        )
        # self.ln = nn.LayerNorm(c_s)
        # self.dropout = nn.Dropout(p=dropout)
        # self.node_transition = StructureModuleTransition(
        #     c_s,
        #     dropout=dropout
        # )
        # self.edge_transition = EdgeTransition(
        #     node_embed_size=c_s,
        #     edge_embed_in=c_z,
        #     edge_embed_out=c_z,
        #     dropout=dropout
        # )

    def forward(self,
                node_features,
                edge_features,
                edge_index,
                rigids,
                node_mask):
        # dropout_edge_index, dropout_edge_mask = dropout_edge(edge_index, p=self.p_dropout, training=self.training)
        # dropout_edge_features = edge_features[dropout_edge_mask]
        # node_update = self.ipmp(
        #     s=node_features,
        #     z=dropout_edge_features,
        #     edge_index=dropout_edge_index,
        #     r=rigids)
        # node_features = self.ln(node_features + self.dropout(node_update) * node_mask[..., None])
        # # node_features = node_features + node_update * node_mask[..., None]
        # node_features = self.node_transition(node_features) * node_mask[..., None]
        # edge_features = self.edge_transition(node_features, edge_features, edge_index)
        node_features, edge_features = self.ipmp(
            s=node_features,
            z=edge_features,
            edge_index=edge_index,
            r=rigids,
            mask=node_mask)
        # edge_features = self.edge_transition(node_features, edge_features, edge_index)
        return node_features, edge_features



class IPMPDiscriminator(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_hidden=256,
                 c_s_in=6,
                 c_s_out=128,
                 num_rbf=16,
                 num_pos_embed=16,
                 num_layers=4,
                 k=30,
                 num_aa=21,
                 dropout=0.1,
                 classic_mode=True):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.num_rbf = num_rbf
        self.num_pos_embed = num_pos_embed
        self.classic_mode = classic_mode
        # atoms_per_res = 5
        # self.c_z_in = (
        #     num_rbf * (atoms_per_res ** 2)  # bb x bb distances
        #     # + atoms_per_res * 3  # src CA to dst bb direction unit vectors
        #     + num_pos_embed  # rel pos embed
        # )
        self.num_aa = num_aa
        self.c_atomic = (
            14 * 3  # rotamer
            + self.num_aa  # seq identity
        )

        self.embed_node = nn.Sequential(
            nn.Linear(c_s_in + self.c_atomic, 2*c_s),
            nn.ReLU(),
            nn.Linear(2*c_s, 2*c_s),
            nn.ReLU(),
            nn.Linear(2*c_s, c_s),
            nn.LayerNorm(c_s),
        )
        self.init_edge_embed = PairwiseAtomicEmbedding(
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
        self.update = nn.ModuleList([
            IPMPUpdateLayer(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.score = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, 1, bias=False)
        )

        self.k = k

    @torch.no_grad()
    def _prep_features(self, graph, eps=1e-8):
        res_data = graph['residue']
        res_mask = res_data['res_mask']

        atom14 = res_data['atom14_gt_positions']
        # node features
        X_ca = atom14[:, 1]
        bb = atom14[:, :4]
        dihedrals = _dihedrals(bb)


        # TODO: fix this inconsistency
        if "rigids_1" in res_data:
            rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_1)
        else:
            rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_0)

        seq = res_data['seq_one_hot']
        if seq.shape[-1] < self.num_aa:
            seq = F.pad(seq, (0, self.num_aa - seq.shape[-1]))
        node_scalars = torch.cat(
            [
                dihedrals,
                seq
            ],
        dim=-1)
        node_vectors = rigids[..., None].invert_apply(res_data['atom14_gt_positions'])
        node_features = torch.cat(
            [node_scalars, node_vectors.view([node_scalars.shape[0], -1])],
            dim=-1
        ).float()
        node_features = node_features * res_mask[..., None]

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

        return node_features, edge_features, edge_index


    def forward(self, graph, eps=1e-8):
        ## prep features
        res_data = graph['residue']
        res_mask = res_data['res_mask']
        # TODO: fix this inconsistency
        if "rigids_1" in res_data:
            rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_1)
        else:
            rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_0)

        node_features, edge_features, edge_index = self._prep_features(graph, eps=eps)

        node_features = self.embed_node(node_features)
        edge_features = self.embed_edge(edge_features)

        for layer in self.update:
            node_features, edge_features = layer(
                node_features,
                edge_features,
                edge_index,
                rigids,
                res_mask.float()
            )

        num_batch = graph.num_graphs
        score_features = scatter(
            node_features * res_mask[..., None],
            res_data.batch,
            dim_size=num_batch,
            reduce='sum'
        )
        norm = scatter(
            res_mask.float(),
            res_data.batch,
            dim_size=num_batch,
            reduce='sum'
        ).clip(min=1)
        score = self.score(score_features / norm[..., None])

        return score