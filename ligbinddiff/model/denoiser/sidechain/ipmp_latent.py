import torch
from torch import nn
import torch.nn.functional as F
from torch_cluster import knn_graph

from ligbinddiff.model.modules.layers.node.mpnn import IPMP
from ligbinddiff.model.modules.layers.edge.embed import PairwiseAtomicEmbedding
from ligbinddiff.model.modules.common import GaussianRandomFourierBasis
from ligbinddiff.model.modules.openfold.frames import Linear
from ligbinddiff.data.datasets.featurize.sidechain import _dihedrals, _ideal_virtual_Cb
from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings, _rbf

from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.model.utils.graph import batchwise_to_nodewise


class IPMPUpdateLayer(nn.Module):
    def __init__(self,
                 c_s,
                 c_latent,
                 c_z,
                 c_hidden,
                 self_conditioning=True):
        super().__init__()
        self.c_s = c_s
        self.c_latent = c_latent
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.self_conditioning = self_conditioning

        h_dim = c_s + c_latent + c_latent*self_conditioning

        self.ipmp = IPMP(
            c_s=h_dim,
            c_z=c_z,
            c_hidden=c_hidden,
            dropout=0.,
            edge_dropout=0.,
        )

        self.latent_update = Linear(h_dim, c_latent, init='final')

        self.node_update = Linear(
            h_dim,
            c_s,
            init='final')
        self.node_ln = nn.LayerNorm(c_s)

    def forward(self,
                latent_features,
                node_features,
                edge_features,
                edge_index,
                rigids,
                node_mask,
                self_condition=None):

        input_features = [node_features, latent_features]
        if self.self_conditioning and self_condition is not None:
            input_features.append(self_condition['pred_latent_sidechain'])
        elif self.self_conditioning:
            input_features.append(torch.zeros_like(latent_features))

        input_features = torch.cat(input_features, dim=-1)

        # node_update = self.ipmp(
        joint_features, edge_features = self.ipmp(
            s=input_features,
            z=edge_features,
            edge_index=edge_index,
            r=rigids,
            mask=node_mask)
        # node_features = self.node_ln(node_features + node_update * node_mask[..., None])
        # node_features = self.node_transition(node_features) * node_mask[..., None]

        latent_features = latent_features + self.latent_update(joint_features)
        node_features = self.node_ln(node_features + self.node_update(joint_features))

        return latent_features, node_features, edge_features


class IPMPDenoiser(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_latent=128,
                 c_z=128,
                 c_hidden=256,
                 c_s_in=6,
                 c_time=64,
                 num_rbf=16,
                 num_pos_embed=16,
                 num_layers=4,
                 k=30,
                 num_aa=20,
                 self_conditioning=True):
        super().__init__()
        self.c_s = c_s
        self.c_latent = c_latent
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.num_rbf = num_rbf
        self.num_pos_embed = num_pos_embed
        self.c_z_in = num_rbf * (25 + 1)
        self.self_conditioning = self_conditioning
        self.num_aa = num_aa + 1  # + X

        self.time_rbf = GaussianRandomFourierBasis(n_basis=c_time//2)

        self.c_cond = (
            37 * 3  # rotamer
            + self.num_aa  # seq identity
            + 1  # mask
        )

        self.embed_node = nn.Sequential(
            nn.Linear(c_s_in + c_time + self.c_cond, 2*c_s),
            nn.ReLU(),
            nn.Linear(2*c_s, 2*c_s),
            nn.ReLU(),
            nn.Linear(2*c_s, c_s),
            nn.LayerNorm(c_s),
        )
        self.init_edge_embed = PairwiseAtomicEmbedding(
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
        self.update = nn.ModuleList([
            IPMPUpdateLayer(
                c_s=c_s,
                c_latent=c_latent,
                c_z=c_z,
                c_hidden=c_hidden,
                self_conditioning=self_conditioning
            )
            for _ in range(num_layers)
        ])

        self.k = k

    def _prep_features(self, graph, eps=1e-8):
        res_data = graph['residue']
        res_mask = res_data['res_mask']

        # time features
        ts = res_data['t']
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (B x h_time,)
        fourier_time = batchwise_to_nodewise(fourier_time, res_data.batch)

        # node features
        X_ca = res_data['x'].float()
        bb = res_data['bb'].float()
        dihedrals = _dihedrals(bb)

        mask = res_data['mlm_mask']

        seq = F.one_hot(res_data['seq'], num_classes=self.num_aa).to(mask.device)
        masked_seq = seq * mask[..., None]
        # print((masked_seq == 0).all())

        node_scalars = torch.cat(
            [
                dihedrals,
                mask.float()[..., None],
                masked_seq,
                fourier_time
            ],
        dim=-1)
        rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_0)
        node_vectors = rigids[..., None].invert_apply(res_data['atom37'])
        node_vectors[..., 4:, :] = node_vectors[..., 4:, :] * mask[..., None, None]
        # print((node_vectors[..., 4:, :] == 0).all())
        node_features = torch.cat(
            [node_scalars, node_vectors.view([node_scalars.shape[0], -1])],
            dim=-1
        ).float()
        node_features = node_features * res_mask[..., None]

        # edge graph
        masked_X_ca = X_ca.clone()
        masked_X_ca[~res_mask] = torch.inf
        edge_index = knn_graph(masked_X_ca, self.k, graph['residue'].batch)

        # edge features
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

        return node_features, rigids, edge_features, edge_index


    def forward(self, graph, intermediates, self_condition=None):
        latent_features = intermediates['noised_latent_sidechain']
        res_mask = graph['residue'].res_mask
        node_features, rigids, edge_features, edge_index = self._prep_features(graph)
        node_features = self.embed_node(node_features)
        edge_features = self.embed_edge(edge_features)

        for layer in self.update:
            latent_features, node_features, edge_features = layer(
                latent_features=latent_features,
                node_features=node_features,
                edge_features=edge_features,
                edge_index=edge_index,
                rigids=rigids,
                node_mask=res_mask.float(),
                self_condition=self_condition
            )

        intermediates['pred_latent_sidechain'] = latent_features
        return intermediates