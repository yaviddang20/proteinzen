""" Denoising model """

import torch
from torch import nn
import torch.nn.functional as F
from ligbinddiff.model.modules.common import RBF
from ligbinddiff.model.modules.layers.edge.sitewise import EdgeTransition
from ligbinddiff.model.modules.layers.node.attention import GraphInvariantPointAttention
from ligbinddiff.model.modules.layers.node.mpnn import IPMP

from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.utils.framediff.all_atom import compute_backbone

from ligbinddiff.data.datasets.featurize.common import _node_positional_embeddings

from ligbinddiff.model.modules.openfold.frames import BackboneUpdate, StructureModuleTransition
from ligbinddiff.model.modules.layers.edge.embed import MLMPairwiseAtomicEmbedding, SelfConditionPairwiseAtomicEmbedding

from ligbinddiff.model.utils.graph import sample_inv_cubic_edges, sequence_local_graph, gen_spatial_graph_features, batchwise_to_nodewise

from ligbinddiff.model.denoiser.bb.framediff import TorsionAngles
from ligbinddiff.model.modules.openfold.frames import Linear


from torch_geometric.utils import coalesce


class BackboneDenoisingLayer(nn.Module):
    """ Denoising layer on sidechain densities """
    def __init__(self,
                 c_s,
                 c_latent,
                 edge_in,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 ):
        """
        Args
        ----
        """
        super().__init__()
        self.edge_in = edge_in
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_in, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.LayerNorm(c_z)
        )

        self.node_update = nn.Linear(c_s+c_latent, c_s)

        self.attn_spatial = GraphInvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=num_heads,
            no_qk_points=num_qk_pts,
            no_v_points=num_v_pts,
        )
        self.attn_seq = GraphInvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=num_heads,
            no_qk_points=num_qk_pts,
            no_v_points=num_v_pts,
        )
        self.ln_s1 = nn.LayerNorm(c_s)
        self.ln_s2 = nn.LayerNorm(c_s)

        self.node_transition = StructureModuleTransition(
            c=c_s
        )
        self.bb_update = BackboneUpdate(
            c_s
        )
        self.latent_update = Linear(c_s, c_latent, init='final')

    def forward(
            self,
            node_features,
            rigids,
            latent_features,
            edge_features,
            edge_index,
            seq_edge_features,
            seq_edge_index,
            res_data,
    ):
        res_mask = res_data['res_mask'].bool()
        noising_mask = res_data['noising_mask']

        edge_features = self.edge_embed(edge_features)
        node_features = self.node_update(
            torch.cat([node_features, latent_features], dim=-1)
        )

        node_s_update = self.attn_spatial(
            s=node_features,
            z=edge_features,
            edge_index=edge_index,
            r=rigids,
            mask=res_mask.float()
        )
        node_s_update = node_s_update * res_mask[..., None]
        node_features = self.ln_s1(node_features + node_s_update)

        node_s_update = self.attn_seq(
            s=node_features,
            z=seq_edge_features,
            edge_index=seq_edge_index,
            r=rigids,
            mask=res_mask.float()
        )
        node_s_update = node_s_update * res_mask[..., None]
        node_features = self.ln_s1(node_features + node_s_update)

        node_features = self.node_transition(node_features)
        node_features = node_features * res_mask[..., None]
        rigids_update = self.bb_update(
            node_features * noising_mask[..., None])

        rigids = rigids.compose_q_update_vec(
            rigids_update * noising_mask[..., None]
        )
        latent_features = latent_features + self.latent_update(node_features)

        return node_features, rigids, latent_features


class DynamicGraphIpaFrameDenoiser(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 c_s=256,
                 c_latent=128,
                 c_z=128,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_pts=8,
                 num_v_pts=12,
                 h_time=64,
                 n_layers=4,
                 knn_k=20,
                 lrange_k=40,
                 num_vn=4,
                 num_aa=20,
                 self_conditioning=False,
                 impute_oxy=False):
        super().__init__()

        self.c_s = c_s
        self.c_latent = c_latent
        self.c_z = c_z
        self.num_vn = num_vn
        self.self_conditioning = self_conditioning
        self.n_layers = n_layers
        self.impute_oxy = impute_oxy
        self.num_aa = num_aa+1
        self.knn_k = knn_k
        self.lrange_k = lrange_k
        self.h_time = h_time

        self.time_rbf = RBF(n_basis=h_time//2)

        # node_embedding + time_embedding + fixed_mask + noised_latent + self_conditioning
        self.node_in = (c_s + h_time + 1 + c_latent) + self_conditioning * (
            c_s             # last node embedding
            + c_latent      # last latent pred
            + 7             # rel transformation to last rigid pred
            + self.num_aa   # last seq pred
            + 14*3          # last aa pred
        )
        self.embed_node = nn.Sequential(
            nn.Linear(
                self.node_in,
                2*c_s
            ),
            nn.ReLU(),
            nn.Linear(2*c_s, 2*c_s),
            nn.ReLU(),
            nn.Linear(2*c_s, c_s),
            nn.LayerNorm(c_s)
        )
        self.seq_edge_embed = nn.Sequential(
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.LayerNorm(c_z)
        )

        self.gen_mlm_features = MLMPairwiseAtomicEmbedding(
            num_rbf=self.c_z//2,
            num_pos_embed=self.c_z//2,
        )
        self.gen_sc_edge_features = SelfConditionPairwiseAtomicEmbedding(
            num_rbf=self.c_z//2,
            num_pos_embed=self.c_z//2,
        )

        self.edge_in = self.gen_mlm_features.out_dim + self_conditioning * self.gen_sc_edge_features.out_dim

        self.denoiser = nn.ModuleList([
            BackboneDenoisingLayer(
                 c_s=c_s,
                 c_latent=c_latent,
                 edge_in=self.edge_in,
                 c_z=c_z,
                 c_hidden=c_hidden,
                 num_heads=num_heads,
                 num_qk_pts=num_qk_pts,
                 num_v_pts=num_v_pts,
            )
            for i in range(n_layers)
        ])
        self.torsion_angles = TorsionAngles(c_s, 1)

    def _gen_spatial_edge_features(self, data, rigids, res_mask, batch, self_condition):
        # generate spatial edges
        X_ca = rigids.get_trans()
        masked_X_ca = X_ca.clone()
        masked_X_ca[~res_mask] = torch.inf
        edge_index = sample_inv_cubic_edges(masked_X_ca, ~res_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)

        edge_features = []
        edge_features.append(
            self.gen_mlm_features(data, rigids, edge_index)
        )

        if self.self_conditioning and self_condition is not None:
            self_cond_edge_features = self.gen_sc_edge_features(self_condition, edge_index)
            edge_features.append(self_cond_edge_features)
        elif self.self_conditioning:
            edge_features.append(
                torch.zeros((
                    edge_index.shape[1],
                    self.gen_sc_edge_features.out_dim
                ), device=edge_index.device))

        return torch.cat(edge_features, dim=-1), edge_index

    def _gen_inital_features(self, data, intermediates, self_condition):
        res_data = data['residue']
        ## prep features
        # ts = res_data['t']  # (B,)
        ts = data['t']  # (B,)
        res_mask = res_data['res_mask'].bool()
        rigids_t = ru.Rigid.from_tensor_7(res_data['rigids_t'])
        device = ts.device

        # center the training example at the mean of the x_cas
        center = ru.batchwise_center(rigids_t, res_data.batch, res_mask)
        rigids_t = rigids_t.translate(-center)

        # # generate sequence edges
        seq_local_edge_index = []
        residx = []
        offset = 0
        for i in range(res_data.batch.max().item() + 1):
            select = (res_data.batch == i)
            subset_num_nodes = select.sum().item()
            local_residx = torch.arange(subset_num_nodes, device=device)
            residx.append(local_residx)
            seq_local_edge_index.append(
                sequence_local_graph(subset_num_nodes, ~res_mask[select]) + offset
            )
            offset += select.sum().item()
        seq_local_edge_index = torch.cat(seq_local_edge_index, dim=-1)

        # # generate spatial edges
        X_ca = rigids_t.get_trans()
        seq_edge_features, _ = gen_spatial_graph_features(X_ca, seq_local_edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (N_graph x h_time,)
        fourier_time = batchwise_to_nodewise(fourier_time, res_data.batch)  # (N_node x h_time,)

        # generate node features
        residx = torch.cat(residx, dim=-1)
        res_pos = _node_positional_embeddings(
            residx,
            num_embeddings=self.c_s,
            device=device)
        node_input = torch.cat([
                res_pos,
                fourier_time,
                res_data['noising_mask'].float()[..., None],
                intermediates['noised_latent_sidechain']
            ], dim=-1)

        if self.self_conditioning and self_condition is not None:
            self_cond_rigids = self_condition['final_rigids']
            self_cond_nodes = self_condition['node_features']
            self_cond_latent = self_condition['pred_latent_sidechain']

            # relative transformation to last prediction, in tensor7 form
            trans_rel = self_cond_rigids.get_trans() - rigids_t.get_trans()
            rigids_t_quat = rigids_t.get_rots().get_quats()
            self_cond_quat = self_cond_rigids.get_rots().get_quats()
            quat_rel = ru.quat_multiply(
                ru.invert_quat(rigids_t_quat),
                self_cond_quat
            )
            t7_rel = torch.cat([quat_rel, trans_rel], dim=-1)
            # last seq
            sc_seq = self_condition["decoded_seq_logits"].argmax(dim=-1)
            # last atom14
            atom14_local = self_cond_rigids[..., None].invert_apply(self_condition["decoded_atom14"])

            node_input = torch.cat(
                [
                    node_input,
                    self_cond_nodes,
                    self_cond_latent,
                    t7_rel,
                    F.one_hot(sc_seq, num_classes=self.num_aa),
                    atom14_local.view(res_data.num_nodes, -1)
                ],
                dim=-1
            )

        elif self.self_conditioning:
            node_input = F.pad(
                node_input,
                (0, self.c_s + self.c_latent + 7 + self.num_aa + 14*3)
            )


        return node_input, seq_edge_features, seq_local_edge_index


    def forward(self, data, intermediates, self_condition=None):
        res_data = data['residue']
        res_mask = (res_data['res_mask']).bool()
        rigids_t = ru.Rigid.from_tensor_7(res_data['rigids_t'])

        (
            node_input,
            seq_edge_features,
            seq_local_edge_index
        ) = self._gen_inital_features(data, intermediates, self_condition=self_condition)
        raw_edge_features, edge_index = self._gen_spatial_edge_features(data, rigids_t, res_mask, res_data.batch, self_condition)
        latent_features = intermediates['noised_latent_sidechain']

        # embed features
        node_features = self.embed_node(node_input)
        node_features = node_features * res_mask[..., None]
        seq_edge_features = self.seq_edge_embed(seq_edge_features)

        # center the training example at the mean of the x_cas
        center = ru.batchwise_center(rigids_t, res_data.batch, res_data['res_mask'].bool())
        rigids_t = rigids_t.translate(-center)

        ## denoising
        rigids = rigids_t.scale_translation(0.1)

        for i, layer in enumerate(self.denoiser):
            if i > 0:
                # recompute graph
                raw_edge_features, edge_index = self._gen_spatial_edge_features(
                    data,
                    rigids,
                    res_mask,
                    res_data.batch,
                    self_condition)
            node_features, rigids, latent_features = layer(
                node_features,
                rigids,
                latent_features,
                raw_edge_features,
                edge_index,
                seq_edge_features,
                seq_local_edge_index,
                res_data,
            )

        _, psi = self.torsion_angles(node_features)

        rigids = rigids.scale_translation(10)
        rigids = rigids.translate(center)
        ret = {}
        ret['denoised_frames'] = rigids
        ret['final_rigids'] = rigids
        denoised_bb_items = compute_backbone(rigids.unsqueeze(0), psi.unsqueeze(0))
        denoised_bb = denoised_bb_items[-1].squeeze(0)
        denoised_bb = denoised_bb[:, :5]
        ret['denoised_bb'] = denoised_bb
        ret['psi'] = psi
        ret['node_features'] = node_features
        ret['pred_latent_sidechain'] = latent_features
        ret['edge_index'] = edge_index

        return ret
