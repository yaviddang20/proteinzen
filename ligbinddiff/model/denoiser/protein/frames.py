""" Denoising model """

import torch
from torch import nn
import torch.nn.functional as F
from ligbinddiff.model.modules.common import RBF
from ligbinddiff.model.modules.layers.edge.sitewise import EdgeTransition
from ligbinddiff.model.modules.layers.node.attention import GraphInvariantPointAttention
from ligbinddiff.model.modules.layers.lrange.vn import VirtualNodeAttnUpdate, VirtualNodeMPNNUpdate

from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.utils.framediff.all_atom import compute_backbone

from ligbinddiff.data.datasets.featurize.common import _node_positional_embeddings

from ligbinddiff.model.modules.openfold.frames import BackboneUpdate, StructureModuleTransition

from ligbinddiff.model.utils.graph import sample_inv_cubic_edges, sequence_local_graph, gen_spatial_graph_features, batchwise_to_nodewise

from ligbinddiff.model.denoiser.bb.framediff import TorsionAngles
from ligbinddiff.model.modules.openfold.frames import Linear


from ligbinddiff.model.modules.layers.lrange.anchor import ProjectivePoolUpdate, AnchorUpdate
from ligbinddiff.model.modules.layers.interres import OneParamPairwiseEquilibrate

from torch_geometric.utils import coalesce


class GraphIpaDenoisingLayer(nn.Module):
    """ Denoising layer on sidechain densities """
    def __init__(self,
                 c_s,
                 c_latent,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 use_anchors=True,
                 vn_mode='attn',
                 ):
        """
        Args
        ----
        """
        super().__init__()

        self.fuse_sidechain_node = nn.Linear(c_s + c_latent, c_s)

        self.attn_spatial = GraphInvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=num_heads,
            no_qk_points=num_qk_pts,
            no_v_points=num_v_pts,
        )
        self.ln_s1 = nn.LayerNorm(c_s)

        self.use_anchors = use_anchors
        if self.use_anchors:
            # self.pool_update = AnchorUpdate(c_s, c_z, ratio=0.1)
            self.pool_update = ProjectivePoolUpdate(c_s, ratio=0.05, connect_dim=3)

        self.bb_update = BackboneUpdate(
            c_s
        )
        self.sidechain_update = Linear(c_s, c_latent, init='final')

        self.node_transition = StructureModuleTransition(
            c=c_s
        )
        self.edge_transition = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z
        )
        if vn_mode == 'attn':
            self.vn_update = VirtualNodeAttnUpdate(
                c_s,
                c_s // num_heads,
                num_heads=num_heads
            )
        elif vn_mode == 'mpnn':
            self.vn_update = VirtualNodeMPNNUpdate(
                c_s,
                num_heads=num_heads
            )

    def forward(
            self,
            node_features,
            vn_features,
            rigids,
            sidechain,
            edge_features,
            edge_index,
            res_data
    ):
        res_mask = ~(res_data['res_mask'].bool())
        noising_mask = res_data['noising_mask']

        input_features = self.fuse_sidechain_node(
            torch.cat([node_features, sidechain], dim=-1)
        )

        node_s_update = self.attn_spatial(
            s=input_features,
            z=edge_features,
            edge_index=edge_index,
            r=rigids,
            mask=(~res_mask).float()
        )
        node_s_update = node_s_update * (~res_mask)[..., None]
        node_features = self.ln_s1(node_features + node_s_update)

        node_features, vn_features = self.vn_update(
            node_features,
            vn_features,
            res_data.batch,
            res_data['res_mask'].bool()
        )

        node_features = self.node_transition(node_features)
        node_features = node_features * (~res_mask)[..., None]
        rigids_update = self.bb_update(
            node_features * noising_mask[..., None])
        rigids = rigids.compose_q_update_vec(
            rigids_update * noising_mask[..., None]
        )

        sidechain_update = self.sidechain_update(
            node_features * noising_mask[..., None]
        )
        sidechain = sidechain + sidechain_update * noising_mask[..., None]

        edge_features = self.edge_transition(node_features, edge_features, edge_index)

        return node_features, vn_features, rigids, sidechain, edge_features


class GraphIpaDenoiser(nn.Module):
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
                 vn_mode='attn',
                 self_conditioning=False,
                 graph_conditioning=False,
                 use_anchors=False):
        super().__init__()

        self.c_s = c_s
        self.c_latent = c_latent
        self.c_z = c_z
        self.num_vn = num_vn
        self.self_conditioning = self_conditioning
        if graph_conditioning:
            assert self_conditioning, "graph conditioning requires self-conditioning"
        self.graph_conditioning = graph_conditioning
        self.n_layers = n_layers

        self.h_time = h_time
        self.time_rbf = RBF(n_basis=h_time//2)

        self.embed_node = nn.Sequential(
            nn.Linear(
                # node_embedding + time_embedding + fixed_mask + self_conditioning
                c_s + h_time + 1 + self_conditioning * (c_s + 7 + c_latent),
                c_s
            ),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.LayerNorm(c_s)
        )
        self.embed_edge = nn.Sequential(
            nn.Linear(c_z + self_conditioning * (c_z//2), c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.LayerNorm(c_z)
        )
        self.embed_vn = nn.Sequential(
            nn.Linear(
                h_time,
                c_s
            ),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s * num_vn),
            nn.LayerNorm(c_s * num_vn)
        )

        self.denoiser = nn.ModuleList([
            GraphIpaDenoisingLayer(
                 c_s,
                 c_latent,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 use_anchors=use_anchors,
                 vn_mode=vn_mode
            )
            for _ in range(n_layers)
        ])
        self.knn_k = knn_k
        self.lrange_k = lrange_k

        self.torsion_angles = TorsionAngles(c_s, 1)

    def forward(self, data, intermediates, self_condition=None):
        res_data = data['residue']
        ## prep features
        # ts = res_data['t']  # (B,)
        ts = data['t']  # (B,)
        res_mask = ~((res_data['res_mask']).bool())
        rigids_t = ru.Rigid.from_tensor_7(res_data['rigids_t'])
        batch = res_data.batch
        device = ts.device

        # center the training example at the mean of the x_cas
        center = ru.batchwise_center(rigids_t, res_data.batch, res_data['res_mask'].bool())
        rigids_t = rigids_t.translate(-center)

        sidechain = intermediates['noised_latent_sidechain']

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
                sequence_local_graph(subset_num_nodes, res_mask[select]) + offset
            )
            offset += select.sum().item()
        seq_local_edge_index = torch.cat(seq_local_edge_index, dim=-1)

        # generate spatial edges
        X_ca = rigids_t.get_trans()
        if self.graph_conditioning and self_condition is not None:
            self_cond_X_ca = self_condition['final_rigids'].get_trans()
            masked_X_ca = self_cond_X_ca.clone()
            masked_X_ca[res_mask] = torch.inf
            edge_index = sample_inv_cubic_edges(masked_X_ca, res_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)
        else:
            masked_X_ca = X_ca.clone()
            masked_X_ca[res_mask] = torch.inf
            edge_index = sample_inv_cubic_edges(masked_X_ca, res_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k)

        # remove duplicate edges
        edge_index = coalesce(
            torch.cat([edge_index, seq_local_edge_index], dim=-1)
        )

        # compute edge features
        edge_features, _ = gen_spatial_graph_features(X_ca, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)
        # seq_edge_features, _ = gen_spatial_graph_features(X_ca, seq_local_edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (N_graph x h_time,)
        vn_features = self.embed_vn(fourier_time).view(-1, self.num_vn, self.c_s)
        fourier_time = batchwise_to_nodewise(fourier_time, res_data.batch)  # (N_node x h_time,)
        # embedded_time = self.time_mlp(fourier_time)  # (N_node x h_time)

        # generate node features
        residx = torch.cat(residx, dim=-1)
        res_pos = _node_positional_embeddings(
            residx,
            num_embeddings=self.c_s,
            device=device)
        node_input = torch.cat([
                res_pos,
                fourier_time,
                res_data['noising_mask'].float()[..., None]
            ], dim=-1)
        if self.self_conditioning and self_condition is not None:
            self_cond_rigids = self_condition['final_rigids']
            self_cond_sidechain = self_condition['denoised_latent_sidechain']
            self_cond_nodes = self_condition['node_features']

            trans_rel = self_cond_rigids.get_trans() - rigids_t.get_trans()
            rigids_t_quat = rigids_t.get_rots().get_quats()
            self_cond_quat = self_cond_rigids.get_rots().get_quats()
            quat_rel = ru.quat_multiply(
                ru.invert_quat(rigids_t_quat),
                self_cond_quat
            )

            t7_rel = torch.cat([quat_rel, trans_rel], dim=-1)

            node_input = torch.cat(
                [node_input, self_cond_nodes, t7_rel, self_cond_sidechain],
                dim=-1
            )

            self_cond_X_ca = self_cond_rigids.get_trans()
            self_cond_edge_features, _ = gen_spatial_graph_features(self_cond_X_ca, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=0)
            edge_features = torch.cat(
                [edge_features, self_cond_edge_features],
                dim=-1
            )

        elif self.self_conditioning:
            node_input = F.pad(
                node_input,
                (0, self.c_s + 7 + self.c_latent)
            )
            edge_features = F.pad(
                edge_features,
                (0, self.c_z//2)
            )
        node_features = self.embed_node(node_input)
        node_features = node_features * (~res_mask)[..., None]
        edge_features = self.embed_edge(edge_features)


        ## denoising
        rigids_t = rigids_t.scale_translation(0.1)
        rigids = rigids_t

        for i, layer in enumerate(self.denoiser):
            node_features, vn_features, rigids, sidechain, edge_features = layer(
                node_features,
                vn_features,
                rigids,
                sidechain,
                edge_features,
                edge_index,
                res_data,
            )

        _, psi = self.torsion_angles(node_features)

        rigids = rigids.scale_translation(10)
        rigids = rigids.translate(center)
        ret = {}
        ret['denoised_frames'] = rigids
        ret['denoised_latent_sidechain'] = sidechain
        ret['pred_latent_sidechain'] = sidechain
        ret['final_rigids'] = rigids
        denoised_bb_items = compute_backbone(rigids.unsqueeze(0), psi.unsqueeze(0))
        denoised_bb = denoised_bb_items[-1].squeeze(0)[:, :5]
        ret['denoised_bb'] = denoised_bb
        ret['psi'] = psi
        ret['node_features'] = node_features
        ret['edge_index'] = edge_index

        return ret
