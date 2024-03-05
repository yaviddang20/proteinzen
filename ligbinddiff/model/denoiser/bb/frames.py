""" Denoising model """

import torch
from torch import nn
import torch.nn.functional as F
from ligbinddiff.model.modules.common import GaussianRandomFourierBasis
from ligbinddiff.model.modules.layers.edge.sitewise import EdgeTransition
from ligbinddiff.model.modules.layers.node.attention import GraphInvariantPointAttention
from ligbinddiff.model.modules.layers.lrange.vn import VirtualNodeAttnUpdate, VirtualNodeMPNNUpdate
from ligbinddiff.model.modules.layers.triangle.attention import SparseTriangleAttention
from ligbinddiff.model.modules.layers.triangle.mult import FusedSparseTriangleMultiplicativeTransition
from ligbinddiff.model.modules.layers.node.mpnn import IPMP

from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.utils.framediff.all_atom import compute_backbone, adjust_oxygen_pos

from ligbinddiff.data.datasets.featurize.common import _node_positional_embeddings
from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings, _rbf

from ligbinddiff.model.modules.openfold.frames import BackboneUpdate, StructureModuleTransition, Linear

from ligbinddiff.model.utils.graph import sample_inv_cubic_edges, sequence_local_graph, gen_spatial_graph_features, batchwise_to_nodewise

from .framediff import TorsionAngles


from ligbinddiff.model.modules.layers.lrange.anchor import ProjectivePoolUpdate, AnchorUpdate
from ligbinddiff.model.modules.layers.interres import OneParamPairwiseEquilibrate

from torch_geometric.utils import coalesce
from torch_geometric.nn import knn_graph


from torch.profiler import profile, record_function, ProfilerActivity


class GraphIpaFrameDenoisingLayer(nn.Module):
    """ Denoising layer on sidechain densities """
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 vn_mode='attn',
                 update_mode=None
                 ):
        """
        Args
        ----
        """
        super().__init__()
        assert update_mode in ["1joint", "2joint", "sep"], f"unsupported update mode {update_mode}"
        self.update_mode = update_mode

        self.attn_spatial = GraphInvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=num_heads,
            no_qk_points=num_qk_pts,
            no_v_points=num_v_pts,
        )
        if self.update_mode in ["2joint", "sep"]:
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

        self.bb_update = BackboneUpdate(
            c_s
        )
        self.node_transition = StructureModuleTransition(
            c=c_s
        )
        self.edge_transition = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z
        )
        if self.update_mode == "sep":
            self.seq_edge_transition = EdgeTransition(
                node_embed_size=c_s,
                edge_embed_in=c_z,
                edge_embed_out=c_z
            )

        assert vn_mode in ["attn", "mpnn", None], f"unsupported vn_mode {vn_mode}"
        self.vn_mode = vn_mode
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
            edge_features,
            edge_index,
            seq_edge_features,
            seq_edge_index,
            res_data,
            edge_update=True
    ):
        res_mask = ~(res_data['res_mask'].bool())
        noising_mask = res_data['noising_mask']

        if self.update_mode == '1joint':
            node_s_update = self.attn_spatial(
                s=node_features,
                z=edge_features,
                edge_index=edge_index,
                r=rigids,
                mask=(~res_mask).float()
            )
            node_s_update = node_s_update * (~res_mask)[..., None]
            node_features = self.ln_s1(node_features + node_s_update)

        elif self.update_mode == '2joint':
            node_s_update = self.attn_spatial(
                s=node_features,
                z=edge_features,
                edge_index=edge_index,
                r=rigids,
                mask=(~res_mask).float()
            )
            node_s_update = node_s_update * (~res_mask)[..., None]
            node_features = self.ln_s1(node_features + node_s_update)

            node_s_update = self.attn_seq(
                s=node_features,
                z=edge_features,
                edge_index=edge_index,
                r=rigids,
                mask=(~res_mask).float()
            )
            node_s_update = node_s_update * (~res_mask)[..., None]
            node_features = self.ln_s2(node_features + node_s_update)

        elif self.update_mode == 'sep':
            node_s_update = self.attn_spatial(
                s=node_features,
                z=edge_features,
                edge_index=edge_index,
                r=rigids,
                mask=(~res_mask).float()
            )
            node_s_update = node_s_update * (~res_mask)[..., None]
            node_features = self.ln_s1(node_features + node_s_update)

            node_s_update = self.attn_seq(
                s=node_features,
                z=seq_edge_features,
                edge_index=seq_edge_index,
                r=rigids,
                mask=(~res_mask).float()
            )
            node_s_update = node_s_update * (~res_mask)[..., None]
            node_features = self.ln_s2(node_features + node_s_update)
        else:
            raise ValueError(f"update_mode {self.update_mode} is not supported")

        if self.vn_mode is not None:
            node_features, vn_features = self.vn_update(
                node_features,
                vn_features,
                res_data.batch,
                res_data['res_mask'].bool()
            )

        # if self.use_anchors:
        #     # node_features, anchor_kl, node_kl = self.pool_update(rigids.get_trans(), node_features, edge_index, data.batch, (~x_mask).float())
        #     node_features, anchor_kl, node_kl = self.pool_update(rigids.get_trans(), node_features, edge_index, res_data.batch)
        # else:
        #     anchor_kl = torch.zeros(res_data.num_graphs, device=node_features.device)
        #     node_kl = torch.zeros(res_data.num_graphs, device=node_features.device)

        node_features = self.node_transition(node_features)
        node_features = node_features * (~res_mask)[..., None]
        rigids_update = self.bb_update(
            node_features * noising_mask[..., None])

        rigids = rigids.compose_q_update_vec(
            rigids_update * noising_mask[..., None]
        )
        if edge_update:
            edge_features = self.edge_transition(node_features, edge_features, edge_index)
            if self.update_mode == 'sep':
                seq_edge_features = self.seq_edge_transition(node_features, seq_edge_features, seq_edge_index)

        # return node_features, rigids, edge_features, seq_edge_features, 0, 0, #anchor_kl, node_kl
        # return node_features, vn_features, rigids, edge_features, 0, 0, #anchor_kl, node_kl
        return node_features, vn_features, rigids, edge_features, seq_edge_features #, 0, 0, anchor_kl, node_kl


class GraphIpaFrameDenoiser(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_hidden=256,
                 num_heads=8,
                 num_qk_pts=8,
                 num_v_pts=12,
                 h_time=64,
                 n_layers=4,
                 knn_k=20,
                 lrange_k=40,
                 use_self_edge=False,
                 num_vn=4,
                 vn_mode='attn',
                 self_conditioning=False,
                 graph_conditioning=False,
                 update_mode="sep",
                 interres_equilibrate=False):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.vn_mode = vn_mode
        self.num_vn = num_vn
        self.self_conditioning = self_conditioning
        assert update_mode in ["1joint", "2joint", "sep"]
        self.update_mode = update_mode
        self.use_self_edge = use_self_edge
        if graph_conditioning:
            assert self_conditioning, "graph conditioning requires self-conditioning"
        self.graph_conditioning = graph_conditioning
        self.n_layers = n_layers

        self.h_time = h_time
        self.time_rbf = GaussianRandomFourierBasis(n_basis=h_time//2)

        self.embed_node = nn.Sequential(
            nn.Linear(
                # node_embedding + time_embedding + fixed_mask + self_conditioning
                c_s + h_time + 1 + self_conditioning * (c_s + 7),
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
            GraphIpaFrameDenoisingLayer(
                 c_s,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 vn_mode=vn_mode,
                 update_mode=update_mode
            )
            for _ in range(n_layers)
        ])
        self.knn_k = knn_k
        self.lrange_k = lrange_k

        self.interres_equilibrate = interres_equilibrate
        if interres_equilibrate:
            self.equilibrate = OneParamPairwiseEquilibrate(
                 c_s,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
            )

        self.torsion_angles = TorsionAngles(c_s, 1)

    def forward(self, data, self_condition=None):
        res_data = data['residue']
        ## prep features
        # ts = res_data['t']  # (B,)
        ts = data['t']  # (B,)
        res_mask = ~((res_data['res_mask']).bool())
        rigids_t = ru.Rigid.from_tensor_7(res_data['rigids_t'])
        batch = res_data.batch
        device = ts.device
        num_nodes = res_data.num_nodes

        # center the training example at the mean of the x_cas
        center = ru.batchwise_center(rigids_t, res_data.batch, res_data['res_mask'].bool())
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
            edge_index = sample_inv_cubic_edges(
                masked_X_ca,
                res_mask,
                batch,
                knn_k=self.knn_k,
                inv_cube_k=self.lrange_k,
                self_edge=self.use_self_edge)
        else:
            masked_X_ca = X_ca.clone()
            masked_X_ca[res_mask] = torch.inf
            edge_index = sample_inv_cubic_edges(
                masked_X_ca,
                res_mask,
                batch,
                knn_k=self.knn_k,
                inv_cube_k=self.lrange_k,
                self_edge=self.use_self_edge)

        # remove duplicate edges
        if self.update_mode in ['1joint', '2joint']:
            edge_index = coalesce(
                torch.cat([edge_index, seq_local_edge_index], dim=-1)
            )

        # compute edge features
        edge_features, _ = gen_spatial_graph_features(X_ca, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)
        if self.update_mode == 'sep':
            seq_edge_features, _ = gen_spatial_graph_features(X_ca, seq_local_edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)
        else:
            seq_edge_features = 0

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (N_graph x h_time,)
        if self.vn_mode is not None:
            vn_features = self.embed_vn(fourier_time).view(-1, self.num_vn, self.c_s)
        else:
            vn_features = 0
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
                [node_input, self_cond_nodes, t7_rel],
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
                (0, self.c_s + 7)
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

        anchor_kl = []
        node_kl = []
        for i, layer in enumerate(self.denoiser):
            node_features, vn_features, rigids, edge_features, seq_edge_features = layer(
                node_features,
                vn_features,
                rigids,
                edge_features,
                edge_index,
                seq_edge_features,
                seq_local_edge_index,
                res_data,
            )
            # node_features, vn_features, rigids, edge_features, a_kl, n_kl = layer(
            #     node_features,
            #     vn_features,
            #     rigids,
            #     edge_features,
            #     edge_index,
            #     res_data,
            # )
            # anchor_kl.append(a_kl)
            # node_kl.append(n_kl)

        if self.interres_equilibrate:
            node_features, rigids = self.equilibrate(
                node_features,
                edge_features,
                edge_index,
                rigids,
                (~res_mask).float()
            )

        _, psi = self.torsion_angles(node_features)

        rigids = rigids.scale_translation(10)
        rigids = rigids.translate(center)
        ret = {}
        ret['denoised_frames'] = rigids
        ret['final_rigids'] = rigids
        denoised_bb_items = compute_backbone(rigids.unsqueeze(0), psi.unsqueeze(0))
        denoised_bb = denoised_bb_items[-1].squeeze(0)[:, :5]
        ret['denoised_bb'] = denoised_bb
        ret['psi'] = psi
        ret['node_features'] = node_features
        ret['anchor_kl'] = anchor_kl
        ret['node_kl'] = node_kl
        ret['edge_index'] = edge_index

        return ret


class GraphIpaFrameDenoisingLayer2(nn.Module):
    """ Denoising layer on sidechain densities """
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 self_conditioning=False,
                 triangle_attention=False,
                 triangle_mult=False,
                 embed_seq_edge=False,
                 use_seq_edge=True,
                 use_ipmp_seq_edge=False,
                 last=False,
                 num_rbf=64,
                 knn_k=20,
                 lrange_k=40
                 ):
        """
        Args
        ----
        """
        super().__init__()

        if triangle_attention:
            self.triangle_attention = SparseTriangleAttention(
                c_s=c_s,
                c_z=c_z,
                num_heads=num_heads,
                num_rbf=num_rbf,
            )
            self.edge_ln = nn.LayerNorm(c_z)
        else:
            self.triangle_attention = None

        if triangle_mult:
            self.triangle_mult = FusedSparseTriangleMultiplicativeTransition(
                c_s=c_s,
                c_z=c_z,
                num_rbf=num_rbf,
                use_ffn=True
            )
        else:
            self.triangle_mult = None


        self.attn_spatial = GraphInvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=num_heads,
            no_qk_points=num_qk_pts,
            no_v_points=num_v_pts,
        )
        self.ln_s1 = nn.LayerNorm(c_s)

        self.use_seq_edge = use_seq_edge
        self.use_ipmp_seq_edge = use_ipmp_seq_edge
        if self.use_seq_edge:
            if use_ipmp_seq_edge:
                self.attn_seq = IPMP(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_hidden,
                    dropout=0.,
                    edge_dropout=0.,
                    final_init='final',
                    update_edge=last
                )
            else:
                self.attn_seq = GraphInvariantPointAttention(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_hidden,
                    no_heads=num_heads,
                    no_qk_points=num_qk_pts,
                    no_v_points=num_v_pts,
                )

            self.ln_s2 = nn.LayerNorm(c_s)

        self.bb_update = BackboneUpdate(
            c_s
        )
        self.node_transition = StructureModuleTransition(
            c=c_s
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(c_z + self_conditioning * (c_z//2), c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z),
            nn.LayerNorm(c_z)
        )
        self.embed_seq_edge = embed_seq_edge
        if embed_seq_edge:
            self.seq_edge_embed = nn.Sequential(
                nn.Linear(c_z, c_z),
                nn.ReLU(),
                nn.Linear(c_z, c_z),
                nn.ReLU(),
                nn.Linear(c_z, c_z),
                nn.LayerNorm(c_z)
            )

        if not use_ipmp_seq_edge and not last:
            self.seq_edge_transition = EdgeTransition(
                node_embed_size=c_s,
                edge_embed_in=c_z,
                edge_embed_out=c_z
            )

        # self.vn_update = VirtualNodeAttnUpdate(
        #     c_s,
        #     c_s // num_heads,
        #     num_heads=num_heads
        # )
        # self.vn_update_proj = Linear(c_s, c_s, init='final')

    def forward(
            self,
            node_features,
            # vn_features,
            rigids,
            edge_features,
            edge_index,
            seq_edge_features,
            seq_edge_index,
            res_data,
    ):
        edge_features = self.edge_embed(edge_features)
        if self.embed_seq_edge:
            seq_edge_features = self.seq_edge_embed(seq_edge_features)

        if self.triangle_attention is not None:
            edge_update = torch.utils.checkpoint.checkpoint(
                self.triangle_attention,
                node_features,
                rigids,
                edge_features,
                edge_index,
                use_reentrant=False
            )
            edge_features = self.edge_ln(edge_features + edge_update)

        if self.triangle_mult is not None:
            edge_features = torch.utils.checkpoint.checkpoint(
                self.triangle_mult,
                node_features,
                rigids,
                edge_features,
                edge_index,
                use_reentrant=False
            )

        res_mask = ~(res_data['res_mask'].bool())
        noising_mask = res_data['noising_mask']

        node_s_update = self.attn_spatial(
            s=node_features,
            z=edge_features,
            edge_index=edge_index,
            r=rigids,
            mask=(~res_mask).float()
        )
        node_s_update = node_s_update * (~res_mask)[..., None]
        node_features = self.ln_s1(node_features + node_s_update)

        if self.use_seq_edge:
            if self.use_ipmp_seq_edge:
                node_s_update, seq_edge_features = self.attn_seq(
                    s=node_features,
                    z=seq_edge_features,
                    edge_index=seq_edge_index,
                    r=rigids,
                    mask=(~res_mask).float()
                )
            else:
                node_s_update = self.attn_seq(
                    s=node_features,
                    z=seq_edge_features,
                    edge_index=seq_edge_index,
                    r=rigids,
                    mask=(~res_mask).float()
                )
                node_s_update = node_s_update * (~res_mask)[..., None]
                node_features = self.ln_s2(node_features + node_s_update)

        # node_update, vn_features = self.vn_update(
        #     node_features,
        #     vn_features,
        #     res_data.batch,
        #     res_data['res_mask'].bool()
        # )
        # node_features = node_features + self.vn_update_proj(node_update)

        node_features = self.node_transition(node_features)
        node_features = node_features * (~res_mask)[..., None]
        rigids_update = self.bb_update(
            node_features * noising_mask[..., None])

        rigids = rigids.compose_q_update_vec(
            rigids_update * noising_mask[..., None]
        )
        if hasattr(self, "seq_edge_transition"):
            seq_edge_features = self.seq_edge_transition(node_features, seq_edge_features, seq_edge_index)

        # return node_features, vn_features, rigids, seq_edge_features
        return node_features, rigids, seq_edge_features


class IPMPRefineLayer(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden

        self.ipmp = IPMP(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            dropout=0.,
            edge_dropout=0.,
            final_init="final"
        )

        self.bb_update = BackboneUpdate(
            c_s
        )

    def forward(self,
                node_features,
                rigids,
                edge_features,
                edge_index,
                res_data):
        res_mask = res_data['res_mask']
        noising_mask = res_data['noising_mask']

        # node_update = self.ipmp(
        node_features, edge_features = self.ipmp(
            s=node_features,
            z=edge_features,
            edge_index=edge_index,
            r=rigids,
            mask=res_mask)

        rigids_update = self.bb_update(
            node_features * noising_mask[..., None])

        rigids = rigids.compose_q_update_vec(
            rigids_update * noising_mask[..., None]
        )

        return node_features, rigids, edge_features


class DynamicGraphIpaFrameDenoiser(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 c_s=256,
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
                 use_anchors=False,
                 interres_equilibrate=False,
                 use_triangle_attn=False,
                 use_triangle_mult=False,
                 use_self_edge=False,
                 update_seq_edge=False,
                 use_ipmp_seq_edge=False,
                 first_seq_edge_only=False,
                 learnable_scale=False,
                 use_ipmp_refine=False,
                 impute_oxy=False):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.num_rbf = c_z//2
        self.vn_mode = vn_mode
        self.num_vn = num_vn
        self.self_conditioning = self_conditioning
        if graph_conditioning:
            assert self_conditioning, "graph conditioning requires self-conditioning"
        self.graph_conditioning = graph_conditioning
        self.n_layers = n_layers
        self.impute_oxy = impute_oxy
        self.update_seq_edge = update_seq_edge
        self.use_self_edge = use_self_edge
        self.learnable_scale = learnable_scale
        self.first_seq_edge_only = first_seq_edge_only

        self.h_time = h_time
        self.time_rbf = GaussianRandomFourierBasis(n_basis=h_time//2)

        self.embed_node = nn.Sequential(
            nn.Linear(
                # node_embedding + time_embedding + fixed_mask + self_conditioning
                c_s + h_time + 1 + self_conditioning * (c_s + 7),
                c_s
            ),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
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
            GraphIpaFrameDenoisingLayer2(
                 c_s,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts,
                 self_conditioning=self_conditioning,
                 embed_seq_edge=update_seq_edge,
                 use_seq_edge=(i == 0 if first_seq_edge_only else True),
                 last=update_seq_edge,
                 use_ipmp_seq_edge=use_ipmp_seq_edge,
                 knn_k=knn_k,
                 lrange_k=lrange_k,
                 num_rbf=self.num_rbf,
                 triangle_attention=use_triangle_attn,
                 triangle_mult=use_triangle_mult,
            )
            for i in range(n_layers)
        ])
        self.knn_k = knn_k
        self.lrange_k = lrange_k

        self.torsion_angles = TorsionAngles(c_s, 1)

        if self.learnable_scale:
            self.scales = nn.ParameterList([
                nn.Parameter(torch.ones(1,1).float())
                for _ in range(n_layers)
            ])

        self.use_ipmp_refine = use_ipmp_refine
        if self.use_ipmp_refine:
            self.ipmp_num_rbf = 16
            self.ipmp_num_pos_embed = 16
            atoms_per_res = 5
            self.c_z_in = (
                self.ipmp_num_rbf * (atoms_per_res ** 2)  # bb x bb distances
                # + atoms_per_res * 3  # src CA to dst bb direction unit vectors
                + self.ipmp_num_pos_embed  # rel pos embed
            )
            self.embed_ipmp_edge = nn.Sequential(
                nn.Linear(self.c_z_in, c_z),
                nn.ReLU(),
                nn.Linear(c_z, c_z),
                nn.ReLU(),
                nn.Linear(c_z, c_z),
                nn.LayerNorm(c_z)
            )

            self.refine = nn.ModuleList([
                IPMPRefineLayer(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_hidden
                )
                for _ in range(n_layers)
            ])

    def _gen_ipmp_edge_features(self, rigids, psis, batch, res_mask, eps=1e-8):
        # edge graph
        masked_X_ca = rigids.get_trans().clone()
        masked_X_ca[~res_mask] = torch.inf
        edge_index = knn_graph(masked_X_ca, self.knn_k, batch)

        # edge features
        bb = compute_backbone(rigids.unsqueeze(0), psis.unsqueeze(0))[-1].squeeze(0)[..., :5, :]
        src = edge_index[1]
        dst = edge_index[0]

        # ## edge distances
        edge_bb_src = bb[src]
        edge_bb_dst = bb[dst]
        edge_bb_dists = torch.linalg.vector_norm(
            edge_bb_src[..., None, :] - edge_bb_dst[..., None, :, :] + eps,
            dim=-1)
        edge_bb_dists = edge_bb_dists.view(edge_index.shape[1], -1, 1)
        edge_rbf = _rbf(edge_bb_dists, D_min=2.0, D_max=22.0, D_count=self.ipmp_num_rbf, device=edge_index.device)
        edge_rbf = edge_rbf.view(edge_index.shape[1], -1)
        # ## edge rel pos embedding
        edge_dist_rel_pos = _edge_positional_embeddings(edge_index, num_embeddings=self.ipmp_num_pos_embed, device=edge_index.device)
        # edge_features = self.init_edge_embed(graph, edge_index)
        # technically this shouldn't be necessary but just to be safe
        # edge_mask = res_mask[src] & res_mask[dst]
        # edge_features = edge_features * edge_mask[..., None]
        edge_features = torch.cat([edge_rbf, edge_dist_rel_pos], dim=-1)

        return edge_features, edge_index


    def _gen_spatial_edge_features(self, rigids, res_mask, batch, self_condition):
        res_mask = ~res_mask
        # generate spatial edges
        X_ca = rigids.get_trans()
        if self.graph_conditioning and self_condition is not None:
            self_cond_X_ca = self_condition['final_rigids'].get_trans()
            masked_X_ca = self_cond_X_ca.clone()
            masked_X_ca[res_mask] = torch.inf
            edge_index = sample_inv_cubic_edges(masked_X_ca, res_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k, self_edge=self.use_self_edge)
        else:
            masked_X_ca = X_ca.clone()
            masked_X_ca[res_mask] = torch.inf
            edge_index = sample_inv_cubic_edges(masked_X_ca, res_mask, batch, knn_k=self.knn_k, inv_cube_k=self.lrange_k, self_edge=self.use_self_edge)

        # compute edge features
        edge_features, _ = gen_spatial_graph_features(X_ca, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)

        if self.self_conditioning and self_condition is not None:
            self_cond_rigids = self_condition['final_rigids']
            self_cond_X_ca = self_cond_rigids.get_trans()
            self_cond_edge_features, _ = gen_spatial_graph_features(self_cond_X_ca, edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=0)
            edge_features = torch.cat(
                [edge_features, self_cond_edge_features],
                dim=-1
            )
        elif self.self_conditioning:
            edge_features = F.pad(
                edge_features,
                (0, self.c_z//2)
            )

        return edge_features, edge_index


    def _gen_inital_features(self, data, self_condition):
        res_data = data['residue']
        ## prep features
        # ts = res_data['t']  # (B,)
        ts = data['t']  # (B,)
        res_mask = res_data['res_mask'].bool()
        rigids_t = ru.Rigid.from_tensor_7(res_data['rigids_t'])
        batch = res_data.batch
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
        # vn_features = self.embed_vn(fourier_time).view(-1, self.num_vn, self.c_s)
        vn_features = 0
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
                res_data['noising_mask'].float()[..., None]
            ], dim=-1)
        if self.self_conditioning and self_condition is not None:
            self_cond_rigids = self_condition['final_rigids']
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
                [node_input, self_cond_nodes, t7_rel],
                dim=-1
            )

        elif self.self_conditioning:
            node_input = F.pad(
                node_input,
                (0, self.c_s + 7)
            )

        if self.learnable_scale:
            scale = self.scales[0]
        else:
            scale = 1
        edge_features, edge_index = self._gen_spatial_edge_features(rigids_t.scale_translation(scale), res_mask, batch, self_condition)

        return node_input, vn_features, edge_features, edge_index, seq_edge_features, seq_local_edge_index
        # return node_input, vn_features, seq_edge_features, seq_local_edge_index

    def forward(self, data, self_condition=None):
        res_data = data['residue']
        res_mask = (res_data['res_mask']).bool()

        (
            node_input,
            vn_features,
            raw_edge_features,
            edge_index,
            seq_edge_features,
            seq_local_edge_index
        ) = self._gen_inital_features(data, self_condition=self_condition)

        # embed features
        node_features = self.embed_node(node_input)
        node_features = node_features * res_mask[..., None]
        if not self.update_seq_edge:
            seq_edge_features = self.seq_edge_embed(seq_edge_features)

        rigids_t = ru.Rigid.from_tensor_7(res_data['rigids_t'])
        # center the training example at the mean of the x_cas
        center = ru.batchwise_center(rigids_t, res_data.batch, res_data['res_mask'].bool())
        rigids_t = rigids_t.translate(-center)

        ## denoising
        rigids_t = rigids_t.scale_translation(0.1)
        rigids = rigids_t

        rigids_history = []
        for i, layer in enumerate(self.denoiser):
            if i > 0:
                if self.learnable_scale:
                    scale = self.scales[i]
                else:
                    scale = 1
                # recompute graph
                raw_edge_features, edge_index = self._gen_spatial_edge_features(
                    rigids.scale_translation(scale),
                    res_mask,
                    res_data.batch,
                    self_condition)

                if self.update_seq_edge:
                    X_ca = rigids.get_trans()
                    seq_edge_features, _ = gen_spatial_graph_features(X_ca, seq_local_edge_index, num_rbf_embed=self.c_z//2, num_pos_embed=self.c_z//2)

            # node_features, vn_features, rigids, seq_edge_features = layer(
            #     node_features,
            #     vn_features,
            #     rigids,
            #     raw_edge_features,
            #     edge_index,
            #     seq_edge_features,
            #     seq_local_edge_index,
            #     res_data,
            #     update_edge=(i < self.n_layers-1)
            # )
            node_features, rigids, seq_edge_features = layer(
                node_features,
                rigids,
                raw_edge_features,
                edge_index,
                seq_edge_features,
                seq_local_edge_index,
                res_data,
            )
            # if i < len(self.denoiser) - 1:
            #     rigids = rigids.map_tensor_fn(lambda x: x.detach())
            #     rigids._trans.requires_grad = True
            #     rigids._rots._quats.requires_grad = True

            # need to rescale and re-translate rigids to original reference
            rigids_history.append(rigids.scale_translation(10).translate(center))

        _, psi = self.torsion_angles(node_features)

        rigids = rigids.scale_translation(10)

        if self.use_ipmp_refine:
            ipmp_edge_features, ipmp_edge_index = self._gen_ipmp_edge_features(
                rigids, psi, res_data.batch, res_mask
            )
            ipmp_edge_features = self.embed_ipmp_edge(ipmp_edge_features)
            for i, layer in enumerate(self.refine):
                node_features, rigids, ipmp_edge_features = layer(
                    node_features,
                    rigids,
                    ipmp_edge_features,
                    ipmp_edge_index,
                    res_data,
                )
        _, psi = self.torsion_angles(node_features)
        rigids = rigids.translate(center)
        ret = {}
        ret['denoised_frames'] = rigids
        ret['intermediate_rigids'] = rigids_history
        ret['final_rigids'] = rigids
        denoised_bb_items = compute_backbone(rigids.unsqueeze(0), psi.unsqueeze(0))
        denoised_bb = denoised_bb_items[-1].squeeze(0)
        if self.impute_oxy:
            denoised_bb = adjust_oxygen_pos(denoised_bb, res_mask)
        denoised_bb = denoised_bb[:, :5]
        ret['denoised_bb'] = denoised_bb
        ret['psi'] = psi
        ret['node_features'] = node_features
        ret['edge_index'] = edge_index

        return ret
