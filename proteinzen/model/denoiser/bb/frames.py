""" Denoising model """

import torch
from torch import nn
import torch.nn.functional as F
from proteinzen.model.modules.common import GaussianRandomFourierBasis
from proteinzen.model.modules.layers.edge.sitewise import EdgeTransition
from proteinzen.model.modules.layers.node.attention import GraphInvariantPointAttention
from proteinzen.model.modules.layers.lrange.vn import VirtualNodeAttnUpdate, VirtualNodeMPNNUpdate
from proteinzen.model.modules.layers.triangle.attention import TriangleSelfAttention, TriangleCrossAttention, SparseTriangleSelfAttention, SparseTriangleCrossAttention
from proteinzen.model.modules.layers.triangle.mult import TriangleMultiplicativeUpdate, SparseTriangleMultiplicativeUpdate, NestedTriangleMultiplicativeUpdate, SparseTriangleCrossMultiplicativeUpdate, TriangleCrossMultiplicativeUpdate
from proteinzen.model.modules.layers.node.mpnn import IPMP

from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.utils.framediff.all_atom import compute_backbone, adjust_oxygen_pos

from proteinzen.data.datasets.featurize.common import _node_positional_embeddings
from proteinzen.data.datasets.featurize.common import _edge_positional_embeddings, _rbf
from proteinzen.data.datasets.featurize.sidechain import _dihedrals, _ideal_virtual_Cb

from proteinzen.model.modules.openfold.layers import BackboneUpdate, StructureModuleTransition, Linear

from proteinzen.model.utils.graph import sample_inv_cubic_edges, sample_logn_inv_cubic_edges, sequence_local_graph, gen_spatial_graph_features, batchwise_to_nodewise, get_data_lens, sample_all_edges
from proteinzen.model.utils.graph import sparse_to_knn_graph, knn_to_sparse_graph

from .framediff import TorsionAngles


from proteinzen.model.modules.layers.lrange.anchor import ProjectivePoolUpdate, AnchorUpdate
from proteinzen.model.modules.layers.interres import OneParamPairwiseEquilibrate

from torch_geometric.utils import coalesce, sort_edge_index, scatter, to_undirected
from torch_cluster import knn, knn_graph

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


class AnchorNodeLayer(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden,
                 num_heads,
                 num_qk_pts,
                 num_v_pts):
        super().__init__()
        self.anchors_ipa = GraphInvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=num_heads,
            no_qk_points=num_qk_pts,
            no_v_points=num_v_pts,
        )
        self.anchors_ln = nn.LayerNorm(c_s)

        self.anchor_edge_transition = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z
        )

    def forward(self,
                node_features,
                rigids,
                anchor_edge_features,
                anchor_edge_index,
                res_mask):

        node_update = self.anchor_ipa(
            s=node_features,
            z=anchor_edge_features,
            edge_index=anchor_edge_index,
            r=rigids,
            mask=res_mask.float()
        )
        node_update = node_update * res_mask[..., None]
        node_features = self.anchor_ln(node_features + node_update)

        a2a_edge_features = self.anchor_edge_transition(node_features, anchor_edge_features, anchor_edge_index)

        return node_features, anchor_edge_features


def gen_anchor_graphs(num_nodes, nodes_per_anchor, device):
    num_anchors = []
    anchor_load = []
    anchor_node_idx = []
    num_nodes = [int(i) for i in num_nodes]

    offset = 0
    for n in num_nodes:
        num_full_anchors = n // nodes_per_anchor
        num_last_anchor = n % nodes_per_anchor
        anchor_load += [nodes_per_anchor] * num_full_anchors
        if num_last_anchor > 0:
            anchor_load.append(num_last_anchor)
        anchor_node_idx += [
            nodes_per_anchor * i + nodes_per_anchor//2 + offset
            for i in range(num_full_anchors)
        ]
        if num_last_anchor > 0:
            anchor_node_idx.append(num_full_anchors * nodes_per_anchor + num_last_anchor//2 + offset)
        offset += n

        if num_last_anchor > 0:
            num_anchors.append(num_full_anchors + 1)
        else:
            num_anchors.append(num_full_anchors)

    anchor_node_idx = torch.as_tensor(anchor_node_idx)
    anchor_load = torch.as_tensor(anchor_load)

    node_idx = torch.arange(sum(num_nodes))
    anchor_idx = torch.repeat_interleave(anchor_load)
    anchor_idx = anchor_node_idx[anchor_idx]

    n2a_edge_index = torch.stack([node_idx, anchor_idx], dim=0)

    offset = 0
    a2a_edge_index = []
    num_anchors = [int(n) for n in num_anchors]
    for n in num_anchors:
        src = torch.repeat_interleave(
            torch.ones(n).long() * n
        )
        dst = torch.arange(n).repeat(n)
        a2a_edge_index.append(
            torch.stack([dst, src]) + offset
        )
        offset += n
    a2a_edge_index = torch.cat(a2a_edge_index, dim=-1)
    a2a_edge_index = anchor_node_idx[a2a_edge_index]

    return n2a_edge_index.to(device), a2a_edge_index.to(device)

class EdgeEmbed(nn.Module):
    def __init__(self,
                 edge_in,
                 edge_hidden,
                 edge_out,
                 n_layers=3,
                 gate_edge=False,
                 skip_conn=False,
    ):
        super().__init__()
        assert n_layers > 1
        layers = [Linear(edge_in, edge_hidden), nn.ReLU()]
        for _ in range(n_layers-2):
            layers.append(Linear(edge_hidden, edge_hidden))
            layers.append(nn.ReLU())
        self.last = Linear(edge_hidden, edge_out)

        self.mlp = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(edge_out)

        if gate_edge:
            self.gate = Linear(edge_hidden, edge_out)
        else:
            self.gate = None

        if skip_conn:
            self.skip = Linear(edge_in, edge_out)
        else:
            self.skip = None

    def forward(self, edge_embed):
        x = edge_embed
        x = self.mlp(x)

        if self.gate is not None:
            gate = self.gate(x)
            x = self.last(x)
            x = torch.sigmoid(gate) * x
        else:
            x = self.last(x)

        if self.skip is not None:
            x = x + self.skip(edge_embed)

        x = self.ln(x)

        return x


class GraphIpaFrameDenoisingLayer2(nn.Module):
    """ Denoising layer on sidechain densities """
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden,
                 ipa_num_heads,
                 num_qk_pts,
                 num_v_pts,
                 self_conditioning=False,
                 sc_edge_unit_vecs=False,
                 triangle_attention=False,
                 triangle_in_attention=False,
                 triangle_mult=False,
                 triangle_in_mult=False,
                 triangle_transfer=False,
                 triangle_in_transfer=False,
                 triangle_num_heads=4,
                 vn_mode=None,
                 embed_seq_edge=False,
                 use_seq_edge=True,
                 use_transformer=False,
                 use_ipmp_seq_edge=False,
                 add_nodes_to_edge=False,
                 first=False,
                 last=False,
                 transition_edges=False,
                 num_rbf=64,
                 knn_k=20,
                 lrange_k=40,
                 triangle_transfer_k=None,
                 triangle_attn_k=None,
                 use_old_edges=False,
                 fuse_new_edges_to_old_edges=True,
                 use_compile=False,
                 extra_node_transitions=False,
                 extra_seq_ipa=False,
                 use_anchors=False,
                 triangle_dropout=0,
                 grad_checkpoint=True,
                 gate_spatial_edges=False,
                 skip_conn_spatial_edges=False,
                 num_edge_classes=0,
                 ):
        """
        Args
        ----
        """
        super().__init__()

        self.first = first
        self.last = last
        self.grad_checkpoint = grad_checkpoint

        if triangle_attention:
            self.triangle_attention = TriangleSelfAttention(
                c_s=c_s,
                c_z=c_z,
                num_heads=triangle_num_heads,
                num_rbf=num_rbf,
                dropout=triangle_dropout
            )
            self.edge_ln = nn.LayerNorm(c_z)
        else:
            self.triangle_attention = None

        if triangle_in_attention:
            self.triangle_in_attention = SparseTriangleSelfAttention(
                c_s=c_s,
                c_z=c_z,
                num_heads=triangle_num_heads,
                num_rbf=num_rbf,
                dropout=triangle_dropout
            )
            # self.triangle_attention = TriangleCrossAttention2(
            #     c_s,
            #     c_z,
            #     triangle_num_heads,
            #     num_rbf
            # )
        else:
            self.triangle_in_attention = None

        if triangle_mult:
            self.triangle_mult = TriangleMultiplicativeUpdate(
                c_s=c_s,
                c_z=c_z,
                num_rbf=num_rbf,
            )
            self.triangle_mult = torch.compile(self.triangle_mult, dynamic=True)
            self.triangle_mult_ln = nn.LayerNorm(c_z)
        else:
            self.triangle_mult = None

        if triangle_in_mult:
            self.triangle_in_mult = SparseTriangleMultiplicativeUpdate(
                c_s=c_s,
                c_z=c_z,
                num_rbf=num_rbf,
            )
        else:
            self.triangle_in_mult = None

        if triangle_transfer:
            # self.triangle_transfer = TriangleCrossAttention(
            #     c_s,
            #     c_z,
            #     num_heads=triangle_num_heads,
            #     dropout=triangle_dropout
            # )
            self.triangle_transfer = TriangleCrossMultiplicativeUpdate(
                c_s,
                c_z,
                # num_heads=triangle_num_heads,
                dropout=triangle_dropout
            )
            self.triangle_transfer = torch.compile(self.triangle_transfer, dynamic=True)
            self.transfer_edge_ln = nn.LayerNorm(c_z)
        else:
            self.triangle_transfer = None

        if triangle_in_transfer:
            self.triangle_in_transfer = SparseTriangleCrossAttention(
                c_s,
                c_z,
                num_heads=triangle_num_heads,
                dropout=triangle_dropout
            )
        else:
            self.triangle_in_transfer = None


        self.attn_spatial = GraphInvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=ipa_num_heads,
            no_qk_points=num_qk_pts,
            no_v_points=num_v_pts,
        )
        self.ln_s1 = nn.LayerNorm(c_s)
        if extra_node_transitions:
            self.spatial_node_transition = StructureModuleTransition(c_s)
        else:
            self.spatial_node_transition = None


        self.use_seq_edge = use_seq_edge
        self.extra_seq_ipa = extra_seq_ipa
        if use_seq_edge:
            if add_nodes_to_edge:
                self.seq_edge_update = nn.Sequential(
                    nn.Linear(c_s*2 + c_z + 4 + self_conditioning * (c_z//2 + 4 + 3 * sc_edge_unit_vecs), c_z),
                    nn.ReLU(),
                    nn.Linear(c_z, c_z),
                    nn.ReLU(),
                    nn.Linear(c_z, c_z),
                )
            else:
                self.seq_edge_update = nn.Sequential(
                    nn.Linear(c_z + 4 + self_conditioning * (c_z//2 + 4 + 3 * sc_edge_unit_vecs), c_z),
                    nn.ReLU(),
                    nn.Linear(c_z, c_z),
                    nn.ReLU(),
                    nn.Linear(c_z, c_z),
                )
            self.seq_edge_ln = nn.LayerNorm(c_z)

            self.seq_edge_transition = EdgeTransition(
                node_embed_size=c_s,
                edge_embed_in=c_z,
                edge_embed_out=c_z
            )
            self.attn_seq = GraphInvariantPointAttention(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                no_heads=ipa_num_heads,
                no_qk_points=num_qk_pts,
                no_v_points=num_v_pts,
            )
            self.ln_s2 = nn.LayerNorm(c_s)

            if extra_node_transitions:
                self.seq_node_transition = StructureModuleTransition(c_s)
            else:
                self.seq_node_transition = None

            if extra_seq_ipa:
                self.extra_attn_seq = GraphInvariantPointAttention(
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_hidden,
                    no_heads=ipa_num_heads,
                    no_qk_points=num_qk_pts,
                    no_v_points=num_v_pts,
                )
                self.extra_ln_s2 = nn.LayerNorm(c_s)
                if extra_node_transitions:
                    self.extra_seq_node_transition = StructureModuleTransition(c_s)
                else:
                    self.extra_seq_node_transition = None
            else:
                self.extra_attn_seq = None
                self.extra_ln_s2 = None
                self.extra_seq_node_transition = None

        self.use_anchors = use_anchors
        if use_anchors:
            if add_nodes_to_edge:
                self.anchor_edge_update = nn.Sequential(
                    nn.Linear(c_s*2 + c_z + 4 + self_conditioning * (c_z//2 + 4), c_z),
                    nn.ReLU(),
                    nn.Linear(c_z, c_z),
                    nn.ReLU(),
                    nn.Linear(c_z, c_z),
                )
            else:
                self.anchor_edge_update = nn.Sequential(
                    nn.Linear(c_z + 4 + self_conditioning * (c_z//2 + 4), c_z),
                    nn.ReLU(),
                    nn.Linear(c_z, c_z),
                    nn.ReLU(),
                    nn.Linear(c_z, c_z),
                )
            self.anchor_edge_ln = nn.LayerNorm(c_z)

            self.attn_anchor = GraphInvariantPointAttention(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                no_heads=ipa_num_heads,
                no_qk_points=num_qk_pts,
                no_v_points=num_v_pts,
            )
            self.anchor_ln = nn.LayerNorm(c_s)

        else:
            self.anchor_edge_update = None
            self.anchor_update = None
            self.anchor_edge_ln = None
            self.anchor_ln = None


        self.use_transformer = use_transformer
        if use_transformer:
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=c_s,
                nhead=4,
                dim_feedforward=c_s,
                batch_first=True,
                dropout=0.2,
                norm_first=False
            )
            self.seq_tfmr = torch.nn.TransformerEncoder(
                tfmr_layer, 4, enable_nested_tensor=True)
            self.post_tfmr = Linear(
                c_s, c_s, init="final")

        assert vn_mode in ["attn", "mpnn", None], f"unsupported vn_mode {vn_mode}"
        self.vn_mode = vn_mode
        if vn_mode == 'attn':
            self.vn_update = VirtualNodeAttnUpdate(
                c_s,
                c_s // ipa_num_heads,
                num_heads=ipa_num_heads
            )
        elif vn_mode == 'mpnn':
            self.vn_update = VirtualNodeMPNNUpdate(
                c_s,
                num_heads=ipa_num_heads
            )

        self.bb_update = BackboneUpdate(
            c_s
        )
        self.node_transition = StructureModuleTransition(
            c=c_s
        )
        self.add_nodes_to_edge = add_nodes_to_edge
        if add_nodes_to_edge:
            self.edge_embed = EdgeEmbed(
                edge_in=c_s*2 + c_z + 4 + self_conditioning * (c_z//2 + 4 + 3 * sc_edge_unit_vecs) + num_edge_classes,
                edge_hidden=c_z,
                edge_out=c_z,
                gate_edge=gate_spatial_edges,
                skip_conn=skip_conn_spatial_edges
            )
            # self.edge_embed = nn.Sequential(
            #     nn.Linear(c_s*2 + c_z + 4 + self_conditioning * (c_z//2 + 4), c_z),
            #     nn.ReLU(),
            #     nn.Linear(c_z, c_z),
            #     nn.ReLU(),
            #     nn.Linear(c_z, c_z),
            #     nn.LayerNorm(c_z)
            # )
        else:
            self.edge_embed = EdgeEmbed(
                edge_in=c_z + 4 + self_conditioning * (c_z//2 + 4 + 3 * sc_edge_unit_vecs) + num_edge_classes,
                edge_hidden=c_z,
                edge_out=c_z,
                gate_edge=gate_spatial_edges,
                skip_conn=skip_conn_spatial_edges
            )
            # self.edge_embed = nn.Sequential(
            #     nn.Linear(c_z + 4 + self_conditioning * (c_z//2 + 4), c_z),
            #     nn.ReLU(),
            #     nn.Linear(c_z, c_z),
            #     nn.ReLU(),
            #     nn.Linear(c_z, c_z),
            #     nn.LayerNorm(c_z)
            # )

        if transition_edges:
            self.edge_transition = EdgeTransition(
                node_embed_size=c_s,
                edge_embed_in=c_z,
                edge_embed_out=c_z
            )
        else:
            self.edge_transition = None

        self.use_old_edges = use_old_edges
        self.fuse_new_edges_to_old_edges = fuse_new_edges_to_old_edges
        if use_old_edges and fuse_new_edges_to_old_edges:
            self.fuse_edge_features = nn.Linear(c_z, c_z)
            self.fuse_ln = nn.LayerNorm(c_z)

        if use_compile:
            # for mod in ['attn_spatial', 'attn_seq', 'seq_edge_update', 'seq_edge_transition', 'edge_embed', 'edge_transition']:
            for mod in ['triangle_attention', 'triangle_mult']:
                if hasattr(self, mod):
                    mod_obj = getattr(self, mod)
                    if mod_obj is not None:
                        setattr(self, mod, torch.compile(mod_obj))

        self.lrange_k = lrange_k
        self.knn_k = knn_k
        if triangle_transfer_k is None:
            self.triangle_transfer_k = lrange_k + knn_k
        else:
            self.triangle_transfer_k = triangle_transfer_k

        if triangle_attn_k is None:
            self.triangle_attn_k = lrange_k + knn_k
        else:
            self.triangle_attn_k = triangle_attn_k

    def forward(
            self,
            node_features,
            rigids,
            vn_features,
            old_edge_features,
            old_edge_index,
            edge_features,
            edge_index,
            knn_edge_select,
            lrange_edge_select,
            new_seq_edge_inputs,
            seq_edge_features,
            seq_edge_index,
            new_anchor_edge_inputs,
            anchor_edge_features,
            anchor_edge_index,
            res_data,
    ):
        if self.use_old_edges and not self.fuse_new_edges_to_old_edges and not self.first:
            edge_features = old_edge_features
            edge_index = old_edge_index
        else:
            if self.add_nodes_to_edge:
                edge_inputs = torch.cat([
                    edge_features,
                    node_features[edge_index[0]],
                    node_features[edge_index[1]]
                ], dim=-1)
                edge_features = self.edge_embed(edge_inputs)
            else:
                edge_features = self.edge_embed(edge_features)

            if self.use_old_edges and self.fuse_new_edges_to_old_edges:
                if self.first:
                    old_edge_features = edge_features
                    old_edge_index = edge_index
                sorted_old_edge_index, sorted_old_edge_features = sort_edge_index(
                    old_edge_index,
                    old_edge_features,
                    sort_by_row=False
                )
                sorted_edge_index, sorted_edge_features = sort_edge_index(
                    edge_index,
                    edge_features,
                    sort_by_row=False)
                assert sorted_old_edge_index.shape == sorted_edge_index.shape, (sorted_old_edge_index.shape, sorted_edge_index.shape)
                assert (sorted_old_edge_index == sorted_edge_index).all()
                edge_index = sorted_edge_index
                edge_features = self.fuse_ln(
                    self.fuse_edge_features(sorted_edge_features)
                    + sorted_old_edge_features
                )

        if self.use_seq_edge:
            if self.add_nodes_to_edge:
                seq_edge_inputs = torch.cat([
                    new_seq_edge_inputs,
                    node_features[seq_edge_index[0]],
                    node_features[seq_edge_index[1]]
                ], dim=-1)
                seq_edge_features = self.seq_edge_ln(
                    seq_edge_features +
                    self.seq_edge_update(seq_edge_inputs)
                )
            else:
                seq_edge_features = self.seq_edge_ln(
                    seq_edge_features +
                    self.seq_edge_update(new_seq_edge_inputs)
                )

        if self.triangle_transfer is not None:
            if self.first:
                old_edge_features = edge_features
                old_edge_index = edge_index

            if self.grad_checkpoint:
                # edge_update = torch.utils.checkpoint.checkpoint(
                #     self.triangle_transfer,
                #     node_features,
                #     rigids,
                #     old_edge_features,
                #     old_edge_index,
                #     edge_features,
                #     edge_index,
                #     k=self.triangle_transfer_k,
                #     use_reentrant=False)
                # the min edge stuff is to safeguard against issues with "bad edges"
                min_num_edges = scatter(
                    torch.ones_like(old_edge_index[1]),
                    old_edge_index[1],
                    dim=0,
                    dim_size=old_edge_index.max()+1
                )
                min_num_edges = min_num_edges[min_num_edges != 0]
                transfer_k = min(min_num_edges.min().item(), self.triangle_transfer_k)

                old_dst = old_edge_index[0]
                old_src = old_edge_index[1]
                rigid_trans = rigids.get_trans()
                old_X_dst = rigid_trans[old_dst]
                new_dst = edge_index[0]
                new_src = edge_index[1]
                new_X_dst = rigid_trans[new_dst]
                transfer_index = knn(old_X_dst, new_X_dst, k=transfer_k, batch_x=old_src, batch_y=new_src)
                transfer_index = torch.stack([transfer_index[1], transfer_index[0]], dim=0)  # "source_to_target"
                # transfer_edge_features = old_edge_features[transfer_index[0]]
                # transfer_edge_index = old_edge_index[:, transfer_index[0]]

                # # this isn't resistant to bugs but it should be much faster
                # num_edges = edge_features.shape[0]
                # knn_transfer_edge_features = transfer_edge_features.view(num_edges, transfer_k, -1)
                # knn_transfer_edge_index = transfer_edge_index.view(2, num_edges, transfer_k)
                edge_update = torch.utils.checkpoint.checkpoint(
                    self.triangle_transfer,
                    node_features,
                    rigids,
                    transfer_index,
                    transfer_k,
                    old_edge_features,
                    old_edge_index,
                    edge_features,
                    edge_index,
                    use_reentrant=False
                    # k=self.triangle_transfer_k
                )
            else:
                # # the min edge stuff is to safeguard against issues with "bad edges"
                # min_num_edges = scatter(
                #     torch.ones_like(old_edge_index[1]),
                #     old_edge_index[1],
                #     dim=0,
                #     dim_size=old_edge_index.max()+1
                # )
                # min_num_edges = min_num_edges[min_num_edges != 0]
                # transfer_k = min(min_num_edges.min().item(), self.triangle_transfer_k)

                # old_dst = old_edge_index[0]
                # old_src = old_edge_index[1]
                # rigid_trans = rigids.get_trans()
                # old_X_dst = rigid_trans[old_dst]
                # new_dst = edge_index[0]
                # new_src = edge_index[1]
                # new_X_dst = rigid_trans[new_dst]
                # transfer_index = knn(old_X_dst, new_X_dst, k=transfer_k, batch_x=old_src, batch_y=new_src)
                # transfer_index = torch.stack([transfer_index[1], transfer_index[0]], dim=0)  # "source_to_target"
                # # transfer_edge_features = old_edge_features[transfer_index[0]]
                # # transfer_edge_index = old_edge_index[:, transfer_index[0]]

                # # # this isn't resistant to bugs but it should be much faster
                # # num_edges = edge_features.shape[0]
                # # knn_transfer_edge_features = transfer_edge_features.view(num_edges, transfer_k, -1)
                # # knn_transfer_edge_index = transfer_edge_index.view(2, num_edges, transfer_k)
                # edge_update = self.triangle_transfer(
                #     node_features,
                #     rigids,
                #     transfer_index,
                #     transfer_k,
                #     old_edge_features,
                #     old_edge_index,
                #     edge_features,
                #     edge_index,
                #     # k=self.triangle_transfer_k
                # )
                edge_update = self.triangle_transfer(
                    node_features,
                    rigids,
                    old_edge_features,
                    old_edge_index,
                    edge_features,
                    edge_index,
                    batch=res_data.batch,
                    res_mask=res_data['res_mask']
                    # k=self.triangle_transfer_k
                )

            if self.triangle_in_transfer is not None:
                flip_old_edge_index = old_edge_index.flip(0)
                flip_edge_index = edge_index.flip(0)

                if self.grad_checkpoint:
                    edge_in_update = torch.utils.checkpoint.checkpoint(
                        self.triangle_in_transfer,
                        node_features,
                        rigids,
                        old_edge_features,
                        flip_old_edge_index,
                        edge_features,
                        flip_edge_index,
                        k=self.triangle_transfer_k,
                        use_reentrant=False)
                else:
                    edge_in_update = self.triangle_in_transfer(
                        node_features,
                        rigids,
                        old_edge_features,
                        flip_old_edge_index,
                        edge_features,
                        flip_edge_index,
                        k=self.triangle_transfer_k)
            else:
                edge_in_update = 0
            edge_features = self.transfer_edge_ln(edge_features + edge_update)

        if self.triangle_attention is not None:
            if self.grad_checkpoint:
                edge_update = torch.utils.checkpoint.checkpoint(
                    self.triangle_attention,
                    node_features,
                    rigids,
                    edge_features,
                    edge_index,
                    k=self.triangle_attn_k-1,
                    use_reentrant=False)
            else:
                edge_update = self.triangle_attention(
                    node_features,
                    rigids,
                    edge_features,
                    edge_index,
                    #k=self.triangle_attn_k-1
                )

            if self.triangle_in_attention is not None:
                flip_edge_index = edge_index.flip(0)
                if self.grad_checkpoint:
                    edge_in_update = torch.utils.checkpoint.checkpoint(
                        self.triangle_in_attention,
                        node_features,
                        rigids,
                        edge_features,
                        flip_edge_index,
                        k=self.triangle_attn_k-1,
                        use_reentrant=False)
                else:
                    edge_in_update = self.triangle_in_attention(
                        node_features,
                        rigids,
                        edge_features,
                        flip_edge_index,
                        k=self.triangle_attn_k-1)
            else:
                edge_in_update = 0
            edge_features = self.edge_ln(edge_features + edge_update + edge_in_update)


        if self.triangle_mult is not None:
            if self.grad_checkpoint:
                edge_update = torch.utils.checkpoint.checkpoint(
                    self.triangle_mult,
                    node_features,
                    rigids,
                    edge_features,
                    edge_index,
                    use_reentrant=False
                )
            else:
                edge_update = self.triangle_mult(
                    node_features,
                    rigids,
                    edge_features,
                    edge_index,
                    batch=res_data.batch,
                    res_mask=res_data['res_mask']
                )
            if self.triangle_in_mult is not None:
                if self.grad_checkpoint:
                    edge_in_update = torch.utils.checkpoint.checkpoint(
                        self.triangle_in_mult,
                        node_features,
                        rigids,
                        edge_features,
                        edge_index.flip(0),
                        use_reentrant=False
                    )
                else:
                    edge_in_update = self.triangle_mult(
                        node_features,
                        rigids,
                        edge_features,
                        edge_index.flip(0),
                    )
            else:
                edge_in_update = 0
            edge_features = self.triangle_mult_ln(edge_features + edge_update + edge_in_update)


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
        if self.spatial_node_transition is not None:
            node_features = self.spatial_node_transition(node_features)

        if self.use_anchors:
            if self.add_nodes_to_edge:
                anchor_edge_inputs = torch.cat([
                    new_anchor_edge_inputs,
                    node_features[anchor_edge_index[0]],
                    node_features[anchor_edge_index[1]]
                ], dim=-1)
                anchor_edge_features = self.anchor_edge_ln(
                    anchor_edge_features +
                    self.anchor_edge_update(anchor_edge_inputs)
                )
            else:
                anchor_edge_features = self.anchor_edge_ln(
                    anchor_edge_features +
                    self.anchor_edge_update(anchor_edge_inputs)
                )

            node_s_update = self.attn_anchor(
                s=node_features,
                z=anchor_edge_features,
                edge_index=anchor_edge_index,
                r=rigids,
                mask=(~res_mask).float()
            )
            node_s_update = node_s_update * (~res_mask)[..., None]
            node_features = self.anchor_ln(node_features + node_s_update)


        if self.use_seq_edge:
            node_s_update = self.attn_seq(
                s=node_features,
                z=seq_edge_features,
                edge_index=seq_edge_index,
                r=rigids,
                mask=(~res_mask).float()
            )
            node_s_update = node_s_update * (~res_mask)[..., None]
            node_features = self.ln_s2(node_features + node_s_update)
            if self.seq_node_transition is not None:
                node_features = self.seq_node_transition(node_features)

            if self.extra_seq_ipa:
                node_s_update = self.extra_attn_seq(
                    s=node_features,
                    z=seq_edge_features,
                    edge_index=seq_edge_index,
                    r=rigids,
                    mask=(~res_mask).float()
                )
                node_s_update = node_s_update * (~res_mask)[..., None]
                node_features = self.extra_ln_s2(node_features + node_s_update)
                if self.extra_seq_node_transition is not None:
                    node_features = self.extra_seq_node_transition(node_features)


        if self.use_transformer:
            data_lens = get_data_lens(res_data)
            split_node_features = node_features.split(data_lens, dim=0)

            padded_node_features = nn.utils.rnn.pad_sequence(
                split_node_features,
                batch_first=True
            )
            split_mask = res_mask.split(data_lens, dim=0)
            padded_mask = nn.utils.rnn.pad_sequence(
                split_mask,
                batch_first=True,
                padding_value=True)
            select_mask = nn.utils.rnn.pad_sequence(
                [torch.ones_like(m).bool() for m in split_mask],
                batch_first=True,
                padding_value=False)
            padded_node_updates = self.seq_tfmr(
                padded_node_features,
                src_key_padding_mask=padded_mask)
            node_updates = self.post_tfmr(padded_node_updates[select_mask])
            node_features = node_features + node_updates * (~res_mask)[..., None]


        if self.vn_mode is not None:
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
        if self.use_seq_edge:
            seq_edge_features = self.seq_edge_transition(node_features, seq_edge_features, seq_edge_index)

        if self.edge_transition is not None:
            edge_features = self.edge_transition(
                node_features,
                edge_features,
                edge_index
            )

        return node_features, rigids, vn_features, edge_features, seq_edge_features, anchor_edge_features
        # return node_features, rigids, seq_edge_features


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


class PairwiseAtomicEmbedding(nn.Module):
    def __init__(self,
                 num_rbf=16,
                 num_pos_embed=16,
                 scale=1,
                 D_min=0.0,
                 D_max=20.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.num_pos_embed = num_pos_embed

        self.scale = scale
        self.D_min = D_min
        self.D_max = D_max

        self.out_dim = (
            + (5 * 5) * num_rbf     # bb x bb dist
            + num_pos_embed         # rel pos embed
        )

    def forward(self, bb, edge_index, eps=1e-8):
        dst = edge_index[0]
        src = edge_index[1]
        num_edges = edge_index.shape[-1]
        edge_features = []

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

        return torch.cat(edge_features, dim=-1).float()


class IPMPSelfConditioningEncoder(nn.Module):
    def __init__(self,
                 c_s=128,
                 c_z=128,
                 c_hidden=128,
                 c_s_in=6,
                 num_rbf=16,
                 num_pos_embed=16,
                 num_layers=4,
                 k=30,
                 dropout=0.0):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.num_rbf = num_rbf
        self.num_pos_embed = num_pos_embed

        self.embed_node = nn.Sequential(
            nn.Linear(c_s_in, 2*c_s),
            nn.ReLU(),
            nn.Linear(2*c_s, 2*c_s),
            nn.ReLU(),
            nn.Linear(2*c_s, c_s),
            nn.LayerNorm(c_s),
        )
        self.init_edge_embed = PairwiseAtomicEmbedding(
            num_rbf=self.num_rbf,
            num_pos_embed=self.num_pos_embed,
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
            IPMP(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.k = k

    @torch.no_grad()
    def _prep_features(self, bb, res_mask, batch):
        # node features
        X_ca = bb[..., 1, :]
        dihedrals = _dihedrals(bb)
        node_features = dihedrals

        # edge graph
        masked_X_ca = X_ca.clone()
        masked_X_ca[~res_mask] = torch.inf
        edge_index = knn_graph(masked_X_ca, self.k, batch)

        # # edge features
        # virtual_Cb = _ideal_virtual_Cb(bb)
        # bb = torch.cat([bb, virtual_Cb[..., None, :]], dim=-2)
        src = edge_index[1]
        dst = edge_index[0]

        edge_features = self.init_edge_embed(bb, edge_index)
        # technically this shouldn't be necessary but just to be safe
        edge_mask = res_mask[src] & res_mask[dst]
        edge_features = edge_features * edge_mask[..., None]
        return node_features, edge_features, edge_index


    def forward(self, bb, rigids, res_mask, batch, eps=1e-8):
        ## prep features
        node_features, edge_features, edge_index = self._prep_features(bb, res_mask, batch)

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

        return node_features


class DynamicGraphIpaFrameDenoiser(nn.Module):
    """ Denoising model on sidechain densities """
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_hidden=16,
                 num_heads=16,
                 triangle_num_heads=4,
                 num_qk_pts=8,
                 num_v_pts=12,
                 h_time=64,
                 n_layers=8,
                 knn_k=20,
                 lrange_k=40,
                 undirected_spatial_edges=False,
                 knn_triangle_spatial_edges=False,
                 lrange_triangle_spatial_edges=False,
                 cross_triangle_spatial_edges=False,
                 triangle_transfer_k=None,
                 triangle_attn_k=None,
                 lrange_logn_scale=0,
                 lrange_logn_offset=0,
                 num_vn=4,
                 vn_mode=None,
                 self_conditioning=False,
                 graph_conditioning=False,
                 use_anchors=False,
                 interres_equilibrate=False,
                 use_triangle_attn=False,
                 use_triangle_in_attn=False,
                 use_triangle_mult=False,
                 use_triangle_in_mult=False,
                 use_triangle_transfer=False,
                 use_triangle_in_transfer=False,
                 triangle_dropout=0,
                 use_self_edge=False,
                 update_seq_edge=False,
                 use_ipmp_seq_edge=False,
                 use_seq_edge=True,
                 first_seq_edge_only=False,
                 learnable_scale=False,
                 use_ipmp_refine=False,
                 sc_learned_features=False,
                 sc_t7=True,
                 sc_edge_unit_vecs=False,
                 rbf_encode_sc_trans=False,
                 use_transformer=False,
                 add_nodes_to_edge=True,
                 use_edge_transition=False,
                 preserve_edges=False,
                 fuse_new_edges_to_old=True,
                 impute_oxy=False,
                 use_plddt=False,
                 num_plddt_bins=50,
                 ipmp_self_condition=False,
                 extra_node_transitions=False,
                 extra_seq_ipa=False,
                 grad_checkpoint=True,
                 gate_spatial_edges=True,
                 skip_conn_spatial_edges=False,
                 use_edge_colors=False,
                 num_edge_classes=5,
                 compile_mods=False
                 ):
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
        self.sc_learned_features = sc_learned_features
        self.sc_t7 = sc_t7
        self.sc_edge_unit_vecs = sc_edge_unit_vecs
        self.rbf_encode_sc_trans = rbf_encode_sc_trans
        self.use_plddt = use_plddt
        self.preserve_edges = preserve_edges
        self.ipmp_self_condition = ipmp_self_condition
        self.undirected_spatial_edges = undirected_spatial_edges
        self.knn_triangle_edges = knn_triangle_spatial_edges
        self.lrange_triangle_edges = lrange_triangle_spatial_edges
        self.cross_triangle_edges = cross_triangle_spatial_edges
        self.use_edge_colors = use_edge_colors
        self.num_edge_classes = num_edge_classes

        self.h_time = h_time
        self.time_rbf = GaussianRandomFourierBasis(n_basis=h_time//2)

        # node_embedding + time_embedding + fixed_mask + self_conditioning
        if rbf_encode_sc_trans:
            self.node_in = c_s + h_time + 1 + self_conditioning * (c_s + 7 + self.num_rbf + use_plddt * num_plddt_bins)
        elif ipmp_self_condition:
            self.self_condition_embed = IPMPSelfConditioningEncoder()
            self.node_in = c_s + h_time + 1 + self_conditioning * (c_s + 7 + self.self_condition_embed.c_s)
        else:
            self.node_in = c_s + h_time + 1 + self_conditioning * (c_s + 7 + use_plddt * num_plddt_bins)

        self.embed_node = nn.Sequential(
            nn.Linear(
                self.node_in,
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
                 sc_edge_unit_vecs=sc_edge_unit_vecs,
                 embed_seq_edge=update_seq_edge,
                 use_seq_edge=use_seq_edge,
                 last=update_seq_edge,
                 use_ipmp_seq_edge=use_ipmp_seq_edge,
                 knn_k=knn_k,
                 lrange_k=lrange_k,
                 triangle_transfer_k=triangle_transfer_k,
                 triangle_attn_k=triangle_attn_k,
                 num_rbf=self.num_rbf,
                 triangle_attention=use_triangle_attn,
                 triangle_in_attention=use_triangle_in_attn,
                 triangle_mult=use_triangle_mult,
                 triangle_in_mult=use_triangle_in_mult,
                 triangle_num_heads=triangle_num_heads,
                 vn_mode=vn_mode,
                 use_transformer=use_transformer,
                 add_nodes_to_edge=add_nodes_to_edge,
                 first=(i==0),
                 triangle_transfer=use_triangle_transfer,
                 triangle_in_transfer=use_triangle_in_transfer,
                 transition_edges=use_edge_transition,
                 use_old_edges=preserve_edges,
                 fuse_new_edges_to_old_edges=fuse_new_edges_to_old,
                 extra_node_transitions=extra_node_transitions,
                 extra_seq_ipa=extra_seq_ipa,
                 use_anchors=use_anchors,
                 grad_checkpoint=grad_checkpoint,
                 gate_spatial_edges=gate_spatial_edges,
                 skip_conn_spatial_edges=skip_conn_spatial_edges,
                 num_edge_classes=num_edge_classes * use_edge_colors,
                 use_compile=compile_mods,
            )
            for i in range(n_layers)
        ])
        self.knn_k = knn_k
        self.lrange_k = lrange_k
        self.lrange_logn_scale = lrange_logn_scale
        self.lrange_logn_offset = lrange_logn_offset

        self.torsion_angles = TorsionAngles(c_s, 1)

        if use_plddt:
            self.plddt_head = nn.Sequential(
                nn.LayerNorm(c_s),
                nn.Linear(c_s, c_s),
                nn.ReLU(),
                nn.Linear(c_s, c_s),
                nn.ReLU(),
                nn.Linear(c_s, num_plddt_bins)
            )

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


    def _gen_spatial_edge_features(self, rigids, res_mask, batch, self_condition, old_edge_index=None):
        res_mask = ~res_mask
        # generate spatial edges
        X_ca = rigids.get_trans()
        edge_colors = None
        if self.graph_conditioning and self_condition is not None:
            self_cond_X_ca = self_condition['final_rigids'].get_trans()
            masked_X_ca = self_cond_X_ca.clone()
            masked_X_ca[res_mask] = torch.inf
            edge_index, knn_edge_select, lrange_edge_select, edge_colors = sample_logn_inv_cubic_edges(
                masked_X_ca,
                res_mask,
                batch,
                knn_k=self.knn_k,
                min_inv_cube_edges=self.lrange_k,
                logn_scale=self.lrange_logn_scale,
                logn_offset=self.lrange_logn_offset,
                self_edge=self.use_self_edge,
                gen_knn_triangle_edges=self.knn_triangle_edges,
                gen_lrange_triangle_edges=self.lrange_triangle_edges,
                gen_cross_range_triangle_edges=self.cross_triangle_edges)
        else:
            masked_X_ca = X_ca.clone()
            masked_X_ca[res_mask] = torch.inf
            # TODO: fix this hack
            if self.lrange_logn_offset > 1000:
                edge_index, knn_edge_select, lrange_edge_select = sample_all_edges(
                    res_mask,
                    batch)
            else:
                if not self.preserve_edges or old_edge_index is None:
                    edge_index, knn_edge_select, lrange_edge_select, edge_colors = sample_logn_inv_cubic_edges(
                        masked_X_ca,
                        res_mask,
                        batch,
                        knn_k=self.knn_k,
                        min_inv_cube_edges=self.lrange_k,
                        logn_scale=self.lrange_logn_scale,
                        logn_offset=self.lrange_logn_offset,
                        self_edge=self.use_self_edge,
                        gen_knn_triangle_edges=self.knn_triangle_edges,
                        gen_lrange_triangle_edges=self.lrange_triangle_edges,
                        gen_cross_range_triangle_edges=self.cross_triangle_edges)
                else:
                    edge_index = old_edge_index
                    knn_edge_select = None
                    lrange_edge_select = None

        if self.undirected_spatial_edges:
            if knn_edge_select is None and lrange_edge_select is None:
                edge_index = to_undirected(edge_index)
            else:
                edge_index, (knn_edge_select, lrange_edge_select) = to_undirected(
                    edge_index,
                    [knn_edge_select.long(), lrange_edge_select.long()],
                    reduce="mul"
                )
                knn_edge_select = knn_edge_select.bool()
                lrange_edge_select = lrange_edge_select.bool()

        # compute edge features
        edge_features, _ = gen_spatial_graph_features(
            rigids,
            edge_index,
            num_rbf_embed=self.c_z//2,
            num_pos_embed=self.c_z//2
        )

        if edge_colors is not None and self.use_edge_colors:
            edge_colors = F.one_hot(edge_colors, num_classes=self.num_edge_classes)
            edge_features = torch.cat([edge_features, edge_colors], dim=-1)

        if self.self_conditioning and self_condition is not None:
            self_cond_rigids = self_condition['final_rigids']
            self_cond_edge_features, _ = gen_spatial_graph_features(
                self_cond_rigids,
                edge_index,
                num_rbf_embed=self.c_z//2,
                num_pos_embed=0,
                use_unit_vec=self.sc_edge_unit_vecs,
            )
            edge_features = torch.cat(
                [edge_features, self_cond_edge_features],
                dim=-1
            )
        elif self.self_conditioning:
            edge_features = F.pad(
                edge_features,
                (0, self.c_z//2 + 4 + 3 * self.sc_edge_unit_vecs)
            )

        return edge_features, edge_index, knn_edge_select, lrange_edge_select


    def _gen_seq_edge_features(self, rigids, seq_edge_index, self_condition):
        # compute edge features
        edge_features, _ = gen_spatial_graph_features(
            rigids,
            seq_edge_index,
            num_rbf_embed=self.c_z//2,
            num_pos_embed=self.c_z//2
        )

        if self.self_conditioning and self_condition is not None:
            self_cond_rigids = self_condition['final_rigids']
            self_cond_edge_features, _ = gen_spatial_graph_features(
                self_cond_rigids,
                seq_edge_index,
                num_rbf_embed=self.c_z//2,
                num_pos_embed=0,
                use_unit_vec=self.sc_edge_unit_vecs,
            )
            edge_features = torch.cat(
                [edge_features, self_cond_edge_features],
                dim=-1
            )
        elif self.self_conditioning:
            edge_features = F.pad(
                edge_features,
                (0, self.c_z//2 + 4 + 3 * self.sc_edge_unit_vecs)
            )
        return edge_features


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

        ## create time embedding
        fourier_time = self.time_rbf(ts.unsqueeze(-1))  # (N_graph x h_time,)
        ## create vns
        if self.vn_mode is not None:
            vn_features = self.embed_vn(fourier_time).view(-1, self.num_vn, self.c_s)
        else:
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
            if not self.sc_learned_features:
                self_cond_nodes = torch.zeros_like(self_cond_nodes)

            trans_rel = self_cond_rigids.get_trans() - rigids_t.get_trans()
            rigids_t_quat = rigids_t.get_rots().get_quats()
            self_cond_quat = self_cond_rigids.get_rots().get_quats()
            quat_rel = ru.quat_multiply(
                ru.invert_quat(rigids_t_quat),
                self_cond_quat
            )

            if self.rbf_encode_sc_trans:
                assert self.sc_t7
                eps = 1e-8
                trans_unit_vec = F.normalize(trans_rel + eps)
                trans_dist = torch.linalg.vector_norm(trans_rel + eps, dim=-1)
                trans_rbf = _rbf(trans_dist, D_count=self.num_rbf, device=trans_dist.device)

                node_input = torch.cat(
                    [node_input, self_cond_nodes, quat_rel, trans_unit_vec, trans_rbf],
                    dim=-1
                )
            elif self.ipmp_self_condition:
                t7_rel = torch.cat([quat_rel, trans_rel], dim=-1)
                if not self.sc_t7:
                    t7_rel = torch.zeros_like(t7_rel)

                sc_bb = self_condition['denoised_bb'][:, :4]
                sc_embed_node_features = self.self_condition_embed(
                    sc_bb,
                    self_cond_rigids,
                    res_mask,
                    batch)

                node_input = torch.cat(
                    [node_input, self_cond_nodes, t7_rel, sc_embed_node_features],
                    dim=-1
                )
            else:
                t7_rel = torch.cat([quat_rel, trans_rel], dim=-1)
                if not self.sc_t7:
                    t7_rel = torch.zeros_like(t7_rel)

                node_input = torch.cat(
                    [node_input, self_cond_nodes, t7_rel],
                    dim=-1
                )

            if self.use_plddt:
                node_input = torch.cat([node_input, self_condition['plddt_bins']], dim=-1)

        elif self.self_conditioning:
            node_input = F.pad(
                node_input,
                (0, self.node_in - node_input.shape[-1])
            )

        _, anchor_edge_index = gen_anchor_graphs(res_data['data_lens'], nodes_per_anchor=10, device=node_input.device)

        return node_input, vn_features, seq_local_edge_index, anchor_edge_index


    def forward(self, data, self_condition=None):
        res_data = data['residue']
        res_mask = (res_data['res_mask']).bool()
        print(data.num_graphs, res_data.num_nodes)

        rigids_t = ru.Rigid.from_tensor_7(res_data['rigids_t'])
        # center the training example at the mean of the x_cas
        center = ru.batchwise_center(rigids_t, res_data.batch, res_data['res_mask'].bool())
        rigids_t = rigids_t.translate(-center)

        (
            node_input,
            vn_features,
            seq_edge_index,
            anchor_edge_index
        ) = self._gen_inital_features(data, self_condition=self_condition)

        # embed features
        node_features = self.embed_node(node_input)
        node_features = node_features * res_mask[..., None]
        seq_edge_features = torch.zeros((seq_edge_index.shape[-1], self.c_z), device=node_features.device)
        anchor_edge_features = torch.zeros((anchor_edge_index.shape[-1], self.c_z), device=node_features.device)

        ## denoising
        rigids_t = rigids_t.scale_translation(0.1)
        rigids = rigids_t

        rigids_history = []
        old_edge_features = None
        old_edge_index = None
        for i, layer in enumerate(self.denoiser):
            # recompute graph
            raw_edge_features, edge_index, knn_edge_select, lrange_edge_select = self._gen_spatial_edge_features(
                rigids.scale_translation(10),
                res_mask,
                res_data.batch,
                self_condition,
                old_edge_index=old_edge_index)
            new_seq_edge_inputs = self._gen_seq_edge_features(
                rigids.scale_translation(10),
                seq_edge_index,
                self_condition)
            new_anchor_edge_inputs = self._gen_seq_edge_features(
                rigids.scale_translation(10),
                anchor_edge_index,
                self_condition)

            node_features, rigids, vn_features, edge_features, seq_edge_features, anchor_edge_features = layer(
                node_features,
                rigids,
                vn_features,
                old_edge_features,
                old_edge_index,
                raw_edge_features,
                edge_index,
                knn_edge_select,
                lrange_edge_select,
                new_seq_edge_inputs,
                seq_edge_features,
                seq_edge_index,
                new_anchor_edge_inputs,
                anchor_edge_features,
                anchor_edge_index,
                res_data,
            )
            old_edge_features = edge_features#[lrange_edge_select]
            old_edge_index = edge_index#[:, lrange_edge_select]
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

        if self.use_plddt:
            plddt_bin_logits = self.plddt_head(node_features)
            ret['plddt_bins'] = torch.softmax(plddt_bin_logits, dim=-1)
            ret['plddt_logits'] = plddt_bin_logits


        return ret
