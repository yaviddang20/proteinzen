import torch
from torch import nn
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.utils import dropout_edge

from proteinzen.model.modules.common import GaussianRandomFourierBasis, FeedForward
from proteinzen.model.modules.openfold.layers import  StructureModuleTransition, InvariantPointAttention
from proteinzen.model.modules.layers.edge.sitewise import DenseEdgeTransition as EdgeTransition
from proteinzen.data.datasets.featurize.sidechain import _dihedrals, _ideal_virtual_Cb
from proteinzen.data.datasets.featurize.common import _edge_positional_embeddings, _rbf, _node_positional_embeddings

from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.utils.framediff.all_atom import compute_all_atom14, compute_atom14
from proteinzen.data.openfold.data_transforms import make_atom14_masks


class IPAUpdateLayer(nn.Module):
    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden=16,
                 num_qk_points=8,
                 num_v_points=12,
                 num_heads=16,
                 dropout=0.,
                 pre_ln=False,
                 lin_bias=True,
        ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden

        self.ipa = InvariantPointAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            num_heads=num_heads,
            pre_ln=pre_ln,
            lin_bias=lin_bias
        )
        self.ln = nn.LayerNorm(c_s)
        self.dropout = nn.Dropout(p=dropout)
        self.node_transition = StructureModuleTransition(
            c_s,
            dropout=dropout
        )
        self.edge_transition = EdgeTransition(
            node_embed_size=c_s,
            edge_embed_in=c_z,
            edge_embed_out=c_z,
        )

    def forward(self,
                node_features,
                edge_features,
                rigids,
                node_mask):
        node_update = self.ipa(
            s=node_features,
            z=edge_features,
            r=rigids,
            mask=node_mask)
        node_features = self.ln(node_features + self.dropout(node_update) * node_mask[..., None])
        node_features = self.node_transition(node_features) * node_mask[..., None]
        edge_features = self.edge_transition(node_features, edge_features)
        return node_features, edge_features


class IPADecoder(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_s_in=32,
                 c_z_in=32,
                 c_hidden=16,
                 num_qk_points=8,
                 num_v_points=12,
                 num_heads=16,
                 dropout=0.,
                 pre_ln=False,
                 lin_bias=True,
                 num_pos_embed=16,
                 num_layers=4,
                 h_time=64,
                 k=30
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_in = c_s_in
        self.c_z_in = c_z_in
        self.c_hidden = c_hidden
        self.num_pos_embed = num_pos_embed
        self.pre_ln = pre_ln
        self.lin_bias = lin_bias

        self.embed_time = GaussianRandomFourierBasis(h_time)

        self.embed_node = nn.Sequential(
            nn.Linear(c_s_in + h_time*2, 2*c_s),
            nn.ReLU(),
            nn.Linear(2*c_s, c_s),
        )
        self.node_ln = nn.LayerNorm(c_s)

        self.embed_edge = nn.Sequential(
            nn.Linear(c_z_in + num_pos_embed, 2*c_z),
            nn.ReLU(),
            nn.Linear(2*c_z, 2*c_z),
            nn.ReLU(),
            nn.Linear(2*c_z, c_z),
            nn.LayerNorm(c_z),
        )
        self.update = nn.ModuleList([
            IPAUpdateLayer(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                num_heads=num_heads,
                dropout=dropout,
                pre_ln=pre_ln,
                lin_bias=lin_bias,
            )
            for _ in range(num_layers)
        ])

        if pre_ln:
            self.ln_s = nn.LayerNorm(c_s)


        self.seq_head = nn.Linear(c_s, 20)
        self.seq_embed = nn.Embedding(21, 20)
        self.torsion_pred = nn.Linear(c_s + 20, (4 + 1) * 2)

        self.k = k


    def forward(self,
                graph,
                intermediates,
                t=None,
                eps=1e-8,
                use_gt_seq=True):
        res_data = graph['residue']
        n_batch = graph.num_graphs
        # TODO: fix this inconsistency
        if "rigids_1" in res_data:
            tensor_7 = graph['residue'].rigids_1
        else:
            tensor_7 = graph['residue'].rigids_0
        tensor_7 = tensor_7.view(n_batch, -1, 7)
        rigids = ru.Rigid.from_tensor_7(tensor_7)
        res_mask = res_data['res_mask'].view(n_batch, -1)

        if t is None:
            if "rigids_1" in res_data:
                t = torch.ones(graph.num_graphs, device=tensor_7.device)
            else:
                t = torch.zeros(graph.num_graphs, device=tensor_7.device)

        node_features = intermediates['latent_sidechain']

        timestep_embed = self.embed_time(t[..., None])
        timestep_embed = torch.tile(timestep_embed[:, None], (1, node_features.shape[1], 1))

        node_update = self.embed_node(
            torch.cat([timestep_embed, node_features], dim=-1)
        )
        # node_features = self.node_ln(node_features + node_update * res_mask[..., None])
        node_features = self.node_ln(node_update * res_mask[..., None])

        node_pos = torch.arange(rigids.shape[-1], device=node_features.device)
        edge_rel_pos = node_pos[..., None] - node_pos[None]
        edge_rel_pos_embed = _node_positional_embeddings(edge_rel_pos, num_embeddings=self.num_pos_embed, device=node_pos.device)

        edge_features = torch.cat([
            intermediates['latent_edge'],
            torch.tile(edge_rel_pos_embed[None], (n_batch, 1, 1, 1))
        ], dim=-1)

        edge_features = self.embed_edge(edge_features)

        res_mask = res_mask.view(node_features.shape[:2])

        for layer in self.update:
            node_features, edge_features = layer(
                node_features,
                edge_features,
                rigids,
                res_mask.float()
            )

        if self.pre_ln:
            node_features = self.ln_s(node_features)

        node_features = node_features.view(-1, node_features.shape[-1])
        rigids = rigids.view(-1)

        seq_logits = self.seq_head(node_features)

        seq = seq_logits.argmax(dim=-1)
        seq = seq.to(node_features.device)
        atom14_mask_dict = make_atom14_masks({"aatype": seq})

        unnorm_torsions = self.torsion_pred(
            torch.cat([
                node_features,
                self.seq_embed(seq)
            ], dim=-1)
        )
        chi_per_aatype = unnorm_torsions.view(-1, 5, 2)
        chi_per_aatype = F.normalize(chi_per_aatype, dim=-1)
        psi_torsions, chi_per_aatype = chi_per_aatype.split([1, 4], dim=-2)

        # chi_per_aatype = chi_per_aatype.view(-1, 4, 2)
        output_atom14 = compute_atom14(rigids, psi_torsions, chi_per_aatype, seq)


        out_dict = {}
        # out_dict['decoded_all_atom14'] = all_atom14
        # out_dict['decoded_all_chis'] = chi_per_aatype
        out_dict['decoded_atom14'] = output_atom14
        out_dict['decoded_atom14_mask'] = atom14_mask_dict['atom14_atom_exists']
        out_dict['decoded_chis'] = chi_per_aatype
        out_dict['decoded_seq_logits'] = seq_logits

        if use_gt_seq:
            gt_seq = res_data['seq']
            unnorm_torsions = self.torsion_pred(
                torch.cat([
                    node_features,
                    self.seq_embed(gt_seq)
                ], dim=-1)
            )
            chi_per_aatype = unnorm_torsions.view(-1, 5, 2)
            chi_per_aatype = F.normalize(chi_per_aatype, dim=-1)
            psi_torsions, chi_per_aatype = chi_per_aatype.split([1, 4], dim=-2)

            # chi_per_aatype = chi_per_aatype.view(-1, 4, 2)
            output_atom14 = compute_atom14(rigids, psi_torsions, chi_per_aatype, gt_seq)
            out_dict['decoded_atom14_gt_seq'] = output_atom14
            out_dict['decoded_chis_gt_seq'] = chi_per_aatype

        return out_dict
