import torch
from torch import nn
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.utils import dropout_edge

from ligbinddiff.model.modules.common import GaussianRandomFourierBasis
from ligbinddiff.model.modules.layers.node.mpnn import BilevelIPMP
from ligbinddiff.model.modules.layers.edge.embed import PairwiseAtomicEmbedding
from ligbinddiff.data.datasets.featurize.sidechain import _dihedrals, _ideal_virtual_Cb
from ligbinddiff.data.datasets.featurize.common import _edge_positional_embeddings, _rbf

from ligbinddiff.utils.openfold import rigid_utils as ru
from ligbinddiff.utils.openfold.feats import atom14_to_atom37
from ligbinddiff.utils.framediff.all_atom import compute_atom14, torsion_angles_to_frames, frames_to_atom14_pos, prot_to_torsion_angles, adjust_oxygen_pos
from ligbinddiff.utils.atom_reps import restype_1to3, atom91_start_end
from ligbinddiff.data.openfold.residue_constants import restypes
from ligbinddiff.data.openfold.data_transforms import atom37_to_frames
from ligbinddiff.data.openfold.data_transforms import make_atom14_masks



class BilevelIPMPEncoder(nn.Module):
    def __init__(self,
                 c_s_in=6,
                 c_s=256,
                 c_z=128,
                 c_hidden=256,
                 c_out=128,
                 num_rbf=16,
                 num_pos_embed=16,
                 num_layers=4,
                 k=30,
                 num_aa=20,
                 dropout=0.1,
                 classic_mode=False):
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
        self.num_aa = num_aa + 1
        self.c_atomic = (
            + self.num_aa  # seq identity
            + 1  # mask position
        )

        self.embed_node = nn.Sequential(
            nn.Linear(c_s_in + self.c_atomic, 2*c_s),
            nn.ReLU(),
            nn.Linear(2*c_s, 2*c_s),
            nn.ReLU(),
            nn.Linear(2*c_s, c_s * 5),
        )
        self.node_ln = nn.LayerNorm(c_s)

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
            BilevelIPMP(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                dropout=dropout,
                no_points=4
            )
            for _ in range(num_layers)
        ])
        self.output_mu = nn.Linear(c_s*5, c_out)
        self.output_logvar = nn.Linear(c_s*5, c_out)

        self.k = k

    @torch.no_grad()
    def _prep_features(self, graph, eps=1e-8):
        res_data = graph['residue']
        res_mask = res_data['res_mask']

        # node features
        X_ca = res_data['x'].float()
        bb = res_data['bb'].float()
        dihedrals = _dihedrals(bb)

        mask = res_data['mlm_mask']
        noising_mask = res_data['noising_mask']

        seq = F.one_hot(res_data['seq'], num_classes=self.num_aa).to(mask.device)
        rigids_dict = atom37_to_frames({
            "aatype": res_data['seq'],
            "all_atom_positions": res_data['atom37'].double(),
            "all_atom_mask": res_data['atom37_mask']
        })
        rigids = rigids_dict["rigidgroups_gt_frames"].float()
        rigids_mask = rigids_dict["rigidgroups_gt_exists"].bool()
        rigids_mask = rigids_mask[:, (0, 4, 5, 6, 7)]

        rigids = ru.Rigid.from_tensor_4x4(rigids)[:, (0, 4, 5, 6, 7)]
        bb_rigids = rigids[:, 0]

        # lys
        fake_torsions = torch.tensor([[[0., 1]]], device=seq.device).expand(bb_rigids.shape[0], 7, -1)
        fake_seq = torch.ones(bb_rigids.shape).long() * 11
        fake_rigids = torsion_angles_to_frames(bb_rigids, fake_torsions, fake_seq)
        fake_rigids = ru.Rigid.cat(
            [bb_rigids[:, None], fake_rigids[:, 4:]],
            dim=-1
        )
        rigids = ru.Rigid.from_tensor_7(
            rigids.to_tensor_7() * noising_mask[..., None, None] + fake_rigids.to_tensor_7() * (~noising_mask)[..., None, None]
        )
        rigids_mask[noising_mask] = True

        seq = F.one_hot(res_data['seq'], num_classes=self.num_aa).to(mask.device)
        masked_seq = seq * mask[..., None]
        node_features = torch.cat(
            [
                dihedrals,
                mask.float()[..., None],
                masked_seq
            ],
        dim=-1)
        node_features = node_features * res_mask[..., None]

        # edge graph
        masked_X_ca = X_ca.clone()
        masked_X_ca[~res_mask] = torch.inf
        edge_index = knn_graph(masked_X_ca, self.k, graph['residue'].batch)

        # edge features
        src = edge_index[1]
        dst = edge_index[0]
        edge_features = self.init_edge_embed(graph, edge_index)
        # technically this shouldn't be necessary but just to be safe
        edge_mask = res_mask[src] & res_mask[dst]
        edge_features = edge_features * edge_mask[..., None]


        return node_features, edge_features, edge_index, rigids, rigids_mask


    def forward(self, graph, eps=1e-8):
        ## prep features
        res_data = graph['residue']
        res_mask = res_data['res_mask']
        noising_mask = res_data['noising_mask']

        node_features, edge_features, edge_index, rigids, rigids_mask = self._prep_features(graph, eps=eps)

        node_features = self.embed_node(node_features)
        node_features = self.node_ln(node_features.view(-1, 5, self.c_s))
        edge_features = self.embed_edge(edge_features)

        for layer in self.update:
            node_features, edge_features, rigids = layer(
                node_features,
                edge_features,
                edge_index,
                rigids,
                res_mask.float(),
                rigid_mask=rigids_mask,
                design_targets=noising_mask
            )

        latent_mu = self.output_mu(node_features.view(node_features.shape[0], -1))
        latent_logvar = self.output_logvar(node_features.view(node_features.shape[0], -1))

        out_dict = {}
        out_dict['latent_mu'] = latent_mu
        out_dict['latent_logvar'] = latent_logvar

        return out_dict


class BilevelIPMPDecoder(nn.Module):
    def __init__(self,
                 c_s=256,
                 c_z=128,
                 c_hidden=256,
                 c_s_in=6,
                 c_latent_in=128,
                 num_rbf=16,
                 num_pos_embed=16,
                 num_layers=4,
                 num_aa=20,
                 h_time=64,
                 k=30,
                 dropout=0.1):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.num_rbf = num_rbf
        self.num_pos_embed = num_pos_embed

        self.embed_time = GaussianRandomFourierBasis(h_time//2)
        # self.c_atomic = (
        #     + self.num_aa  # seq identity
        #     + 1  # mask position
        # )
        self.c_atomic=0

        self.embed_node = nn.Sequential(
            nn.Linear(
                c_s_in + c_latent_in + h_time + self.c_atomic,
                2*c_s
            ),
            nn.ReLU(),
            nn.Linear(2*c_s, 2*c_s),
            nn.ReLU(),
            nn.Linear(2*c_s, c_s * 5),
        )
        self.node_ln = nn.LayerNorm(c_s)

        self.init_edge_embed = PairwiseAtomicEmbedding(
            num_rbf=self.num_rbf,
            num_pos_embed=self.num_pos_embed,
            num_aa=num_aa,
            classic_mode=True
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
            BilevelIPMP(
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                dropout=dropout,
                no_points=4,
                # design_frames_transition="update"
            )
            for _ in range(num_layers)
        ])

        self.seq_head = nn.Linear(c_s * 5, 20)

        self.k = k


    # @torch.no_grad()
    def _prep_features(self, graph, eps=1e-8):
        res_data = graph['residue']
        res_mask = res_data['res_mask']
        noising_mask = res_data['noising_mask']

        # node features
        X_ca = res_data['x'].float()
        bb = res_data['bb'].float()
        dihedrals = _dihedrals(bb)
        node_features = torch.cat([
            dihedrals
        ], dim=-1)
        node_features = node_features * res_mask[..., None]

        rigids_dict = atom37_to_frames({
            "aatype": res_data['seq'],
            "all_atom_positions": res_data['atom37'].double(),
            "all_atom_mask": res_data['atom37_mask']
        })
        rigids = rigids_dict["rigidgroups_gt_frames"].float()
        rigids_mask = rigids_dict["rigidgroups_gt_exists"].bool()
        rigids_mask = rigids_mask[:, (0, 4, 5, 6, 7)]

        rigids = ru.Rigid.from_tensor_4x4(rigids)[:, (0, 4, 5, 6, 7)]
        bb_rigids = rigids[:, 0]

        # lys
        fake_torsions = torch.tensor([[[0., 1]]], device=node_features.device).expand(bb_rigids.shape[0], 7, -1)
        fake_seq = torch.ones(bb_rigids.shape).long() * 11
        fake_rigids = torsion_angles_to_frames(bb_rigids, fake_torsions, fake_seq)
        fake_rigids = ru.Rigid.cat(
            [bb_rigids[:, None], fake_rigids[:, 4:]],
            dim=-1
        )
        rigids = ru.Rigid.from_tensor_7(
            rigids.to_tensor_7() * noising_mask[..., None, None] + fake_rigids.to_tensor_7() * (~noising_mask)[..., None, None]
        )
        rigids_mask[noising_mask] = True

        # edge graph
        masked_X_ca = X_ca.clone()
        masked_X_ca[~res_mask] = torch.inf
        edge_index = knn_graph(masked_X_ca, self.k, graph['residue'].batch)

        # edge features
        virtual_Cb = _ideal_virtual_Cb(bb)
        bb = torch.cat([bb, virtual_Cb[..., None, :]], dim=-2)
        src = edge_index[1]
        dst = edge_index[0]

        # edge features
        src = edge_index[1]
        dst = edge_index[0]
        edge_features = self.init_edge_embed(graph, edge_index)
        # technically this shouldn't be necessary but just to be safe
        edge_mask = res_mask[src] & res_mask[dst]
        edge_features = edge_features * edge_mask[..., None]

        return node_features, edge_features, edge_index, rigids, rigids_mask

    def forward(self,
                graph,
                intermediates,
                t=None,
                eps=1e-8,
                use_gt_seq=True):
        res_data = graph['residue']
        res_mask = res_data['res_mask']
        noising_mask = res_data['noising_mask']

        node_features, edge_features, edge_index, rigids, rigids_mask = self._prep_features(graph, eps=eps)

        if t is None:
            t = torch.zeros(graph.num_graphs, device=edge_features.device)
        timestep_embed = self.embed_time(t[..., None])
        timestep_embed = timestep_embed[res_data.batch]


        node_features = self.embed_node(
            torch.cat([timestep_embed, node_features, intermediates['latent_sidechain']], dim=-1)
        ) * res_mask[..., None]
        node_features = self.node_ln(node_features.view(-1, 5, self.c_s))
        edge_features = self.embed_edge(edge_features)

        for layer in self.update:
            node_features, edge_features, rigids = layer(
                node_features,
                edge_features,
                edge_index,
                rigids,
                res_mask.float(),
                rigid_mask=rigids_mask,
                design_targets=noising_mask,
            )

        seq_logits = self.seq_head(node_features.view(node_features.shape[0], -1))

        seq = seq_logits.argmax(dim=-1)
        seq = seq.to(node_features.device)
        atom14_mask_dict = make_atom14_masks({"aatype": seq})

        # a jenk way to get the backbone frames
        # lys
        fake_seq = torch.ones(node_features.shape[0]).long() * 11
        fake_torsions = torch.tensor([[[0., 1.]]], device=node_features.device).expand(
            node_features.shape[0],
            7,
            -1)
        fake_rigids = torsion_angles_to_frames(
            rigids[:, 0],
            fake_torsions,
            fake_seq
        )

        all_rigids = ru.Rigid.cat([
            fake_rigids[:, :3],
            rigids
        ], dim=-1)

        # chi_per_aatype = chi_per_aatype.view(-1, 4, 2)
        output_atom14 = frames_to_atom14_pos(all_rigids, seq.to("cpu"))
        atom37, atom37_mask = atom14_to_atom37(output_atom14, atom14_mask_dict)
        chi_angles, _ = prot_to_torsion_angles(seq, atom37, atom37_mask)
        # output_atom14 = compute_atom14(rigids[:, 0], chi_angles[:, 2:3], chi_angles[:, 3:], seq)
        atom37 = adjust_oxygen_pos(atom37, atom37_mask[:, 1])
        output_atom14[:, 3, :] = atom37[:, 4, :]


        out_dict = {}
        # out_dict['decoded_all_atom14'] = all_atom14
        # out_dict['decoded_all_chis'] = chi_per_aatype
        out_dict['decoded_atom14'] = output_atom14
        out_dict['decoded_atom14_mask'] = atom14_mask_dict['atom14_atom_exists']
        out_dict['decoded_chis'] = chi_angles[:, 3:, :]
        out_dict['decoded_seq_logits'] = seq_logits

        if use_gt_seq:
            gt_seq = res_data['seq']
            output_atom14 = frames_to_atom14_pos(all_rigids, gt_seq.to("cpu"))
            atom37, atom37_mask = atom14_to_atom37(output_atom14, atom14_mask_dict)
            chi_angles, _ = prot_to_torsion_angles(gt_seq, atom37, atom37_mask)
            # output_atom14 = compute_atom14(rigids[:, 0], chi_angles[:, 2:3], chi_angles[:, 3:], seq)
            atom37 = adjust_oxygen_pos(atom37, atom37_mask[:, 1])
            output_atom14[:, 3, :] = atom37[:, 4, :]

            out_dict['decoded_atom14_gt_seq'] = output_atom14
            out_dict['decoded_chis_gt_seq'] = chi_angles[:, 3:, :]

        return out_dict
