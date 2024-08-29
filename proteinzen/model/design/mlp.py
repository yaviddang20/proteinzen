import torch
from torch import nn
import torch.nn.functional as F

from proteinzen.model.modules.common import GaussianRandomFourierBasis

from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.utils.framediff.all_atom import compute_all_atom14, compute_atom14
from proteinzen.data.openfold.data_transforms import make_atom14_masks


class MLPDecoder(nn.Module):
    def __init__(self,
                 c_s=128,
                 c_latent=128,
                 h_time=64):
        super().__init__()

        self.embed_time = GaussianRandomFourierBasis(h_time)
        self.embed_node = nn.Sequential(
            nn.Linear(c_latent + h_time*2, 2*c_s),
            nn.SiLU(),# nn.ReLU(),
            nn.Linear(2*c_s, c_s),
            nn.SiLU(), #nn.ReLU()
        )
        self.seq_head = nn.Linear(c_s, 20)
        self.seq_embed = nn.Embedding(21, 20)
        self.torsion_pred = nn.Linear(c_s + 20, (4 + 1) * 2)

    def forward(self,
                graph,
                intermediates,
                t=None,
                eps=1e-8,
                use_gt_seq=True):
        res_data = graph['residue']
        res_mask = res_data['res_mask']

        node_features = intermediates['latent_sidechain']
        if t is None:
            if "rigids_1" in res_data:
                t = torch.ones(graph.num_graphs, device=node_features.device)
            else:
                t = torch.zeros(graph.num_graphs, device=node_features.device)

        timestep_embed = self.embed_time(t[..., None])
        timestep_embed = timestep_embed[res_data.batch]

        node_features = self.embed_node(
            torch.cat([timestep_embed, node_features], dim=-1)
        )
        # node_features = self.node_ln(node_features + node_update * res_mask[..., None])

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

        # TODO: fix this inconsistency
        if "rigids_1" in res_data:
            rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_1)
        else:
            rigids = ru.Rigid.from_tensor_7(graph['residue'].rigids_0)
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
