import torch
from torch import nn

from ligbinddiff.model.modules.equiformer_v2.transformer_block import FeedForwardNetwork
from ligbinddiff.model.modules.equiformer_v2.so3 import CoefficientMappingModule, SO3_Embedding, SO3_Rotation, SO3_Grid, SO3_LinearV2
from ligbinddiff.model.modules.equiformer_v2.layer_norm import EquivariantRMSNormArraySphericalHarmonicsV2 as NormSO3


class Atom91Discriminator(nn.Module):
    def __init__(self,
                 atom_lmax_list=[1],
                 atom_channels=91,
                 out_dim=2,
                 num_layers=3):
        super().__init__()

        atom_SO3_grid_list = nn.ModuleList()
        for l in range(max(atom_lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(atom_lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            atom_SO3_grid_list.append(SO3_m_grid)


        self.atom_lmax_list = atom_lmax_list
        self.atom_channels = atom_channels

        self.ffs = nn.ModuleList(
            [
                FeedForwardNetwork(
                    atom_channels,
                    atom_channels * 2,
                    atom_channels,
                    lmax_list=atom_lmax_list,
                    mmax_list=atom_lmax_list,
                    SO3_grid=atom_SO3_grid_list
                )
                for _ in range(num_layers)
            ]
        )

        self.norms = nn.ModuleList(
            [
                NormSO3(
                    lmax=max(atom_lmax_list),
                    num_channels=atom_channels
                )
                for _ in range(num_layers)
            ]
        )

        self.out = nn.Linear(atom_channels, out_dim)

    def forward(self, graph):
        num_nodes = graph['x'].shape[0]
        atom_features = SO3_Embedding(
            num_nodes * 2,
            lmax_list=self.atom_lmax_list,
            num_channels=self.atom_channels,
            device=graph['x'].device,
            dtype=torch.float
        )

        decoded_atom91 = graph['decoded_latent'].clone()
        atom91_mask = graph['atom91_mask']
        decoded_atom91[atom91_mask] = 0
        atom_features.embedding[:num_nodes, 1:4] = decoded_atom91.transpose(-1, -2)

        real_atom91 = graph['atom91_centered'].clone()
        atom91_mask = graph['atom91_mask']
        real_atom91[atom91_mask] = 0
        atom_features.embedding[num_nodes:, 1:4] = real_atom91.transpose(-1, -2)

        hidden = atom_features
        for ff, norm in zip(self.ffs, self.norms):
            hidden_update = norm(ff(hidden).embedding)
            hidden.embedding = hidden.embedding + hidden_update

        hidden_scalars = hidden.get_invariant_features().squeeze(1)
        prelogits = self.out(hidden_scalars)
        logits = torch.log_softmax(prelogits, dim=-1)

        logits_fake = logits[:num_nodes]
        logits_real = logits[num_nodes:]

        graph['discrim_logits_real'] = logits_real
        graph['discrim_logits_fake'] = logits_fake
        return graph
