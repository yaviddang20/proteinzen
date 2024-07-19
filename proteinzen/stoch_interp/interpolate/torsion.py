import numpy as np
import tqdm
import torch

from proteinzen.utils.torsion import modify_conformer_torsion_angles_batch, modify_conformer_torsion_angles_batch_cached

from torch_geometric.data import HeteroData, Batch
from torch_geometric.utils import scatter
import dataclasses


def sample_torsion_prior(atom_pos, rotatable_bonds, mask_rotate, edge_index,  node_batch):
    tor_0 = torch.rand_like(rotatable_bonds.float()) * torch.pi * 2 - torch.pi
    return modify_conformer_torsion_angles_batch(
        atom_pos,
        edge_index[:, rotatable_bonds],
        node_batch,
        mask_rotate,
        tor_0[rotatable_bonds])


class TorsionInterpolant:
    def __init__(self, min_t=0.01):
        self.min_t = min_t

    def sample_t(self, num_batch, device):
        t = torch.rand(num_batch, device=device)
        return t * (1 - 2 * self.min_t) + self.min_t

    def _corrupt_tors(self, atom_pos, tor_0, t, update_instructs):
        # NOTE: we're implicitly treating tor_0 as (tor_1 + uniform circle noise)
        # which should be equiv because tor_0 is drawn from uniform
        # this lets us avoid computing the ground truth torsions which is nice
        tor_t = (1-t) * tor_0
        new_atom_pos = modify_conformer_torsion_angles_batch_cached(
            atom_pos,
            tor_t,
            update_instructs)
        return new_atom_pos

    def sample_tor_noise(self, num_tors, device):
        # [-pi, pi) to match with _corrupt_rots
        tor_0 = torch.rand(num_tors, device=device) * torch.pi * 2 - torch.pi
        tor_0[:-1] *= 0
        tor_0[-1] *= 1/3
        # tor_0 = torch.ones(num_tors, device=device) * torch.pi * 2 - torch.pi
        return tor_0

    @torch.no_grad()
    def corrupt_batch(self, batch: HeteroData):
        lig_data = batch["ligand"]
        bond_data = batch["ligand", "ligand"]
        atom_pos = lig_data['atom_pos']
        rotatable_bonds = bond_data['rotatable_bonds']
        edge_batch = lig_data.batch[bond_data.edge_index[0]]
        edge_batch = edge_batch[rotatable_bonds]
        update_instructs = batch["conformer_update_instructions"]

        # [B]
        t = self.sample_t(batch.num_graphs, atom_pos.device)
        batch['t'] = t
        edgewise_t = t[edge_batch]
        edge_index = bond_data.edge_index[:, rotatable_bonds]
        tor_0 = self.sample_tor_noise(edge_index.shape[-1], device=atom_pos.device)
        atom_pos_t = self._corrupt_tors(
            atom_pos,
            tor_0,
            edgewise_t,
            update_instructs)
        lig_data['noised_atom_pos'] = atom_pos_t
        bond_data['gt_torsion_noise'] = tor_0 * (1 - edgewise_t)

        return batch

    def euler_step(self, d_t, t, atom_pos_t, tor_noise_pred, rotatable_bonds, mask_rotate, edge_index, node_batch, anneal=None):
        psuedotorque = tor_noise_pred / (1-t)
        if anneal is not None:
            psuedotorque = psuedotorque * (1 - np.exp(-anneal * t))
        tor_update = psuedotorque * d_t #* rotatable_bonds
        new_atom_pos = modify_conformer_torsion_angles_batch(
            atom_pos_t,
            edge_index[:, rotatable_bonds],
            node_batch,
            mask_rotate,
            tor_update#[rotatable_bonds],
        )
        return new_atom_pos


