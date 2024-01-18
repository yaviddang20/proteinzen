import torch

from ligbinddiff.runtime.loss.utils import _nodewise_to_graphwise


def harmonic_fm_loss(batch,
                     denoiser_outputs,
                     t_norm_clip=0.9):
    atom_data = batch['ligand']
    atom_pos = atom_data['atom_pos']
    atom_mask = atom_data['atom_mask']
    pred_atom_pos = denoiser_outputs['pred_atom_pos']

    pos_mse = torch.square(atom_pos - pred_atom_pos).sum(dim=-1)
    pos_mse = _nodewise_to_graphwise(pos_mse, atom_data.batch, atom_mask)

    t = batch['t']
    norm_scale = 1 - torch.min(
        t, torch.as_tensor(t_norm_clip)
    )

    return {
        "atom_pos_mse": pos_mse,
        "fm_norm_scale": norm_scale ** 2
    }
