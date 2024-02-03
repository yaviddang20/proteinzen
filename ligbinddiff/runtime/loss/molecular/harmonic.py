import torch

from ligbinddiff.runtime.loss.utils import _nodewise_to_graphwise


def harmonic_fm_loss(batch,
                     denoiser_outputs,
                     t_norm_clip=0.9,
                     eps=1e-8):
    atom_data = batch['ligand']
    atom_pos = atom_data['atom_pos']
    noised_atom_pos = atom_data['noised_atom_pos']
    atom_mask = atom_data['atom_mask']
    pred_atom_pos = denoiser_outputs['pred_atom_pos']

    traj_loss = []
    for interm_atom_pos in denoiser_outputs['atom_traj']:
        pos_mse = torch.square(atom_pos - interm_atom_pos).sum(dim=-1)
        pos_mse = _nodewise_to_graphwise(pos_mse, atom_data.batch, atom_mask)
        traj_loss.append(pos_mse)
    final_pos_mse = traj_loss[-1]

    ref_pos_mse = torch.square(atom_pos - noised_atom_pos).sum(dim=-1)
    ref_pos_mse = _nodewise_to_graphwise(ref_pos_mse, atom_data.batch, atom_mask)

    t = batch['t']
    norm_scale = 1 - torch.min(
        t, torch.as_tensor(t_norm_clip)
    )

    return {
        "atom_pos_mse": final_pos_mse,
        "atom_pos_traj_mse": torch.stack(traj_loss).sum(dim=0),
        "atom_ref_pos_mse": ref_pos_mse,
        "fm_norm_scale": norm_scale ** 2
    }
