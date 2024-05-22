import torch

from ligbinddiff.runtime.loss.utils import _nodewise_to_graphwise


def torsional_fm_loss(batch,
                     denoiser_outputs,
                     t_norm_clip=0.9,
                     loss_clip=10,
                     eps=1e-8):
    atom_data = batch['ligand']
    bond_data = batch['ligand', 'ligand']
    rotatable_bonds = bond_data['rotatable_bonds']
    atom_pos = atom_data['atom_pos']
    noised_atom_pos = atom_data['noised_atom_pos']
    atom_mask = atom_data['atom_mask']
    pred_atom_pos = denoiser_outputs['pred_atom_pos']
    pred_torsion_update = denoiser_outputs['pred_torsion_update']
    gt_torsion_noise = bond_data['gt_torsion_noise']
    edge_batch = atom_data.batch[bond_data.edge_index[0, rotatable_bonds]]

    traj_loss = []
    for interm_atom_pos in denoiser_outputs['atom_traj']:
        pos_mse = torch.square(atom_pos - interm_atom_pos).sum(dim=-1)
        pos_mse = _nodewise_to_graphwise(pos_mse, atom_data.batch, atom_mask)
        traj_loss.append(pos_mse)

    ref_pos_mse = torch.square(atom_pos - noised_atom_pos).sum(dim=-1)
    ref_pos_mse = _nodewise_to_graphwise(ref_pos_mse, atom_data.batch, atom_mask)

    t = batch['t']
    norm_scale = 1 - torch.min(
        t, torch.as_tensor(t_norm_clip)
    )

    traj_loss = torch.stack(traj_loss)
    if loss_clip is not None:
        traj_loss = traj_loss.clip(max=loss_clip)
    final_pos_mse = traj_loss[-1]

    dummy_mask = torch.ones_like(edge_batch).bool()
    torsion_gt_noise = _nodewise_to_graphwise(gt_torsion_noise, edge_batch, dummy_mask)
    torsion_pred_noise = _nodewise_to_graphwise(-pred_torsion_update, edge_batch, dummy_mask)

    torsion_flow_loss = torch.square(gt_torsion_noise + pred_torsion_update) / torch.square(norm_scale)
    torsion_flow_loss = _nodewise_to_graphwise(
        torsion_flow_loss,
        edge_batch,
        dummy_mask
    )

    return {
        "atom_pos_mse": final_pos_mse,
        "atom_pos_traj_mse": traj_loss.mean(dim=0),
        "atom_ref_pos_mse": ref_pos_mse,
        "torsion_ref_noise": gt_torsion_noise,
        "pred_torsion_update": pred_torsion_update,
        "torsion_fm_loss": torsion_flow_loss,
        "fm_norm_scale": 1 / norm_scale ** 2
    }
