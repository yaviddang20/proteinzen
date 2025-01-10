import torch
import numpy as np
import torch.nn.functional as F
import torch_geometric.utils as pygu
from torch_geometric.nn import radius_graph

from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.model.utils.graph import batchwise_to_nodewise, get_data_lens
from proteinzen.stoch_interp.interpolate import so3_utils as so3_fm_utils

from .utils import _nodewise_to_graphwise
from .frames import angle_axis_rot_vf_loss


def interm_rigid_loss(
        noised_rigids,
        gt_rigids,
        pred_rigids,
        res_batch,
        total_mask
):
    X_ca = gt_rigids.get_trans()
    pred_frame_X_ca = pred_rigids.get_trans()
    pred_X_ca_se = torch.square(X_ca - pred_frame_X_ca).sum(dim=-1)
    trans_loss = _nodewise_to_graphwise(pred_X_ca_se, res_batch, total_mask)

    rots_t = noised_rigids.get_rots().get_rot_mats()
    rots_1_pred = pred_rigids.get_rots().get_rot_mats()
    rots_1 = gt_rigids.get_rots().get_rot_mats()
    pred_rot_vf = so3_fm_utils.calc_rot_vf(rots_t, rots_1_pred)
    gt_rot_vf = so3_fm_utils.calc_rot_vf(rots_t, rots_1)
    rot_loss = angle_axis_rot_vf_loss(
        pred_rot_vf,
        gt_rot_vf,
        res_batch,
        total_mask,
        norm_scale=torch.ones(
            (res_batch.max().item()+1,),
            device=res_batch.device)
        )
    return 0.01 * trans_loss + rot_loss


def interm_pred_dist_loss(
    gt_dist,
    dist_logits,
    edge_mask,
    D_min=2,
    D_max=22,
    num_bins=64
):
    lower = torch.linspace(
        D_min,
        D_max,
        num_bins,
        device=gt_dist.device)
    dgram_labels = (gt_dist[..., None] > lower).type(torch.int32)
    dgram_labels = torch.clip(dgram_labels.sum(dim=-1) - 1, min=0)
    dist_logits = dist_logits.permute((0, 3, 1, 2))
    cce = F.cross_entropy(dist_logits, dgram_labels, reduction="none")
    return (cce * edge_mask).sum(dim=(-1, -2)) / edge_mask.float().sum(dim=(-1, -2))


def traj_loss(
    batch,
    model_outputs,
    traj_decay_factor=0.99,
):
    traj_data = model_outputs['traj_data']
    n_layers = len(traj_data) + 1

    res_data = batch['residue']
    device = res_data['atom14'].device

    if isinstance(res_data['rigids_t'], torch.Tensor):
        rigids_t = ru.Rigid.from_tensor_7(res_data['rigids_t'].to(device))
    else:
        rigids_t = res_data['rigids_t']
        dtype = rigids_t._trans.dtype
        rigids_t = ru.Rigid(
            rots=rigids_t._rots.to(device=device, dtype=dtype),
            trans=rigids_t._trans.to(device)
        )
    rigids_1 = ru.Rigid.from_tensor_7(res_data['rigids_1'].to(device))

    res_mask = res_data['res_mask'].view(batch.num_graphs, -1)
    edge_mask = res_mask[..., None] & res_mask[..., None, :]
    gt_trans = rigids_1.get_trans().reshape(batch.num_graphs, -1, 3)
    gt_dists = torch.cdist(gt_trans, gt_trans)

    bb_loss = 0
    pred_dist_loss = 0
    seq_loss = 0
    for k in range(n_layers-1):
        data_k = traj_data[k]
        scale = traj_decay_factor ** (n_layers - k)
        bb_loss_k = interm_rigid_loss(
            noised_rigids=rigids_t.view(-1),
            gt_rigids=rigids_1.view(-1),
            pred_rigids=data_k['rigids'].view(-1),
            res_batch=res_data.batch,
            total_mask=res_data['res_mask']
        )
        dist_loss_k = interm_pred_dist_loss(
            gt_dist=gt_dists,
            dist_logits=data_k['dist_logits'],
            edge_mask=edge_mask
        )
        seq_loss_k = F.cross_entropy(
            data_k['seq_logits'].transpose(-1, -2),
            res_data['seq'].view(batch.num_graphs, -1),
            reduction="none"
        )
        seq_loss_k = (seq_loss_k * res_mask).sum(dim=-1) / res_mask.float().sum(dim=-1)

        bb_loss += bb_loss_k * scale
        pred_dist_loss += dist_loss_k * scale
        seq_loss += seq_loss_k * scale

    return {
        "traj_bb_loss": bb_loss / (n_layers-1),
        "traj_pred_dist_loss": pred_dist_loss / (n_layers-1),
        "traj_seq_loss": seq_loss / (n_layers-1)
    }

def traj_loss2(
    batch,
    model_outputs,
    traj_decay_factor=0.99,
):
    traj_data = model_outputs['traj_data']
    n_layers = len(traj_data)

    res_data = batch['residue']
    device = res_data['atom14'].device

    if isinstance(res_data['rigids_t'], torch.Tensor):
        rigids_t = ru.Rigid.from_tensor_7(res_data['rigids_t'].to(device))
    else:
        rigids_t = res_data['rigids_t']
        dtype = rigids_t._trans.dtype
        rigids_t = ru.Rigid(
            rots=rigids_t._rots.to(device=device, dtype=dtype),
            trans=rigids_t._trans.to(device)
        )
    rigids_1 = ru.Rigid.from_tensor_7(res_data['rigids_1'].to(device))

    res_mask = res_data['res_mask'].view(batch.num_graphs, -1)
    edge_mask = res_mask[..., None] & res_mask[..., None, :]
    gt_trans = rigids_1.get_trans().reshape(batch.num_graphs, -1, 3)
    gt_dists = torch.cdist(gt_trans, gt_trans)

    bb_loss = 0
    pred_dist_loss = 0
    seq_loss = 0
    for k in range(n_layers):
        data_k = traj_data[k]
        scale = traj_decay_factor ** (n_layers - k)
        bb_loss_k = interm_rigid_loss(
            noised_rigids=rigids_t.view(-1),
            gt_rigids=rigids_1.view(-1),
            pred_rigids=data_k['rigids'].view(-1),
            res_batch=res_data.batch,
            total_mask=res_data['res_mask']
        )
        dist_loss_k = interm_pred_dist_loss(
            gt_dist=gt_dists,
            dist_logits=data_k['dist_logits'],
            edge_mask=edge_mask
        )
        seq_loss_k = F.cross_entropy(
            data_k['seq_logits'].transpose(-1, -2),
            res_data['seq'].view(batch.num_graphs, -1),
            reduction="none"
        )
        seq_loss_k = (seq_loss_k * res_mask).sum(dim=-1) / res_mask.float().sum(dim=-1)

        bb_loss += bb_loss_k * scale
        pred_dist_loss += dist_loss_k * scale
        seq_loss += seq_loss_k * scale

    return {
        "traj_bb_loss": bb_loss / n_layers,
        "traj_pred_dist_loss": pred_dist_loss / n_layers,
        "traj_seq_loss": seq_loss / n_layers-1
    }

def interm_multirigid_loss(
        noised_rigids,
        gt_rigids,
        pred_rigids,
        res_batch,
        total_mask
):
    X_ca = gt_rigids.get_trans()
    pred_frame_X_ca = pred_rigids.get_trans()
    pred_X_ca_se = torch.square(X_ca - pred_frame_X_ca).sum(dim=-1)
    trans_loss = _nodewise_to_graphwise(pred_X_ca_se, res_batch, total_mask)

    rots_t = noised_rigids.get_rots().get_rot_mats()
    rots_1_pred = pred_rigids.get_rots().get_rot_mats()
    rots_1 = gt_rigids.get_rots().get_rot_mats()
    pred_rot_vf = so3_fm_utils.calc_rot_vf(rots_t, rots_1_pred)
    gt_rot_vf = so3_fm_utils.calc_rot_vf(rots_t, rots_1)
    rot_loss = angle_axis_rot_vf_loss(
        pred_rot_vf,
        gt_rot_vf,
        res_batch,
        total_mask,
        norm_scale=torch.ones(
            (res_batch.max().item()+1, 1),
            device=res_batch.device)
        )
    return 0.01 * trans_loss + rot_loss


def interm_pred_framepair_dist_loss(
    gt_rigids,
    framepair_preds_dict,
    rigids_mask,
    D_min=0,
    D_max=10,
    num_bins=22
):
    lower = torch.linspace(
        D_min,
        D_max,
        num_bins,
        device=rigids_mask.device)

    to_queries = framepair_preds_dict['to_queries']
    to_keys = framepair_preds_dict['to_keys']
    n_padding = framepair_preds_dict['n_padding']
    # n_batch x n_block x block_q x block_k x n_bins
    dist_logits = framepair_preds_dict['logits']

    n_batch = gt_rigids.shape[0]
    gt_rigids = gt_rigids.view(n_batch, -1)
    gt_trans = gt_rigids.get_trans()
    gt_trans = F.pad(gt_trans, (0, 0, 0, n_padding))
    gt_q_trans = to_queries(gt_trans)
    gt_k_trans = to_keys(gt_trans)
    gt_dist = torch.linalg.vector_norm(gt_q_trans[..., None, :] - gt_k_trans[..., None, :, :], dim=-1)
    mask_padded = F.pad(rigids_mask.view(n_batch, -1).float(), (0, n_padding))[..., None]
    q_mask = to_queries(mask_padded)
    k_mask = to_keys(mask_padded)
    edge_mask = q_mask[..., None, :] * k_mask[..., None, :, :]

    # n_batch x n_block x block_q x block_k
    dgram_labels = (gt_dist[..., None] > lower).long()
    dgram_labels = torch.clip(dgram_labels.sum(dim=-1) - 1, min=0)
    # n_batch x n_bins x n_block x block_q x block_k
    dist_logits = dist_logits.permute((0, 4, 1, 2, 3))
    cce = F.cross_entropy(dist_logits, dgram_labels, reduction="none")
    edge_mask = edge_mask.squeeze(-1)
    return (cce * edge_mask).sum(dim=(-1, -2, -3)) / edge_mask.float().sum(dim=(-1, -2, -3))

def multiframe_traj_loss(
    batch,
    model_outputs,
    traj_decay_factor=0.99,
):
    traj_data = model_outputs['traj_data']
    n_layers = len(traj_data)

    res_data = batch['residue']
    device = res_data['atom14'].device

    if isinstance(res_data['rigids_t'], torch.Tensor):
        rigids_t = ru.Rigid.from_tensor_7(res_data['rigids_t'].to(device))
    else:
        rigids_t = res_data['rigids_t']
        dtype = rigids_t._trans.dtype
        rigids_t = ru.Rigid(
            rots=rigids_t._rots.to(device=device, dtype=dtype),
            trans=rigids_t._trans.to(device)
        )
    rigids_1 = ru.Rigid.from_tensor_7(res_data['rigids_1'].to(device))

    res_mask = res_data['res_mask'].view(batch.num_graphs, -1)
    edge_mask = res_mask[..., None] & res_mask[..., None, :]
    gt_trans = rigids_1[..., 0].get_trans().reshape(batch.num_graphs, -1, 3)
    gt_dists = torch.cdist(gt_trans, gt_trans)

    rigids_per_res = rigids_1.shape[-1]

    bb_loss = 0
    rigids_loss = 0
    pred_dist_loss = 0
    pred_framepair_dist_loss = 0
    seq_loss = 0
    seqpair_loss = 0
    for k in range(n_layers):
        data_k = traj_data[k]
        scale = traj_decay_factor ** (n_layers - k + 1)
        rigids_loss_k = interm_multirigid_loss(
            noised_rigids=rigids_t.view(-1, rigids_per_res),
            gt_rigids=rigids_1.view(-1, rigids_per_res),
            pred_rigids=data_k['rigids'].view(-1, rigids_per_res),
            res_batch=res_data.batch,
            total_mask=res_data['res_mask'][..., None].tile(
                (1, rigids_per_res)
            )
        )
        dist_loss_k = interm_pred_dist_loss(
            gt_dist=gt_dists,
            dist_logits=data_k['dist_logits'],
            edge_mask=edge_mask
        )
        seq_loss_k = F.cross_entropy(
            data_k['seq_logits'].transpose(-1, -2),
            res_data['seq'].view(batch.num_graphs, -1),
            reduction="none"
        )
        seq_loss_k = (seq_loss_k * res_mask).sum(dim=-1) / res_mask.float().sum(dim=-1)

        rigids_loss = rigids_loss + rigids_loss_k * scale
        pred_dist_loss = pred_dist_loss + dist_loss_k * scale
        seq_loss = seq_loss + seq_loss_k * scale

        if 'bb_rigids' in data_k:
            bb_loss_k = interm_rigid_loss(
                noised_rigids=rigids_t.view(-1, rigids_per_res)[..., 0],
                gt_rigids=rigids_1.view(-1, rigids_per_res)[..., 0],
                pred_rigids=data_k['bb_rigids'].view(-1),
                res_batch=res_data.batch,
                total_mask=res_data['res_mask']
            )
            bb_loss = bb_loss + bb_loss_k * scale
        else:
            bb_loss += torch.zeros_like(rigids_loss_k)

        if 'framepair_logits_dict' in data_k:
            rigids_mask = res_mask[..., None].tile((1, 1, rigids_per_res))
            framepair_dist_loss_k = interm_pred_framepair_dist_loss(
                gt_rigids=rigids_1.view(batch.num_graphs, -1),
                framepair_preds_dict=data_k['framepair_logits_dict'],
                rigids_mask=rigids_mask
            )
            pred_framepair_dist_loss = pred_framepair_dist_loss + framepair_dist_loss_k * scale
        else:
            pred_framepair_dist_loss = pred_framepair_dist_loss + torch.zeros_like(rigids_loss_k)

        if 'seqpair_logits' in data_k:
            # n_batch x n_logits x n_res x n_res
            seqpair_logits = data_k['seqpair_logits'].permute((0, 3, 1, 2))
            n_aa = data_k['seq_logits'].shape[-1]
            assert n_aa * n_aa == seqpair_logits.shape[1]
            gt_seq = res_data['seq'].view(batch.num_graphs, -1)
            # n_batch x n_res x n_res
            gt_pairwise_seq = gt_seq[..., :, None] * n_aa + gt_seq[..., None, :]
            seqpair_loss_k = F.cross_entropy(
                seqpair_logits,
                gt_pairwise_seq.long(),
                reduction="none"
            )
            seqpair_loss_k = (seqpair_loss_k * edge_mask).sum(dim=(-1, -2)) / edge_mask.float().sum(dim=(-1, -2))
            seqpair_loss = seqpair_loss + seqpair_loss_k * scale
        else:
            seqpair_loss = seqpair_loss + torch.zeros_like(rigids_loss_k)

    return {
        "traj_bb_loss": bb_loss / n_layers,
        "traj_rigids_loss": rigids_loss / n_layers,
        "traj_pred_dist_loss": pred_dist_loss / n_layers,
        "traj_pred_framepair_dist_loss": pred_framepair_dist_loss / n_layers,
        "traj_seq_loss": seq_loss / n_layers,
        "traj_seqpair_loss": seqpair_loss / n_layers,
    }

def atomic_traj_loss(
    batch,
    model_outputs,
    traj_decay_factor=0.99,
):
    traj_data = model_outputs['traj_data']
    n_layers = len(traj_data) + 1

    res_data = batch['residue']
    device = res_data['atom14'].device

    gt_atom14 = res_data['atom14'].view(batch.num_graphs, -1, 14, 3)
    gt_ca = gt_atom14[..., 1, :]
    res_mask = res_data['res_mask'].view(batch.num_graphs, -1)
    edge_mask = res_mask[..., None] & res_mask[..., None, :]
    gt_dists = torch.cdist(gt_ca, gt_ca)

    ca_loss = 0
    pred_dist_loss = 0
    seq_loss = 0
    for k in range(n_layers-1):
        data_k = traj_data[k]
        scale = traj_decay_factor ** (n_layers - k)
        ca_loss_k = torch.square(data_k['ca_pos'] - gt_ca).sum(dim=-1)
        ca_loss_k = (ca_loss_k * res_mask).sum(dim=-1) / res_mask.float().sum(dim=-1)

        dist_loss_k = interm_pred_dist_loss(
            gt_dist=gt_dists,
            dist_logits=data_k['dist_logits'],
            edge_mask=edge_mask
        )
        seq_loss_k = F.cross_entropy(
            data_k['seq_logits'].transpose(-1, -2),
            res_data['seq'].view(batch.num_graphs, -1),
            reduction="none"
        )
        seq_loss_k = (seq_loss_k * res_mask).sum(dim=-1) / res_mask.float().sum(dim=-1)

        ca_loss += ca_loss_k * scale
        pred_dist_loss += dist_loss_k * scale
        seq_loss += seq_loss_k * scale

    return {
        "traj_ca_loss": ca_loss / (n_layers-1),
        "traj_pred_dist_loss": pred_dist_loss / (n_layers-1),
        "traj_seq_loss": seq_loss / (n_layers-1)
    }