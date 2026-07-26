"""
Diagnostic: find which block/channel first produces a diff in node_embed
when ligand is translated 200A under cross_type_mask.

Usage:
    python _scripts/debug_cross_type.py \
        --model-dir outputs/plinder_protein_cond/train \
        --dataset-config configs/train/data/com_pocket.yaml
"""
import argparse
import copy
import sys
import torch
import torch.nn as nn
sys.path.insert(0, '.')

from hydra_zen import instantiate, load_from_yaml
from _scripts.train_com_predictor import load_backbone_and_corrupter
from proteinzen.data.featurize.assembler import collate
from proteinzen.model.utils import gather_helper
from proteinzen.openfold.utils import rigid_utils as ru

_orig_load = torch.load
def _patched_load(*args, weights_only=None, **kwargs):
    if weights_only is None:
        weights_only = False
    return _orig_load(*args, weights_only=weights_only, **kwargs)
torch.load = _patched_load

OFFSET = torch.tensor([200.0, 0.0, 0.0])


def apply_etkdg(batch):
    etkdg_pos = batch['rigids']['etkdg_pos']
    noising_mask = batch['rigids']['rigids_noising_mask'].bool()
    rigids_mask = batch['rigids']['rigids_mask'].bool()
    lig_mask = noising_mask & rigids_mask
    etkdg_valid = (etkdg_pos * lig_mask[..., None]).abs().sum(dim=(1, 2)) > 0
    lig_center = (etkdg_pos * lig_mask[..., None]).sum(1) / lig_mask.float().sum(1, keepdim=True).clamp(min=1)
    etkdg_centered = etkdg_pos - lig_center[:, None, :]
    old = ru.Rigid.from_tensor_7(batch['rigids']['rigids_t'])
    replace_mask = (noising_mask & etkdg_valid[:, None])[..., None]
    new_trans = torch.where(replace_mask, etkdg_centered, old.get_trans())
    batch['rigids']['rigids_t'] = ru.Rigid(rots=old.get_rots(), trans=new_trans).to_tensor_7()
    batch['rigids']['trans_t'] = new_trans
    return batch, etkdg_valid


def translate_ligand(batch, offset):
    b = copy.deepcopy(batch)
    noising_mask = b['rigids']['rigids_noising_mask'].bool()
    rigids_mask = b['rigids']['rigids_mask'].bool()
    lig_mask = noising_mask & rigids_mask
    old = ru.Rigid.from_tensor_7(b['rigids']['rigids_t'])
    dev = old.get_trans().device
    new_trans = old.get_trans() + offset.to(dev) * lig_mask[..., None].float()
    b['rigids']['rigids_t'] = ru.Rigid(rots=old.get_rots(), trans=new_trans).to_tensor_7()
    b['rigids']['trans_t'] = new_trans
    return b


def diff(a, b, label):
    d = (a - b).abs()
    print(f"  {label}: max={d.max():.3e}, mean={d.mean():.3e}")
    return d.max().item()


def hook_diff(name, orig_dict, shift_dict):
    """Register a forward hook that records output tensors for comparison."""
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            orig_dict[name] = output.detach()
        elif isinstance(output, (tuple, list)):
            orig_dict[name] = tuple(o.detach() if isinstance(o, torch.Tensor) else o for o in output)
    return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--version-num', type=int, default=0)
    ap.add_argument('--checkpoint-idx', type=int, default=-1)
    ap.add_argument('--dataset-config', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)

    backbone, corrupter, c_s = load_backbone_and_corrupter(
        args.model_dir, args.version_num, args.checkpoint_idx
    )
    backbone = backbone.to(device).eval()

    train_cfg = load_from_yaml(args.dataset_config)
    ds = instantiate(train_cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)

    for batch in loader:
        def to_dev(x):
            if isinstance(x, dict): return {k: to_dev(v) for k, v in x.items()}
            if isinstance(x, torch.Tensor): return x.to(device)
            return x
        batch = to_dev(batch)

        batch['t'] = torch.ones_like(batch['t'])
        batch['trans_t'] = batch['t']
        batch['rot_t'] = batch['t']
        batch = corrupter.corrupt_dense_batch(batch, identity_rot_noise=False)
        batch, etkdg_valid = apply_etkdg(batch)

        if not etkdg_valid.any():
            print("ETKDG failed, skipping")
            continue

        valid_idx = etkdg_valid.nonzero(as_tuple=True)[0]
        batch = {k: ({kk: vv[valid_idx] for kk, vv in v.items()} if isinstance(v, dict) else v[valid_idx])
                 for k, v in batch.items()}

        batch_shifted = translate_ligand(batch, OFFSET)

        token_to_rep = batch['token']['token_to_rep_rigid']
        token_noising_mask = gather_helper(
            batch['rigids']['rigids_noising_mask'][..., None].float(), token_to_rep
        ).squeeze(-1).bool()
        cross_type_mask = token_noising_mask.unsqueeze(-1) ^ token_noising_mask.unsqueeze(-2)

        b1 = copy.deepcopy(batch)
        b2 = copy.deepcopy(batch_shifted)
        b1['cross_type_mask'] = cross_type_mask
        b2['cross_type_mask'] = cross_type_mask

        print(f"\nToken noising mask: {token_noising_mask.sum().item()} ligand tokens, "
              f"{(~token_noising_mask & batch['token']['token_mask'].bool()).sum().item()} protein tokens")

        # ── Phase 1: compare embedder outputs ─────────────────────────────────
        print("\n=== EMBEDDER ===")
        with torch.no_grad():
            feats1 = backbone.embedder(
                token_mask=b1['token']['token_mask'],
                token_seq_idx=(b1['token']['residue_idx'] if backbone.use_residue_indexing else b1['token']['token_seq_idx']),
                token_seq=b1['token']['res_type'],
                token_seq_mask=b1['token']['token_mask'],
                token_seq_noising_mask=b1['token']['seq_noising_mask'],
                token_asym_id=b1['token']['asym_id'],
                token_entity_id=b1['token']['entity_id'],
                token_is_unindexed_mask=b1['token']['token_is_unindexed_mask'],
                token_is_copy_mask=b1['token']['token_is_copy_mask'],
                token_hotspot_type=b1['token']['hotspot_type'],
                token_gather_idx=b1['token']['token_to_rep_rigid'],
                t=b1['t'],
                rigids=b1['rigids']['rigids_t'],
                rigids_element=b1['rigids']['rigids_ref_element'],
                rigids_charge=b1['rigids']['rigids_ref_charge'],
                rigids_chirality=b1['rigids']['rigids_ref_chirality'],
                rigids_num_real_axes=b1['rigids']['rigids_num_real_axes'],
                rigids_token_uid=b1['rigids']['rigids_to_token'],
                rigids_idx=b1['rigids']['rigids_sidechain_idx'],
                rigids_mask=b1['rigids']['rigids_mask'],
                rigids_is_atomized_mask=b1['rigids']['rigids_is_atom_mask'],
                rigids_noising_mask=b1['rigids']['rigids_noising_mask'],
                token_bonds=b1['token']['token_bonds'],
                sc_rigids=None,
                sf_rigids=None,
                rigids_lap_pe=b1['rigids'].get('rigids_lap_pe'),
                cross_type_mask=cross_type_mask,
            )
            feats2 = backbone.embedder(
                token_mask=b2['token']['token_mask'],
                token_seq_idx=(b2['token']['residue_idx'] if backbone.use_residue_indexing else b2['token']['token_seq_idx']),
                token_seq=b2['token']['res_type'],
                token_seq_mask=b2['token']['token_mask'],
                token_seq_noising_mask=b2['token']['seq_noising_mask'],
                token_asym_id=b2['token']['asym_id'],
                token_entity_id=b2['token']['entity_id'],
                token_is_unindexed_mask=b2['token']['token_is_unindexed_mask'],
                token_is_copy_mask=b2['token']['token_is_copy_mask'],
                token_hotspot_type=b2['token']['hotspot_type'],
                token_gather_idx=b2['token']['token_to_rep_rigid'],
                t=b2['t'],
                rigids=b2['rigids']['rigids_t'],
                rigids_element=b2['rigids']['rigids_ref_element'],
                rigids_charge=b2['rigids']['rigids_ref_charge'],
                rigids_chirality=b2['rigids']['rigids_ref_chirality'],
                rigids_num_real_axes=b2['rigids']['rigids_num_real_axes'],
                rigids_token_uid=b2['rigids']['rigids_to_token'],
                rigids_idx=b2['rigids']['rigids_sidechain_idx'],
                rigids_mask=b2['rigids']['rigids_mask'],
                rigids_is_atomized_mask=b2['rigids']['rigids_is_atom_mask'],
                rigids_noising_mask=b2['rigids']['rigids_noising_mask'],
                token_bonds=b2['token']['token_bonds'],
                sc_rigids=None,
                sf_rigids=None,
                rigids_lap_pe=b2['rigids'].get('rigids_lap_pe'),
                cross_type_mask=cross_type_mask,
            )

        d_node = diff(feats1['node_embed'], feats2['node_embed'], 'node_embed')
        d_edge = diff(feats1['edge_embed'], feats2['edge_embed'], 'edge_embed')
        d_fp = diff(feats1['framepair_embed'], feats2['framepair_embed'], 'framepair_embed')
        d_re = diff(feats1['rigids_embed'], feats2['rigids_embed'], 'rigids_embed')

        # ── Phase 2: walk through IpaDenoiser blocks manually ─────────────────
        print("\n=== IPA DENOISER BLOCKS ===")

        denoiser = backbone.ipa_denoiser

        def make_input(b, feats):
            inp = dict(feats)
            inp['condition_embed'] = feats['time_condition_embed']
            inp['token_mask'] = b['token']['token_mask']
            inp['token_gather_idx'] = b['token']['token_to_rep_rigid']
            inp['t'] = b['t']
            inp['cross_type_mask'] = cross_type_mask
            return inp

        inp1 = make_input(b1, feats1)
        inp2 = make_input(b2, feats2)

        # Reproduce IpaDenoiser.forward() step by step
        def run_block_by_block(inp, label):
            from proteinzen.model.utils import gather_helper
            import functools as fn

            node_embed = inp['node_embed']
            node_mask = inp['token_mask'].float()
            condition_embed = inp['condition_embed']
            edge_embed = inp['edge_embed']
            edge_mask = node_mask[..., None] * node_mask[..., None, :]
            framepair_embed = inp['framepair_embed']
            init_rigids = inp['rigids_t']
            rigids_embed_flat = inp['rigids_embed']
            rigids_token_uid = inp['rigids_to_token']
            rigids_mask_flat = inp['rigids_mask']
            rigids_noising_mask_flat = inp['rigids_noising_mask']
            to_queries = inp['to_queries']
            to_keys = inp['to_keys']
            to_pairs = inp['to_pairs']
            ctm = inp.get('cross_type_mask')
            cross_mask_bias = ctm.float() * -1e5 if ctm is not None else None
            rigids_to_nodes = fn.partial(gather_helper, token_gather_idx=inp['token_gather_idx'])

            curr_rigids = denoiser.scale_rigids(init_rigids)

            outputs = {'node_embed_0': node_embed, 'edge_embed_0': edge_embed}

            for b in range(denoiser.num_blocks):
                curr_rigids_tensor_7 = curr_rigids.to_tensor_7()
                token_rigids = ru.Rigid.from_tensor_7(rigids_to_nodes(curr_rigids_tensor_7))

                with torch.no_grad():
                    if denoiser.use_conditioned_ipa:
                        ipa_embed = denoiser.trunk[f'ipa_{b}'](
                            s=node_embed, cond=condition_embed, z=edge_embed, r=token_rigids,
                            mask=node_mask, cross_mask_bias=cross_mask_bias)
                    else:
                        ipa_embed = denoiser.trunk[f'ipa_{b}'](
                            s=node_embed, z=edge_embed, r=token_rigids,
                            mask=node_mask, cross_mask_bias=cross_mask_bias)
                    node_embed = (node_embed + ipa_embed) * node_mask[..., None]
                    outputs[f'node_after_ipa_{b}'] = node_embed

                    seq_tfmr_out = denoiser.trunk[f'tfmr_{b}'](node_embed, condition_embed, edge_embed, node_mask)
                    node_embed = node_embed + denoiser.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
                    node_embed = node_embed * node_mask[..., None]
                    outputs[f'node_after_tfmr_{b}'] = node_embed

                    node_embed = node_embed + denoiser.trunk[f'transition_{b}'](node_embed, condition_embed)
                    node_embed = node_embed * node_mask[..., None]
                    outputs[f'node_after_trans_{b}'] = node_embed

                    rigids_embed_flat, node_embed, framepair_embed, curr_rigids = denoiser.trunk[f'rigids_tfmr_{b}'](
                        node_embed, edge_embed, framepair_embed, curr_rigids,
                        rigids_embed_flat, rigids_token_uid, rigids_mask_flat,
                        rigids_noising_mask_flat, to_queries, to_keys, to_pairs,
                        rigid_cross_type_mask=rigids_noising_mask_flat if ctm is not None else None,
                    )
                    outputs[f'node_after_rigid_tfmr_{b}'] = node_embed

                    if not denoiser.rigid_transformer_rigid_updates:
                        rigid_update = denoiser.trunk[f'rigids_update_{b}'](rigids_embed_flat * rigids_noising_mask_flat[..., None])
                        curr_rigids = curr_rigids.compose_q_update_vec(rigid_update * rigids_noising_mask_flat[..., None])

                    if b < denoiser.num_blocks - 1:
                        curr_rigids_tensor_7 = curr_rigids.to_tensor_7()
                        token_rigids = ru.Rigid.from_tensor_7(rigids_to_nodes(curr_rigids_tensor_7))
                        edge_embed = denoiser.trunk[f'edge_transition_{b}'](
                            node_embed, edge_embed, token_rigids, edge_mask, cross_type_mask=ctm)
                        edge_embed *= edge_mask[..., None]
                        outputs[f'edge_after_trans_{b}'] = edge_embed

            return outputs

        print("\n[Original]")
        out1 = run_block_by_block(inp1, 'orig')
        print("\n[Shifted]")
        out2 = run_block_by_block(inp2, 'shifted')

        print("\n=== DIFFS BY STEP ===")
        keys_sorted = sorted(out1.keys())
        for k in keys_sorted:
            if k in out2:
                diff(out1[k], out2[k], k)

        break

    print("\nDone.")


if __name__ == '__main__':
    main()
