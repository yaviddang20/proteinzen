import torch
import torch.nn.functional as F

from proteinzen.data.openfold import residue_constants as rc
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.utils.coarse_grain import compute_atom14_from_cg_frames, compute_atom14_frames_from_cg_frames
from proteinzen.utils.framediff.all_atom import adjust_oxygen_pos


def compute_splice_lens(mask):
    split_boundaries = mask[:-1] != mask[1:]
    idx = torch.arange(split_boundaries.numel() + 1)[1:]
    split_ptr = idx[split_boundaries]
    split_ptr = torch.cat([
        torch.zeros(1, device=mask.device),
        split_ptr,
        torch.tensor([mask.numel()], device=mask.device)
    ], dim=0)
    splice_lens = (split_ptr[:-1] - split_ptr[1:]).tolist()
    take_true = torch.stack(
        [m.all() for m in mask.split(splice_lens)]
    )
    return splice_lens, take_true


class RigidAssembler:
    num_cg_rigids = 3
    num_atom_rigids = 14

    def __init__(
            self,
            cg_rigids = None,
            res_mask = None,
            seq = None,
            ligand_rigids = None,
            cg_version=1,
            atomize = False,
            promote_full_motif_to_token = True
    ):
        assert (
            (cg_rigids is not None and res_mask is not None and seq is not None)
            or
            ligand_rigids is not None
        )

        self.cg_rigids = cg_rigids
        self.res_mask = res_mask
        self.seq = seq
        self.cg_version = cg_version
        self.ligand_rigids = ligand_rigids
        self.promote_full_motif_to_token = promote_full_motif_to_token

        if cg_rigids is not None:
            self.seq_idx = torch.arange(cg_rigids.shape[0])

            self.cg_rigid_idx = torch.tile(
                torch.arange(self.num_cg_rigids)[None],
                (len(self.seq_idx), 1)
            )
            self.cg_token_rigid_mask = torch.zeros_like(self.cg_rigid_idx, dtype=torch.bool)
            self.cg_token_rigid_mask[..., 0] = True

            if atomize:
                atom_rigids, atom_rigids_mask = compute_atom14_frames_from_cg_frames(
                    cg_rigids,
                    res_mask,
                    seq,
                    cg_version
                )
                self.atom_rigids = atom_rigids
                self.atom_rigids_mask = atom_rigids_mask
                self.atom_rigid_idx = torch.tile(
                    torch.arange(self.num_cg_rigids, self.num_cg_rigids + self.num_atom_rigids)[None],
                    (len(self.seq_idx), 1)
                )
                self.atom_token_rigid_mask = torch.zeros_like(self.atom_rigid_idx, dtype=torch.bool)
                self.atom_token_rigid_mask[..., 1] = True
            else:
                self.atom_rigids = None
                self.atom_rigids_mask = None
                self.atom_rigid_idx = None
                self.atom_token_rigid_mask = None

        else:
            self.atom_rigids = None
            self.atom_rigids_mask = None
            self.atom_rigid_idx = None
            self.atom_token_rigid_mask = None
            self.cg_rigid_idx = None
            self.cg_token_rigid_mask = None


    def _assemble_protein_no_atomization(
        self,
        rigids_noising_mask,
        res_is_unindexed_mask,
    ):
        assert self.cg_rigids is not None
        assert rigids_noising_mask is not None

        unindexed_motif_mask = (~rigids_noising_mask & res_is_unindexed_mask[..., None]).view(-1)
        indexed_motif_mask = (~rigids_noising_mask & ~res_is_unindexed_mask[..., None]).view(-1)
        # print(self.cg_rigids.shape[0], unindexed_motif_mask.sum(), indexed_motif_mask.sum())

        flat_rigids = self.cg_rigids.flatten(0, 1)
        flat_rigids_mask = (torch.ones_like(rigids_noising_mask) * self.res_mask[..., None]).view(-1)
        flat_noising_mask = torch.ones_like(indexed_motif_mask)
        _seq_idx = torch.tile(
            self.seq_idx[..., None],
            (1, self.cg_rigids.shape[1])
        ).flatten(0, 1)
        flat_seq_idx = _seq_idx.clone()
        flat_rigid_token_uid = flat_seq_idx.clone()
        flat_rigid_idx = self.cg_rigid_idx.flatten(0, 1)
        flat_is_atomized = torch.zeros_like(flat_seq_idx, dtype=torch.bool)
        flat_is_unindexed = torch.zeros_like(flat_seq_idx, dtype=torch.bool)
        flat_is_token_rigid = self.cg_token_rigid_mask.flatten(0, 1)
        flat_is_ligand = torch.zeros_like(flat_noising_mask, dtype=torch.bool)
        flat_is_protein_output = torch.ones_like(flat_noising_mask, dtype=torch.bool)

        if indexed_motif_mask.any():
            indexed_cg_rigids = self.cg_rigids.flatten(0, 1)[indexed_motif_mask]
            indexed_cg_rigids_noising_mask = torch.zeros(indexed_motif_mask.sum().item(), device=indexed_cg_rigids.device)
            indexed_cg_rigids_seq_idx = _seq_idx[indexed_motif_mask]
            indexed_cg_rigids_idx = self.cg_rigid_idx.flatten(0, 1)[indexed_motif_mask]


            flat_rigids = torch.cat([flat_rigids, indexed_cg_rigids], dim=0)
            flat_rigids_mask = torch.cat([flat_rigids_mask, torch.ones_like(indexed_cg_rigids_noising_mask)], dim=0)
            flat_noising_mask = torch.cat([flat_noising_mask, indexed_cg_rigids_noising_mask], dim=0)
            flat_seq_idx = torch.cat([
                flat_seq_idx,
                indexed_cg_rigids_seq_idx
            ], dim=0)
            flat_rigid_idx = torch.cat([
                flat_rigid_idx,
                indexed_cg_rigids_idx
            ], dim=0)
            flat_is_atomized = torch.cat([
                flat_is_atomized,
                torch.zeros_like(indexed_cg_rigids_noising_mask, dtype=torch.bool)
            ], dim=0)
            flat_is_unindexed = torch.cat([
                flat_is_unindexed,
                torch.zeros_like(indexed_cg_rigids_noising_mask, dtype=torch.bool)
            ], dim=0)
            flat_is_ligand = torch.cat([
                flat_is_ligand,
                torch.zeros_like(indexed_cg_rigids_noising_mask, dtype=torch.bool)
            ], dim=0)
            flat_is_protein_output = torch.cat([
                flat_is_protein_output,
                torch.zeros_like(indexed_cg_rigids_noising_mask, dtype=torch.bool)
            ], dim=0)

            if self.promote_full_motif_to_token:
                max_token_uid = flat_rigid_token_uid.max()
                num_indexed_rigids = indexed_motif_mask.long().sum().item()
                new_token_uids = torch.arange(num_indexed_rigids) + max_token_uid + 1

                flat_is_token_rigid = torch.cat([
                    flat_is_token_rigid,
                    torch.ones_like(indexed_cg_rigids_noising_mask, dtype=torch.bool)
                ], dim=0)
                flat_rigid_token_uid = torch.cat([
                    flat_rigid_token_uid,
                    new_token_uids
                ], dim=0)

            else:
                indexed_motif_res_mask = (~rigids_noising_mask & ~res_is_unindexed_mask[..., None]).any(dim=-1)

                max_token_uid = flat_rigid_token_uid.max()
                num_indexed_rigids = indexed_motif_res_mask.long().sum().item()
                new_token_uids = torch.arange(num_indexed_rigids) + max_token_uid + 1

                indexed_motif_token_uids = torch.tile(
                    new_token_uids[..., None],
                    (1, self.num_cg_rigids)
                ).flatten(0, 1)

                flat_is_token_rigid = torch.cat([
                    flat_is_token_rigid,
                    self.cg_token_rigid_mask.flatten(0, 1)[indexed_motif_mask]
                ], dim=0)
                flat_rigid_token_uid = torch.cat([
                    flat_rigid_token_uid,
                    indexed_motif_token_uids
                ], dim=0)


        if unindexed_motif_mask.any():
            unindexed_cg_rigids = self.cg_rigids.flatten(0, 1)[unindexed_motif_mask]
            unindexed_cg_rigids_noising_mask = torch.zeros(unindexed_motif_mask.sum().item(), device=unindexed_cg_rigids.device)
            unindexed_cg_rigids_seq_idx = _seq_idx[unindexed_motif_mask]
            unindexed_cg_rigids_idx = self.cg_rigid_idx.flatten(0, 1)[unindexed_motif_mask]

            max_token_uid = flat_rigid_token_uid.max()
            num_unindexed_rigids = unindexed_motif_mask.long().sum().item()
            new_token_uids = torch.arange(num_unindexed_rigids) + max_token_uid + 1

            flat_rigids = torch.cat([flat_rigids, unindexed_cg_rigids], dim=0)
            flat_rigids_mask = torch.cat([flat_rigids_mask, torch.ones_like(unindexed_cg_rigids_noising_mask)], dim=0)
            flat_noising_mask = torch.cat([flat_noising_mask, unindexed_cg_rigids_noising_mask], dim=0)
            flat_seq_idx = torch.cat([
                flat_seq_idx,
                unindexed_cg_rigids_seq_idx
            ], dim=0)
            flat_rigid_idx = torch.cat([
                flat_rigid_idx,
                unindexed_cg_rigids_idx
            ], dim=0)
            flat_is_atomized = torch.cat([
                flat_is_atomized,
                torch.zeros_like(unindexed_cg_rigids_noising_mask, dtype=torch.bool)
            ], dim=0)
            flat_is_unindexed = torch.cat([
                flat_is_unindexed,
                torch.ones_like(unindexed_cg_rigids_noising_mask, dtype=torch.bool)
            ], dim=0)

            flat_is_ligand = torch.cat([
                flat_is_ligand,
                torch.zeros_like(unindexed_cg_rigids_noising_mask, dtype=torch.bool)
            ], dim=0)
            flat_is_protein_output = torch.cat([
                flat_is_protein_output,
                torch.zeros_like(unindexed_cg_rigids_noising_mask, dtype=torch.bool)
            ], dim=0)


            if self.promote_full_motif_to_token:
                max_token_uid = flat_rigid_token_uid.max()
                num_unindexed_rigids = unindexed_motif_mask.long().sum().item()
                new_token_uids = torch.arange(num_unindexed_rigids) + max_token_uid + 1

                flat_is_token_rigid = torch.cat([
                    flat_is_token_rigid,
                    torch.ones_like(unindexed_cg_rigids_noising_mask, dtype=torch.bool)
                ], dim=0)
                flat_rigid_token_uid = torch.cat([
                    flat_rigid_token_uid,
                    new_token_uids
                ], dim=0)
            else:
                unindexed_motif_res_mask = (~rigids_noising_mask & res_is_unindexed_mask[..., None]).any(dim=-1)

                max_token_uid = flat_rigid_token_uid.max()
                num_unindexed_rigids = unindexed_motif_res_mask.long().sum().item()
                new_token_uids = torch.arange(num_unindexed_rigids) + max_token_uid + 1

                unindexed_motif_token_uids = torch.tile(
                    new_token_uids[..., None],
                    (1, self.num_cg_rigids)
                ).flatten(0, 1)

                flat_is_token_rigid = torch.cat([
                    flat_is_token_rigid,
                    self.cg_token_rigid_mask.flatten(0, 1)[unindexed_motif_mask]
                ], dim=0)
                flat_rigid_token_uid = torch.cat([
                    flat_rigid_token_uid,
                    unindexed_motif_token_uids
                ], dim=0)

        return {
            "rigids": flat_rigids,
            "rigids_mask": flat_rigids_mask.bool(),
            "rigids_noising_mask": flat_noising_mask.bool(),
            "seq_idx": flat_seq_idx,
            "token_uid": flat_rigid_token_uid,
            "rigid_idx": flat_rigid_idx,
            "is_atomized_mask": flat_is_atomized.bool(),
            "is_unindexed_mask": flat_is_unindexed.bool(),
            "is_token_rigid_mask": flat_is_token_rigid.bool(),
            "is_ligand_mask": flat_is_ligand.bool(),
            "is_protein_output_mask": flat_is_protein_output.bool(),
            "splice_lens": torch.tensor([flat_is_atomized.numel()], dtype=torch.long)
        }


    def _assemble_protein_full_atomization(
        self,
        res_atom_noising_mask,
        res_is_unindexed_mask,
    ):
        assert self.atom_rigids is not None
        assert self.atom_rigids_mask is not None
        assert res_atom_noising_mask is not None

        # now, splice together the proper chunks
        flat_rigids = self.atom_rigids[self.atom_rigids_mask]
        flat_noising_mask = res_atom_noising_mask[self.atom_rigids_mask]
        flat_seq_idx = torch.tile(
            self.seq_idx[..., None],
            (1, self.cg_rigids.shape[1])
        )[self.atom_rigids_mask]
        flat_rigid_idx = self.atom_rigid_idx[self.atom_rigids_mask]
        flat_is_atomized = torch.ones_like(flat_seq_idx, dtype=torch.bool)
        flat_is_unindexed = torch.zeros_like(flat_seq_idx, dtype=torch.bool)
        flat_is_center_rigid = self.atom_center_rigid_mask[self.atom_rigids_mask]

        if res_is_unindexed_mask is not None:
            unindexed_atom_rigids = self.atom_rigids[res_is_unindexed_mask][self.atom_rigids_mask[res_is_unindexed_mask]]
            unindexed_atom_rigids_noising_mask = res_atom_noising_mask[res_is_unindexed_mask][self.atom_rigids_mask[res_is_unindexed_mask]]

            flat_rigids = torch.cat([flat_rigids, unindexed_atom_rigids], dim=0)
            flat_noising_mask = torch.cat([flat_noising_mask, unindexed_atom_rigids_noising_mask], dim=0)

            num_unindexed_res = int(res_is_unindexed_mask.long().sum())
            num_indexed_res = res_is_unindexed_mask.numel()
            unindexed_seq_idx = torch.tile(
                torch.arange(num_unindexed_res)[..., None],
                (1, self.num_atom_rigids)
            ) + num_indexed_res
            flat_seq_idx = torch.cat([
                flat_seq_idx,
                unindexed_seq_idx[self.atom_rigids_mask[res_is_unindexed_mask]]
            ], dim=0)
            unindexed_atom_rigid_idx = torch.tile(
                torch.arange(self.atom_rigids.shape[1])[None],
                (int(res_is_unindexed_mask.sum()), 1)
            )
            flat_rigid_idx = torch.cat([
                flat_rigid_idx,
                unindexed_atom_rigid_idx.flatten(0, 1)
            ], dim=0)
            flat_is_atomized = torch.cat([
                flat_is_atomized,
                torch.zeros_like(unindexed_atom_rigids_noising_mask, dtype=torch.bool)
            ], dim=0)
            flat_is_unindexed = torch.cat([
                flat_is_unindexed,
                torch.ones_like(unindexed_atom_rigids_noising_mask, dtype=torch.bool)
            ], dim=0)
            flat_is_center_rigid = torch.cat([
                flat_is_center_rigid,
                torch.ones_like(unindexed_atom_rigids_noising_mask, dtype=torch.bool)
            ], dim=0)

        flat_is_ligand = torch.zeros(
            sum([t.shape[0] for t in flat_noising_mask]),
            dtype=torch.bool
        )

        return {
            "rigids": flat_rigids,
            "rigids_noising_mask": flat_noising_mask,
            "seq_idx": flat_seq_idx,
            "rigid_idx": flat_rigid_idx,
            "is_atomized_mask": flat_is_atomized,
            "is_center_rigid_mask": flat_is_center_rigid,
            "is_ligand_mask": flat_is_ligand,
            "splice_lens": torch.tensor([flat_is_atomized.numel()], dtype=torch.long)
        }


    def _assemble_protein_partial_atomization(
        self,
        rigids_noising_mask,
        res_atom_noising_mask,
        res_atomized_mask,
        res_is_unindexed_mask,
    ):
        assert self.cg_rigids is not None
        assert self.atom_rigids is not None
        assert self.atom_rigids_mask is not None
        assert rigids_noising_mask is not None
        assert res_atom_noising_mask is not None
        assert res_atomized_mask is not None

        # if we atomize residues
        # we need to place them in the proper sequence order
        # since we use sequence-local blocks to save compute
        # first, split everything into chunks
        splice_lens, take_atom_rigids = compute_splice_lens(res_atomized_mask)
        cg_rigid_chunks = self.cg_rigids.split(splice_lens)
        atom_rigids_chunks = self.atom_rigids.split(splice_lens)
        atom_rigids_mask_chunks = self.atom_rigids_mask.split(splice_lens)
        rigids_noising_mask_chunks = rigids_noising_mask.split(splice_lens)
        res_atom_noising_mask_chunks = res_atom_noising_mask.split(splice_lens)
        seq_idx_chunks = self.seq_idx.split(splice_lens)
        cg_rigid_idx_chunks = self.cg_rigid_idx.split(splice_lens)
        atom_rigid_idx_chunks = self.atom_rigid_idx.split(splice_lens)
        cg_center_rigid_mask_chunks = self.cg_center_rigid_mask.split(splice_lens)
        atom_center_rigid_mask_chunks = self.atom_center_rigid_mask.split(splice_lens)

        # now, splice together the proper chunks
        flat_rigids = []
        flat_noising_mask = []
        flat_seq_idx = []
        flat_rigid_idx = []
        flat_is_atomized = []
        flat_is_center_rigid = []
        for i, take_atom_rigids_i in enumerate(take_atom_rigids):
            if take_atom_rigids_i:
                atom_rigids_mask_i = atom_rigids_mask_chunks[i]
                atom_rigids_i = atom_rigids_chunks[i]
                atom_noising_mask_i = res_atom_noising_mask_chunks[i]

                rigid_idx_i = atom_rigid_idx_chunks[i]
                seq_idx_i = torch.tile(
                    seq_idx_chunks[i][..., None],
                    (1, rigid_idx_i.shape[-1])
                )
                is_center_rigid_mask_i = atom_center_rigid_mask_chunks[i]

                flat_rigids.append(atom_rigids_i[atom_rigids_mask_i])
                flat_noising_mask.append(atom_noising_mask_i[atom_rigids_mask_i])
                flat_seq_idx.append(seq_idx_i[atom_rigids_mask_i])
                flat_rigid_idx.append(rigid_idx_i[atom_rigids_mask_i])
                flat_is_atomized.append(torch.ones_like(seq_idx_i[atom_rigids_mask_i], dtype=torch.bool))
                flat_is_center_rigid.append(is_center_rigid_mask_i[atom_rigids_mask_i])
            else:
                rigid_idx_i = cg_rigid_idx_chunks[i]
                seq_idx_i = torch.tile(
                    seq_idx_chunks[i][..., None],
                    (1, rigid_idx_i.shape[-1])
                )
                is_center_rigid_mask_i = cg_center_rigid_mask_chunks[i]

                flat_rigids.append(cg_rigid_chunks[i].flatten(0, 1))
                flat_noising_mask.append(rigids_noising_mask_chunks[i].flatten(0, 1))
                flat_seq_idx.append(seq_idx_i.flatten(0, 1))
                flat_rigid_idx.append(rigid_idx_i.flatten(0, 1))
                flat_is_atomized.append(torch.zeros_like(seq_idx_i.flatten(0, 1), dtype=torch.bool))
                flat_is_center_rigid.append(is_center_rigid_mask_i.flatten(0, 1))

        if res_is_unindexed_mask is not None:
            # process unindexed rigids
            unindexed_atomized_res_mask = res_atomized_mask & res_is_unindexed_mask
            unindexed_cg_res_mask = (~res_atomized_mask) & res_is_unindexed_mask
            unindexed_atomized_rigids = self.atom_rigids[
                unindexed_atomized_res_mask
            ][self.atom_rigids_mask[unindexed_atomized_res_mask]]
            unindexed_atomized_rigids_noising_mask = res_atom_noising_mask[
                unindexed_atomized_res_mask
            ][self.atom_rigids_mask[unindexed_atomized_res_mask]]
            unindexed_cg_rigids = self.cg_rigids[unindexed_cg_res_mask].flatten(0, 1)
            unindexed_cg_rigids_noising_mask = rigids_noising_mask[unindexed_cg_res_mask].flatten(0, 1)

            flat_rigids.append(unindexed_atomized_rigids)
            flat_noising_mask.append(unindexed_atomized_rigids_noising_mask)
            flat_seq_idx.append(torch.full_like(unindexed_atomized_rigids_noising_mask, -1))
            unindexed_atomized_rigid_idx = torch.tile(
                torch.arange(self.atom_rigids.shape[1])[None],
                (int(unindexed_atomized_res_mask.sum()), 1)
            )
            flat_rigid_idx.append(unindexed_atomized_rigid_idx[self.atom_rigids_mask[unindexed_atomized_res_mask]])
            flat_is_atomized.append(torch.ones_like(unindexed_atomized_rigids_noising_mask, dtype=torch.bool))
            flat_is_center_rigid.append(torch.ones_like(unindexed_atomized_rigids_noising_mask, dtype=torch.bool))

            flat_rigids.append(unindexed_cg_rigids)
            flat_noising_mask.append(unindexed_cg_rigids_noising_mask)
            flat_seq_idx.append(torch.full_like(unindexed_cg_rigids_noising_mask, -1))
            unindexed_cg_rigid_idx = torch.tile(
                torch.arange(self.cg_rigids.shape[1])[None],
                (int(unindexed_cg_res_mask.sum()), 1)
            )
            flat_rigid_idx.append(unindexed_cg_rigid_idx.flatten(0, 1))
            flat_is_atomized.append(torch.zeros_like(unindexed_cg_rigids_noising_mask, dtype=torch.bool))
            flat_is_center_rigid.append(torch.ones_like(unindexed_cg_rigids_noising_mask, dtype=torch.bool))

        flat_is_ligand = torch.zeros(
            sum([t.shape[0] for t in flat_noising_mask]),
            dtype=torch.bool
        )
        flat_rigids = torch.cat(flat_rigids, dim=0)
        flat_noising_mask = torch.cat(flat_noising_mask, dim=0)
        flat_seq_idx = torch.cat(flat_seq_idx, dim=0)
        flat_rigid_idx = torch.cat(flat_rigid_idx, dim=0)
        flat_is_atomized = torch.cat(flat_is_atomized, dim=0)
        flat_is_center_rigid = torch.cat(flat_is_center_rigid, dim=0)

        return {
            "rigids": flat_rigids,
            "rigids_noising_mask": flat_noising_mask,
            "seq_idx": flat_seq_idx,
            "rigid_idx": flat_rigid_idx,
            "is_atomized_mask": flat_is_atomized,
            "is_center_rigid_mask": flat_is_center_rigid,
            "is_ligand_mask": flat_is_ligand,
            "splice_lens": torch.as_tensor(splice_lens, dtype=torch.long)
        }


    def _assemble_ligand(self, ligand_noising_mask):
        assert self.ligand_rigids is not None
        return {
            "rigids": self.ligand_rigids,
            "rigids_noising_mask": ligand_noising_mask,
            "seq_idx": torch.full_like(ligand_noising_mask, -1, dtype=torch.float32),
            "rigid_idx": torch.full_like(ligand_noising_mask, -1, dtype=torch.float32),
            "is_atomized_mask": torch.ones_like(ligand_noising_mask, dtype=torch.bool),
            "is_center_mask": torch.ones_like(ligand_noising_mask, dtype=torch.bool),
            "is_ligand": torch.ones_like(ligand_noising_mask, dtype=torch.bool),
            "splice_lens": torch.tensor([ligand_noising_mask.numel()], dtype=torch.long)  # TODO: should we split by ligand?
        }


    def assemble(
        self,
        rigids_noising_mask = None,
        res_atom_noising_mask = None,
        res_atomized_mask = None,
        res_is_unindexed_mask = None,
        ligand_noising_mask = None,
    ):
        if res_atomized_mask is not None:
            # 2 shortcuts to avoid overhead for partial atomization
            if not res_atomized_mask.any():
                protein_rigids_dict = self._assemble_protein_no_atomization(
                    rigids_noising_mask,
                    res_is_unindexed_mask
                )
            elif res_atomized_mask.all():
                raise NotImplementedError("i'll do this later")
                protein_rigids_dict = self._assemble_protein_full_atomization(
                    res_atom_noising_mask,
                    res_is_unindexed_mask
                )
            else:
                raise NotImplementedError("i'll do this later")
                protein_rigids_dict = self._assemble_protein_partial_atomization(
                    rigids_noising_mask,
                    res_atom_noising_mask,
                    res_atomized_mask,
                    res_is_unindexed_mask
                )
        else:
            protein_rigids_dict = {}

        if ligand_noising_mask is not None:
            raise NotImplementedError("i'll do this later")
            ligand_rigids_dict = self._assemble_ligand(ligand_noising_mask)
        else:
            ligand_rigids_dict = {}

        if len(protein_rigids_dict) > 0 and len(ligand_rigids_dict) > 0:
            rigids_dict = {}
            for key in protein_rigids_dict: # pylint: disable=C0206
                rigids_dict[key] = torch.cat([ligand_rigids_dict[key], protein_rigids_dict[key]], dim=0)
        elif len(protein_rigids_dict) > 0:
            rigids_dict = protein_rigids_dict
        elif len(ligand_rigids_dict) > 0:
            rigids_dict = ligand_rigids_dict
        else:
            raise ValueError("we were not provided enough information to construct input rigids")

        is_token_rigid_mask = rigids_dict['is_token_rigid_mask']
        rigids_dict['token_gather_idx'] = torch.arange(is_token_rigid_mask.numel())[is_token_rigid_mask]

        return rigids_dict


def rigids_to_atom14(
    rigids,
    rigids_mask,
    rigids_is_protein_output_mask,
    rigids_is_atomized_mask,
    token_is_atomized_mask,
    token_is_protein_output_mask,
    seq,
    cg_version
):

    cg_res_pos_flat = ~token_is_atomized_mask[token_is_protein_output_mask]
    cg_res_lens = token_is_protein_output_mask.sum(dim=-1)
    cg_res_pos_list = cg_res_pos_flat.split(cg_res_lens.tolist())
    cg_res_pos_mask = torch.zeros(
        (
            token_is_protein_output_mask.shape[0],
            cg_res_lens.max().item(),
        ),
        dtype=torch.bool,
        device=cg_res_lens.device
    )
    for i, t in enumerate(cg_res_pos_list):
        cg_res_pos_mask[i, :cg_res_lens[i].item()] = t


    atom14 = torch.zeros(
        (
            *cg_res_pos_mask.shape,
            14, 3
        ),
        dtype=torch.float32,
        device=token_is_atomized_mask.device
    )
    cg_rigids = rigids[rigids_is_protein_output_mask].view(-1, 3)
    res_mask = rigids_mask[rigids_is_protein_output_mask].view(-1, 3)[..., 0]

    cg_seq = seq[token_is_protein_output_mask * (~token_is_atomized_mask)]
    # print(cg_rigids.shape, res_mask.shape, cg_seq.shape)
    cg_atom14 = compute_atom14_from_cg_frames(
        cg_rigids,
        res_mask,
        cg_seq,
        cg_version
    )
    # print(cg_atom14.shape, token_is_atomized_mask.sum(), atom14.shape, cg_res_pos_mask.shape)
    atom14[cg_res_pos_mask] += cg_atom14

    if token_is_atomized_mask.any():
        atomized_seq = seq[token_is_atomized_mask]
        atom_mask = torch.as_tensor(rc.restype_atom14_mask, device=seq.device)
        atom_mask = atom_mask[atomized_seq]
        atom14[rigids_is_atomized_mask][atom_mask] += rigids[rigids_is_atomized_mask].get_trans()

        # adjust the bb oxygen to be based off of the other backbone atoms
        atom37_bb = torch.zeros((atom14.shape[0], 37, 3), device=atom14.device)
        # atom14 bb order = ['N', 'CA', 'C', 'O', 'CB']
        # atom37 bb order = ['N', 'CA', 'C', 'CB', 'O']
        atom37_bb[..., :3, :] = atom14[..., :3, :]
        atom37_bb[..., 3, :] = atom14[..., 4, :]
        atom37_bb[..., 4, :] = atom14[..., 3, :]
        atom37 = adjust_oxygen_pos(atom37_bb.view(-1, 37, 3), res_mask.view(-1)).view(atom37_bb.shape)
        atom14[..., 3, :] = atom37[..., 4, :]

    return atom14
