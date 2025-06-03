import torch
from torch import nn
import torch.nn.functional as F

from proteinzen.model.modules.layers.node.attention import TransformerPairBias
from proteinzen.model.modules.openfold.layers_v2 import Linear, LayerNorm, Transition, AdaLN
from proteinzen.utils.openfold import feats, rigid_utils as ru
from proteinzen.data.openfold import residue_constants
from proteinzen.data.constants import (
    coarse_grain,
    _coarse_grain_v3 as cg_v3,
    _coarse_grain_v4 as cg_v4,
    _coarse_grain_v5 as cg_v5,
    _coarse_grain_v6 as cg_v6,
    _coarse_grain_v7 as cg_v7,
    _coarse_grain_v8 as cg_v8,
)

# Residue Constants from OpenFold/AlphaFold2.
ATOM_MASK = torch.tensor(residue_constants.restype_atom14_mask)

def _gen_version_dict(module):
    cg_mask = [module.cg_group_mask[name] for name in module.coarse_grain_sidechain_groups]
    cg_mask.append([0.0, 0.0, 0.0, 0.0]) # for X
    return {
        "IDEALIZED_POS": torch.tensor(module.restype_atom14_cg_group_positions),
        "DEFAULT_FRAMES": torch.tensor(module.restype_cg_group_default_frame),
        "GROUP_IDX": torch.tensor(module.restype_atom14_to_cg_group),
        "CG_MASK": torch.tensor(cg_mask)
    }

cg_version_constants = {
    2: _gen_version_dict(coarse_grain),
    3: _gen_version_dict(cg_v3),
    4: _gen_version_dict(cg_v4),
    5: _gen_version_dict(cg_v5),
    6: _gen_version_dict(cg_v6),
    7: _gen_version_dict(cg_v7),
    8: _gen_version_dict(cg_v8),
}


def _preprocess_cg_constants(cg_version):
    cg_data = cg_version_constants[cg_version]
    element_one_hot_key = {
        "C": 1,
        "N": 2,
        "O": 3,
        "S": 4,
        " ": 0
    }
    #
    atom14_elements = [
        [
            element_one_hot_key[atom_name[0]] if len(atom_name) > 0 else -1 for atom_name in atom_list
        ]
        for _, atom_list in
        residue_constants.restype_name_to_atom14_names.items()
    ]
    # [21, 14]
    atom14_elements = torch.tensor(atom14_elements)


    GROUP_IDX = cg_data["GROUP_IDX"]
    DEFAULT_FRAMES = cg_data["DEFAULT_FRAMES"]
    IDEALIZED_POS = cg_data["IDEALIZED_POS"]
    CG_MASK = cg_data["CG_MASK"]

    # [21, 4, 14]
    group_mask = F.one_hot(
        GROUP_IDX,
        num_classes=DEFAULT_FRAMES.shape[-3],
    ).transpose(-1, -2)
    group_atom_mask = group_mask * ATOM_MASK[..., None, :]
    group_atom_element = group_atom_mask * atom14_elements[..., None, :]

    # [21, 4]
    empty_group = torch.isclose(group_atom_mask.float(), torch.zeros(1)).all(dim=-1)
    group_exists = ~empty_group

    # [N, 14, 3]
    atom_positions = (IDEALIZED_POS[:, None] * group_exists[..., None, None])[group_exists]
    # [N, 14]
    atom_mask = group_atom_mask[group_exists]
    # [N, 14]
    atom_element = group_atom_element[group_exists]
    # [N]
    atom_res_seq = torch.arange(21)[..., None].expand(-1, 4).contiguous()[group_exists]

    return atom_positions, atom_mask, atom_element, atom_res_seq, group_exists, CG_MASK



class FrameMemory(nn.Module):
    def __init__(self,
                 c_frame,
                 c_atompair=32,
                 num_heads=4,
                 cg_version=2):
        super().__init__()

        self.coord_embed = Linear(3, c_frame, bias=False)
        self.seq_embed = Linear(21, c_frame, bias=False)
        self.element_embed = Linear(5, c_frame, bias=False)
        self.element_pair_embed = Linear(5, c_atompair, bias=False)
        self.disp_embed = Linear(3, c_atompair, bias=False)
        self.dist_embed = Linear(1, c_atompair, bias=False)

        self.ffn = nn.Sequential(
            nn.ReLU(),
            Linear(c_atompair, c_atompair, bias=False),
            nn.ReLU(),
            Linear(c_atompair, c_atompair, bias=False),
            nn.ReLU(),
            Linear(c_atompair, c_atompair, bias=False),
        )

        self.tfmr = TransformerPairBias(
            c_s=c_frame,
            c_z=c_atompair,
            no_heads=num_heads,
            n_layers=3
        )

        ref_atom_positions, ref_atom_mask, ref_atom_element, ref_atom_res_seq, ref_group_exists, ref_cg_mask = _preprocess_cg_constants(cg_version)
        self.register_buffer("ref_atom_positions", ref_atom_positions)
        self.register_buffer("ref_atom_mask", ref_atom_mask)
        self.register_buffer("ref_atom_element", ref_atom_element)
        self.register_buffer("ref_atom_residue_seq", ref_atom_res_seq)
        self.register_buffer("ref_group_exists", ref_group_exists)
        self.register_buffer("ref_cg_mask", ref_cg_mask)

    def _pad(self, input):
        assert input.shape[0] == self.ref_group_exists.float().sum()
        out = torch.zeros((21, 4, input.shape[-1]), device=input.device, dtype=torch.float32)
        out[self.ref_group_exists.to(out.device)] = input
        mask_AG = torch.zeros(21, device=out.device)
        mask_AG[residue_constants.resname_to_idx['ALA']] = 1
        mask_AG[residue_constants.resname_to_idx['GLY']] = 1
        out = (
            out
            + out[:, [2]] * (1 - self.ref_cg_mask[..., None]) * (1 - mask_AG[:, None, None])
            + out[:, [0]] * (1 - self.ref_cg_mask[..., None]) * mask_AG[:, None, None]
        )

        return out

    def forward(self):
        atom_pos_embed = self.coord_embed(self.ref_atom_positions)
        atom_elem_embed = self.element_embed(
            F.one_hot(self.ref_atom_element.long(), num_classes=5).float()
        )
        atom_seq_embed = self.seq_embed(
            F.one_hot(self.ref_atom_residue_seq.long(), num_classes=21).float()
        )
        atom_embed = atom_pos_embed + atom_elem_embed + atom_seq_embed[..., None, :]

        atom_elem_pair_embed = self.element_pair_embed(
            F.one_hot(self.ref_atom_element.long(), num_classes=5).float()
        )
        atompair_disp = self.ref_atom_positions[..., None, :, :] - self.ref_atom_positions[..., None, :]
        atompair_disp_embed = self.disp_embed(atompair_disp)
        atompair_dist = torch.linalg.vector_norm(atompair_disp, dim=-1, keepdims=True)
        atompair_dist_embed = self.dist_embed(1 / (1 + atompair_dist ** 2))
        atompair_embed = atompair_disp_embed + atompair_dist_embed + atom_elem_pair_embed[..., None, :] + atom_elem_pair_embed[..., None, :, :]

        # [N, 14, c_frame]
        atom_embed = self.tfmr(atom_embed, atompair_embed, self.ref_atom_mask)
        atom_embed = atom_embed * self.ref_atom_mask[..., None]
        # [N, c_frame]
        frame_embed = atom_embed.sum(dim=-2) / torch.clip(self.ref_atom_mask.float().sum(dim=-1), min=1)[..., None]
        # [21, 3, c_frame]
        frame_embed = self._pad(frame_embed)[..., (0, 2, 3), :]
        frame_mask = self.ref_group_exists[..., (0, 2, 3)]
        print(frame_embed.shape, frame_mask.shape)
        return frame_embed, frame_mask


class FrameMemoryRetrieval(nn.Module):
    def __init__(self,
                 c_frame,
                 num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.ln_q = LayerNorm(c_frame)
        self.ln_kv = LayerNorm(c_frame)
        self.lin_q = Linear(c_frame, c_frame, bias=False)
        self.lin_kv = Linear(c_frame, 2*c_frame, bias=False)

        self.out_gate = Linear(c_frame, c_frame, bias=False)
        self.lin_out = Linear(c_frame, c_frame, bias=False)

        self.ffn = Transition(c_frame, n=2)

    def _attn(self, rigids_embed, rigids_memory, rigids_mask, memory_mask):
        rigids_embed = self.ln_q(rigids_embed)
        rigids_memory = self.ln_kv(rigids_memory)

        # B x N x n_rigids, 21 x n_rigids
        mask = rigids_mask[..., None, :] * memory_mask[None, None]

        # B x N x n_rigids x c_frame
        q = self.lin_q(rigids_embed)
        # B x N x n_rigids x n_heads x c_hidden
        q = q.unflatten(-1, (self.num_heads, -1))

        # 21 x n_rigids x c_frame
        kv = self.lin_kv(rigids_memory).unflatten(-1, (2 * self.num_heads, -1))
        # 21 x n_rigids x n_heads x c_hidden
        k, v = kv.split([self.num_heads, self.num_heads], dim=-2)

        a = torch.einsum("binhc,jnhc->bijnh", q, k)
        a += (mask[..., None].float() - 1) * 1e6
        a = torch.softmax(a, dim=-1)

        out = torch.einsum("bijnh,jnhc->binhc", a, v)
        gate = self.out_gate(rigids_embed).view(out.shape)
        out = out * torch.sigmoid(gate)
        out = self.lin_out(out.reshape(rigids_embed.shape))

        return out * rigids_mask[..., None]

    def forward(self, rigids_embed, rigids_memory, rigids_mask, memory_mask):
        rigids_embed = rigids_embed + self._attn(rigids_embed, rigids_memory, rigids_mask, memory_mask)
        rigids_embed = rigids_embed + self.ffn(rigids_embed) * rigids_mask[..., None]
        return rigids_embed

