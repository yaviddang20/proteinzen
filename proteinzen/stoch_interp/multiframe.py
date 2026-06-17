import tqdm
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
import sys
from lightning.pytorch import seed_everything

from proteinzen.openfold.utils import rigid_utils as ru

from torch_geometric.data import HeteroData, Batch
from torch_geometric.utils import scatter
from torch_scatter import scatter_add
import numpy as np

from proteinzen.model.utils import gather_helper
from . import so3_utils

def center_zero(pos: torch.Tensor, batch_indexes: torch.LongTensor) -> torch.Tensor:
    """
    Move the molecule center to zero for sparse position tensors.

    Args:
        pos: [N, 3] batch positions of atoms in the molecule in sparse batch format.
        batch_indexes: [N] batch index for each atom in sparse batch format.

    Returns:
        pos: [N, 3] zero-centered batch positions of atoms in the molecule in sparse batch format.
    """
    assert len(pos.shape) == 2 and pos.shape[-1] == 3, "pos must have shape [N, 3]"

    means = scatter(pos, batch_indexes, dim=0, reduce="mean")
    return pos - means[batch_indexes]

@torch.no_grad()
def align_structures(
    batch_positions: torch.Tensor,
    batch_indices: torch.Tensor,
    reference_positions: torch.Tensor,
    broadcast_reference: bool = False,
):
    """
    Align structures in a ChemGraph batch to a reference, e.g. for RMSD computation. This uses the
    sparse formulation of pytorch geometric. If the ChemGraph is composed of a single system, then
    the reference can be given as a single structure and broadcasted. Returns the structure
    coordinates shifted to the geometric center and the batch structures rotated to match the
    reference structures. Uses the Kabsch algorithm (see e.g. [kabsch_align1]_). No permutation of
    atoms is carried out.

    Args:
        batch_positions (Tensor): Batch of structures (e.g. from ChemGraph) which should be aligned
          to a reference.
        batch_indices (Tensor): Index tensor mapping each node / atom in batch to the respective
          system (e.g. batch attribute of ChemGraph batch).
        reference_positions (Tensor): Reference structure. Can either be a batch of structures or a
          single structure. In the second case, broadcasting is possible if the input batch is
          composed exclusively of this structure.
        broadcast_reference (bool, optional): If reference batch contains only a single structure,
          broadcast this structure to match the ChemGraph batch. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing the centered positions of batch
          structures rotated into the reference and the centered reference batch.

    References
    ----------
    .. [kabsch_align1] Lawrence, Bernal, Witzgall:
       A purely algebraic justification of the Kabsch-Umeyama algorithm.
       Journal of research of the National Institute of Standards and Technology, 124, 1. 2019.
    """
    # Minimize || Q @ R.T - P ||, which is the same as || Q - P @ R ||
    # batch_positions     -> P [BN x 3]
    # reference_positions -> Q [B / BN x 3]

    if batch_positions.shape[0] != reference_positions.shape[0]:
        if broadcast_reference:
            # Get number of systems in batch and broadcast reference structure.
            # This assumes, all systems in the current batch correspond to the reference system.
            # Typically always the case during evaluation.
            num_molecules = int(torch.max(batch_indices) + 1)
            reference_positions = reference_positions.repeat(num_molecules, 1)
        else:
            raise ValueError("Mismatch in batch dimensions.")

    # Center structures at origin (takes care of translation alignment)
    batch_positions = center_zero(batch_positions, batch_indices)
    reference_positions = center_zero(reference_positions, batch_indices)

    # Compute covariance matrix for optimal rotation (Q.T @ P) -> [B x 3 x 3].
    cov = scatter_add(
        batch_positions[:, None, :] * reference_positions[:, :, None], batch_indices, dim=0
    )

    # Perform singular value decomposition. (all [B x 3 x 3])
    u, _, v_t = torch.linalg.svd(cov)
    # Convenience transposes.
    u_t = u.transpose(1, 2)
    v = v_t.transpose(1, 2)

    # Compute rotation matrix correction for ensuring right-handed coordinate system
    # For comparison with other sources: det(AB) = det(A)*det(B) and det(A) = det(A.T)
    sign_correction = torch.sign(torch.linalg.det(torch.bmm(v, u_t)))
    # Correct transpose of U: diag(1, 1, sign_correction) @ U.T
    u_t[:, 2, :] = u_t[:, 2, :] * sign_correction[:, None]

    # Compute optimal rotation matrix (R = V @ diag(1, 1, sign_correction) @ U.T).
    rotation_matrices = torch.bmm(v, u_t)

    # Rotate batch positions P to optimal alignment with Q (P @ R)
    batch_positions_rotated = torch.bmm(
        batch_positions[:, None, :],
        rotation_matrices[batch_indices],
    ).squeeze(1)

    return batch_positions_rotated, reference_positions, rotation_matrices

# from eigenfold
class MolecularHarmonicPrior:
    """Harmonic prior for small molecules using actual bond connectivity.

    Builds a graph Laplacian from the bond adjacency matrix per sample and
    samples from the resulting Gaussian (zero-ing the translation mode).
    Spring constant a = 1/bond_length^2 with bond_length=1.5 Å by default.
    """
    def __init__(self, bond_length=1.5):
        self.a = 1.0 / (bond_length ** 2)

    def sample(self, adj, std=1.0):
        """
        adj: [N, N] float tensor, binary bond adjacency
        std: scalar scale factor applied to samples
        returns: [N, 3] tensor
        """
        n = adj.shape[0]
        device, dtype = adj.device, adj.dtype
        degree = adj.sum(dim=-1)
        J = -self.a * adj.clone()
        J[torch.arange(n, device=device), torch.arange(n, device=device)] = self.a * degree
        D, P = torch.linalg.eigh(J)
        D_inv = torch.where(D > 1e-6, 1.0 / D, torch.zeros_like(D))
        z = torch.randn(n, 3, device=device, dtype=dtype)
        return std * (P @ (torch.sqrt(D_inv)[:, None] * z))


class HarmonicPrior:
    def __init__(self, N = 256, a=3/(3.8**2)):
        J = torch.zeros(N, N)
        for i, j in zip(np.arange(N-1), np.arange(1, N)):
            J[i,i] += a
            J[j,j] += a
            J[i,j] = J[j,i] = -a
        D, P = torch.linalg.eigh(J)
        D_inv = 1/D
        D_inv[0] = 0
        self.P, self.D_inv = P, D_inv
        self.N = N

    def to(self, device):
        self.P = self.P.to(device)
        self.D_inv = self.D_inv.to(device)

    def sample(self, batch_dims=()):
        return self.P @ (torch.sqrt(self.D_inv)[:,None] * torch.randn(*batch_dims, self.N, 3, device=self.P.device))


def _centered_gaussian(batch, rigids_per_res, device):
    noise = torch.randn(batch.shape[0], rigids_per_res, 3, device=device)
    center = scatter(
        noise,
        index=batch,
        dim=0,
        reduce='mean'
    ).mean(dim=-2)
    return noise - center[batch][..., None, :]


def _uniform_so3(num_res, rigids_per_res, device):
    return torch.tensor(
        Rotation.random(num_res * rigids_per_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_res, rigids_per_res, 3, 3)


def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (~diffuse_mask[..., None])


def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return rotmats_t * diffuse_mask[..., None, None] + rotmats_1 * (
        ~diffuse_mask[..., None, None]
    )

class MultiSE3Interpolant:
    def __init__(self,
                 min_t=1e-2,
                 separate_ot=True,
                 prealign_noise=True,
                 trans_preconditioning=False,
                 trans_prior_std=16,
                 rigids_per_res=3,
                 use_stochastic_centering=True,
                 center_on_motif=False,
                 center_on_motif_then_hotspots=True,
                 center_on_noised=False,
                 sig_perturb=4.0,
                 use_uniform_rot_noise=False,
                 rots_use_brownian_path=False,
                 use_unwrapped_rot_noise=False,
                 unwrapped_rot_noise_sig=1.5,
                 use_euclidean_for_rots=False,
                 rot_sfm=False,
    ):
        self._igso3 = None

        self.min_t = min_t
        self.separate_ot = separate_ot
        self.prealign_noise = prealign_noise
        self.trans_preconditioning = trans_preconditioning
        self.trans_prior_std = trans_prior_std

        print(self.igso3)

        self.rigids_per_res = rigids_per_res
        self.use_stochastic_centering = use_stochastic_centering
        self.center_on_motif = center_on_motif
        self.center_on_motif_then_hotspots = center_on_motif_then_hotspots
        self.center_on_noised = center_on_noised
        self.sig_perturb = sig_perturb
        self.use_uniform_rot_noise = use_uniform_rot_noise
        self.use_euclidean_for_rots = use_euclidean_for_rots
        self.rot_sfm = rot_sfm
        self.use_unwrapped_rot_noise = use_unwrapped_rot_noise
        self.unwrapped_rot_noise_sig = unwrapped_rot_noise_sig
        self.rots_use_brownian_path = rots_use_brownian_path

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(1000, sigma_grid, cache_dir=".cache")
        return self._igso3

    def set_device(self, device):
        self._device = device
        self.igso3.to(device)

    def _sample_trans_0(self, batch, device):
        trans_0 = _centered_gaussian(batch, self.rigids_per_res, device)
        trans_0 = trans_0 * self.trans_prior_std
        return trans_0.to(device)

    def _corrupt_trans(self, trans_1, trans_0, t, rigids_mask, diffuse_mask):
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * rigids_mask[..., None]

    def _sample_rotmats_0(self, rotmats_1):
        if self.use_uniform_rot_noise:
            rotmats_0 = _uniform_so3(rotmats_1.shape[0], rotmats_1.shape[1], rotmats_1.device)
        else:
            num_rigids = rotmats_1.shape[0] * rotmats_1.shape[1]
            self.igso3.to(rotmats_1.device)
            noisy_rotmats = self.igso3.sample(torch.tensor([1.5], device=rotmats_1.device), num_rigids).to(rotmats_1.device)
            noisy_rotmats = noisy_rotmats.view(*rotmats_1.shape[:2], 3, 3).float()
            rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        return rotmats_0

    def brownian_sigma_t(self, t):
        sigma_max = 1.5
        return torch.sqrt((0.1 ** 2) * t**2  + (sigma_max ** 2) * (1 - t) **2)

    def brownian_g_t(self, t):
        g_t = torch.sqrt(
            torch.clip(4.5-4.52 * t, min=0)
        )
        return g_t

    def _corrupt_rotmats(self, rotmats_1, rotmats_0, t, rigids_mask, diffuse_mask):
        if self.rots_use_brownian_path:
            self.igso3.to(t.device)
            sigma = self.brownian_sigma_t(t)
            g_t = self.brownian_g_t(t)
            num_rigids = rigids_mask.shape[1]
            noisy_rotmats = self.igso3.sample(
                sigma.squeeze(-1),
                num_rigids
            ).to(t.device)
            rotmats_t = torch.einsum(
                "...ij,...jk->...ik", rotmats_1, noisy_rotmats)
            omega, _, _ = so3_utils.angle_from_rotmat(noisy_rotmats)
            score_scaling = - self.igso3.get_dlog_igso3(omega, sigma)
            rot_vf = F.normalize(
                so3_utils.calc_rot_vf(rotmats_t, rotmats_1)
            ) * score_scaling[..., None]
            E_dlog_igso3 = - self.igso3.get_E_dlog_igso3(sigma)
            E_dlog_igso3_sq = self.igso3.get_E_dlog_igso3_sq(sigma)
        else:
            rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
            rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1) / (1 - t)[..., None]
            score_scaling = 1
            g_t = 1
            E_dlog_igso3 = 1
            E_dlog_igso3_sq = 1

        if self.rot_sfm:
            eps_t = torch.sqrt(0.01 * t * (1-t) + 1e-4)
            self.igso3.to(rotmats_1.device)
            noisy_rotmats = self.igso3.sample(eps_t.view(-1), rotmats_1.shape[1]).to(rotmats_1.device)
            rotmats_t = torch.einsum("...ij,...jk->...ik", rotmats_t, noisy_rotmats)
            rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1) / (1 - t)[..., None]


        identity = torch.eye(3, device=t.device)
        rotmats_t = rotmats_t * rigids_mask[..., None, None] + identity[None, None] * (
            ~rigids_mask[..., None, None]
        )

        rot_vf = rot_vf * rigids_mask[..., None] + torch.zeros_like(rot_vf) * (
            ~rigids_mask[..., None]
        )
        rot_vf = rot_vf * diffuse_mask[..., None] + torch.zeros_like(rot_vf) * (
            ~diffuse_mask[..., None]
        )

        return _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask), rot_vf, score_scaling, g_t, E_dlog_igso3, E_dlog_igso3_sq

    # @torch.no_grad()
    def corrupt_dense_batch(self, batch, identity_rot_noise=False):
        token_data = batch["token"]
        rigids_data = batch["rigids"]

        rigids_1 = ru.Rigid.from_tensor_7(rigids_data["rigids_1"])
        # [N, 5, 3]
        trans_1 = rigids_1.get_trans()
        # [N, 5, 3, 3]
        rotmats_1 = rigids_1.get_rots().get_rot_mats()

        # [N]
        rigids_mask = rigids_data["rigids_mask"]
        rigids_noising_mask = rigids_data["rigids_noising_mask"]
    
        if "rigids_0" in rigids_data:
            rigids_0 = ru.Rigid.from_tensor_7(rigids_data["rigids_0"])
            trans_0 = rigids_0.get_trans()
            rotmats_0 = rigids_0.get_rots().get_rot_mats()
        else:
            if identity_rot_noise:
                is_atom = rigids_data["rigids_is_atom_mask"].bool()
                rotmats_0_sampled = self._sample_rotmats_0(rotmats_1)
                eye = torch.eye(3, device=rotmats_1.device, dtype=torch.float32).expand(*rotmats_1.shape[:-2], 3, 3)
                rotmats_0 = torch.where(is_atom[..., None, None], eye, rotmats_0_sampled)
                rigids_data["rigids_identity_rot_mask"] = is_atom
            else:
                rotmats_0 = self._sample_rotmats_0(rotmats_1)
                rigids_data["rigids_identity_rot_mask"] = torch.zeros_like(rigids_mask, dtype=torch.bool)
            trans_0 = torch.randn_like(trans_1) * self.trans_prior_std
            trans_0 = trans_0 - trans_0.mean(dim=1)[..., None, :]

        global_center = (trans_1 * rigids_mask[..., None]).sum(dim=1) / rigids_mask.long().sum(dim=1)[..., None].clip(min=1)

        assert int(self.center_on_motif) + int(self.center_on_motif_then_hotspots) + int(self.center_on_noised) < 2, (
            "can only choose one of center_on_motif, center_on_hotspots, and center_on_noised"
        )
        if self.center_on_noised:
            # center samples on the center of the noised region
            center_mask = rigids_mask * rigids_noising_mask
            noised_trans_1 = trans_1 * center_mask[..., None]
            noised_center = noised_trans_1.sum(dim=1) / center_mask.long().sum(dim=1)[..., None].clip(min=1)
            use_noised_center = center_mask.any(dim=-1)
            center = noised_center * use_noised_center[..., None] + global_center * (~use_noised_center[..., None])
        elif self.center_on_motif:
            rigid_is_copy = gather_helper(
                token_data["token_is_copy_mask"][..., None],
                rigids_data["rigids_to_token"]
            ).squeeze(-1)
            # center samples on the center of the fixed region
            # if there is no fixed region, center the whole sample
            center_mask = rigids_mask * (~rigids_noising_mask) * rigid_is_copy
            fixed_trans_1 = trans_1 * center_mask[..., None]
            fixed_center = fixed_trans_1.sum(dim=1) / center_mask.long().sum(dim=1)[..., None].clip(min=1)
            use_fixed_center = center_mask.any(dim=-1)
            center = fixed_center * use_fixed_center[..., None] + global_center * (~use_fixed_center[..., None])
        elif self.center_on_motif_then_hotspots:
            rigid_is_copy = gather_helper(
                token_data["token_is_copy_mask"][..., None],
                rigids_data["rigids_to_token"]
            ).squeeze(-1)
            # center samples on the center of the fixed region
            # if there is no fixed region, center the whole sample
            select_copy_mask = rigids_mask * (~rigids_noising_mask) * rigid_is_copy
            copy_center_available = select_copy_mask.any(dim=-1)
            copy_trans_1 = trans_1 * select_copy_mask[..., None]
            copy_center = copy_trans_1.sum(dim=1) / select_copy_mask.long().sum(dim=1)[..., None].clip(min=1)

            rigids_hotspots = gather_helper(
                token_data["hotspot_type"][..., None],
                rigids_data["rigids_to_token"]
            ).squeeze(-1)
            select_hotspots = (rigids_hotspots == 1) & (rigids_mask & ~rigids_noising_mask)
            hotspots_center_available = select_hotspots.any(dim=-1)
            hotspots_center_mask = (rigids_mask & ~rigids_noising_mask) * select_hotspots
            center_trans_1 = trans_1 * hotspots_center_mask[..., None]
            hotspots_center = center_trans_1.sum(dim=1) / hotspots_center_mask.long().sum(dim=1)[..., None].clip(min=1)

            # fallback: if no copy or hotspot rigids, center on all fixed rigids
            select_fixed_mask = rigids_mask * (~rigids_noising_mask)
            fixed_center_available = select_fixed_mask.any(dim=-1) & ~(copy_center_available | hotspots_center_available)
            fixed_trans_1 = trans_1 * select_fixed_mask[..., None]
            fixed_center = fixed_trans_1.sum(dim=1) / select_fixed_mask.long().sum(dim=1)[..., None].clip(min=1)

            use_copy_center = copy_center_available
            use_hotspots_center = hotspots_center_available & (~use_copy_center)
            use_fixed_center = fixed_center_available & ~(use_copy_center | use_hotspots_center)
            use_global_center = ~(use_hotspots_center | use_copy_center | use_fixed_center)
            center = (
                use_copy_center[..., None] * copy_center
                + use_hotspots_center[..., None] * hotspots_center
                + use_fixed_center[..., None] * fixed_center
                + use_global_center[..., None] * global_center
            )
        else:
            center = global_center

        trans_1 = trans_1 - center[..., None, :]
        if 'atom' in batch:
            # this is just so we can calculate atom14 rmsds
            atom_data = batch['atom']
            atom_data["atom14"] = atom_data["atom14"] - center[..., None, None, :]
            atom_data["atom14"] *= atom_data['atom14_mask'][..., None]
            atom_data['atom14_gt_positions'] = atom_data['atom14_gt_positions'] - center[..., None, None, :]
            atom_data['atom14_gt_positions'] *= atom_data['atom14_mask'][..., None]
            atom_data['atom14_alt_gt_positions'] = atom_data['atom14_alt_gt_positions'] - center[..., None, None, :]
            atom_data['atom14_alt_gt_positions'] *= atom_data['atom14_mask'][..., None]


        do_prealign = torch.tensor(
            [getattr(t, "prealign_noise", self.prealign_noise) for t in batch["task"]],
            dtype=torch.bool, device=trans_0.device,
        )

        if do_prealign.any():
            # rotate each structure to align as best as possible with noise
            align_mask = (rigids_mask * rigids_noising_mask).bool()
            align_batch = torch.tile(
                torch.arange(rigids_mask.shape[0])[..., None],
                (1, rigids_mask.shape[1])
            ).to(align_mask.device)
            align_batch = align_batch[align_mask]

            _, _, align_rot_mats = align_structures(
                trans_0[align_mask],
                align_batch,
                trans_1[align_mask]
            )
            num_batch = trans_0.shape[0]

            if align_rot_mats.shape[0] != num_batch:
                num_pad = num_batch - align_rot_mats.shape[0]
                eye = torch.eye(3, device=align_rot_mats.device, dtype=align_rot_mats.dtype)
                align_rot_mats_safe = torch.cat([
                    align_rot_mats,
                    eye[None].expand(num_pad, -1, -1)
                ], dim=0)
            else:
                align_rot_mats_safe = align_rot_mats

            if not do_prealign.all():
                eye = torch.eye(3, device=trans_0.device, dtype=trans_0.dtype)
                align_rot_mats_safe = torch.where(
                    do_prealign[:, None, None],
                    align_rot_mats_safe,
                    eye[None].expand(num_batch, -1, -1),
                )

            trans_0 = torch.einsum("bni,bij->bnj", trans_0, align_rot_mats_safe)

        if self.use_stochastic_centering:
            stoch_center = torch.randn_like(center) * self.sig_perturb
            trans_0 = trans_0 + stoch_center[..., None, :]

        trans_time = batch['trans_t']
        rot_time = batch['rot_t']

        trans_t = self._corrupt_trans(
            trans_1,
            trans_0,
            trans_time,
            rigids_mask,
            rigids_noising_mask.bool(),
        )
        rotmats_t, rot_vf, rot_brownian_score_scaling, rot_brownian_g_t, rot_E_dlog_igso3, rot_E_dlog_igso3_sq = self._corrupt_rotmats(
            rotmats_1,
            rotmats_0,
            rot_time,
            rigids_mask,
            rigids_noising_mask.bool(),
        )

        rotvecs_t = so3_utils.rotmat_to_rotvec(rotmats_t)
        angle_t = torch.linalg.vector_norm(rotvecs_t + 1e-8, dim=-1)
        axis_t = F.normalize(rotvecs_t, dim=-1)
        rotquats_t = torch.cat([
            torch.cos(angle_t/2)[..., None], torch.sin(angle_t/2)[..., None] * axis_t
        ], dim=-1)
        rigids_t = ru.Rigid(
            rots=ru.Rotation(quats=rotquats_t), trans=trans_t
        )

        rigids_data["rigids_0"] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_0), trans=trans_0
        ).to_tensor_7()
        rigids_data["rigids_t"] = rigids_t.to_tensor_7()
        rigids_data["rotmats_t"] = rotmats_t
        rigids_data["trans_t"] = trans_t

        # we also overwrite the ground truth rigids_1 since we've done some centering
        rigids_1 = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_1), trans=trans_1
        )
        rigids_data["rigids_1"] = rigids_1.to_tensor_7()
        rigids_data["gt_rot_vf"] = rot_vf
        rigids_data["rot_brownian_score_scaling"] = rot_brownian_score_scaling
        rigids_data["rot_brownian_g_t"] = rot_brownian_g_t
        rigids_data["rot_brownian_sigma_t"] = self.brownian_sigma_t(rot_time)
        rigids_data["rot_E_dlog_igso3"] = rot_E_dlog_igso3
        rigids_data["rot_E_dlog_igso3_sq"] = rot_E_dlog_igso3_sq

        # we also overwrite the ground truth rigids_0 since we may have done some alignment
        rigids_0 = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_0), trans=trans_0
        )
        rigids_data["rigids_0"] = rigids_0.to_tensor_7()

        return batch
