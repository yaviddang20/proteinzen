import abc
from typing import Optional

import torch
import tqdm

from proteinzen.stoch_interp.model_wrapper import ModelForwardWrapper
from proteinzen.stoch_interp.diffeq import DifferentialEquation, EDMEulerSDEStep, CustomLangevinEquilibrationStep
from proteinzen.openfold.utils import rigid_utils as ru

from proteinzen.stoch_interp.multiframe import _trans_diffuse_mask, _rots_diffuse_mask, align_structures, _uniform_so3
from proteinzen.stoch_interp import so3_utils

class Integrator(abc.ABC):
    def __init__(
        self,
        *,
        wrapped_model: ModelForwardWrapper,
        diffeq: DifferentialEquation,
        unmask_seq: bool = False,
        **kwargs
    ):
        self.model = wrapped_model
        self.diffeq = diffeq
        self.unmask_seq = unmask_seq

    def sample(
        self,
        batch,
        ts,
    ):
        rigids_data = batch['rigids']
        rigids_data['rigids_t'] = rigids_data['rigids_1']
        rigids_0 = ru.Rigid.from_tensor_7(rigids_data['rigids_t'])
        trans_0 = rigids_0.get_trans()
        rotmats_0 = rigids_0.get_rots().get_rot_mats()

        prot_traj = [(
            trans_0,
            rotmats_0,
            None
        )]

        clean_traj = []

        # set up initial integration conditions
        t_1 = ts[0]
        prev_denoiser_out = None

        num_steps = len(ts) - 1
        if self.unmask_seq:
            # Count initially masked (binder) tokens per sample — fixed reference
            n_masked_init = batch['token']['seq_noising_mask'].sum(dim=-1).long()  # [B]

        for step_idx, t_2 in enumerate(tqdm.tqdm(ts[1:])):
            trans_t, rotmats_t, _ = prot_traj[-1]
            prev_denoiser_out, prot_traj_point, clean_traj_point = self.integration_step(
                batch,
                trans_t,
                rotmats_t,
                t_1,
                t_2,
                self_conditioning=prev_denoiser_out
            )
            prot_traj.append(prot_traj_point)
            clean_traj.append(clean_traj_point)

            if self.unmask_seq:
                seq_logits = prev_denoiser_out['decoded_seq_logits']  # [B, N_res, n_aa+1]
                seq_noising_mask = batch['token']['seq_noising_mask']  # [B, N_res]
                probs = torch.softmax(seq_logits[..., :-1], dim=-1)   # [B, N_res, n_aa]
                confidence = probs.max(dim=-1).values                  # [B, N_res]
                pred_seq = seq_logits[..., :-1].argmax(dim=-1)        # [B, N_res]

                for b in range(seq_noising_mask.shape[0]):
                    n_b = int(n_masked_init[b].item())
                    if n_b == 0:
                        continue
                    target_committed = round((step_idx + 1) * n_b / num_steps)
                    already_committed = n_b - int(seq_noising_mask[b].sum().item())
                    n_to_commit = max(0, target_committed - already_committed)
                    if n_to_commit == 0:
                        continue
                    n_still_masked = int(seq_noising_mask[b].sum().item())
                    k = min(n_to_commit, n_still_masked)
                    masked_conf = confidence[b] * seq_noising_mask[b].float()
                    top_k_idx = masked_conf.topk(k).indices
                    batch['token']['seq_noising_mask'][b, top_k_idx] = False
                    batch['token']['res_type'][b, top_k_idx] = pred_seq[b, top_k_idx]
                    batch['token']['seq'][b, top_k_idx] = pred_seq[b, top_k_idx]

            t_1 = t_2

        trans_t, rotmats_t, _ = prot_traj[-1]
        final_denoiser_out, _, _ = self.integration_step(
            batch,
            trans_t,
            rotmats_t,
            ts[-1],
            ts[-1],
            self_conditioning=prev_denoiser_out,
        )

        # Process model output.
        prot_traj = prot_traj[1:]

        return clean_traj, prot_traj, final_denoiser_out

    @abc.abstractmethod
    def integration_step(
        self,
        batch,
        trans_t,
        rotmats_t,
        t_1,
        t_2,
        self_conditioning=None
    ):
        raise NotImplementedError


class EDMIntegrator(Integrator):
    def __post_init__(self):
        assert isinstance(self.diffeq, EDMEulerSDEStep)

    def integration_step(
        self,
        batch,
        trans_t_1,
        rotmats_t_1,
        t_1,
        t_2,
        aux_inputs=None,
        self_conditioning=None,
        seed=None
    ):
        d_t = t_2 - t_1
        rigids_data = batch['rigids']
        token_data = batch['token']
        rigids_noising_mask = rigids_data['rigids_noising_mask']
        seq_noising_mask = token_data['seq_noising_mask']
        num_batch, num_res = seq_noising_mask.shape
        device = rigids_noising_mask.device
        denoiser_out = None

        t_1 = t_1.to(device=device)
        d_t = d_t.to(device=device)

        t_hat, d_t_hat, trans_t_hat = self.diffeq.trans_churn(
            trans_t_1,
            t_1,
            d_t,
            rigids_noising_mask,
            seed=seed
        )
        _, _, rotmats_t_hat = self.diffeq.rot_churn(
            rotmats_t_1,
            t_1,
            d_t,
            rigids_noising_mask,
            seed=seed
        )

        # Run model.
        rigids_data["trans_t"] = trans_t_hat
        rigids_data["rotmats_t"] = rotmats_t_hat
        rigids_data['rigids_t'] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t_hat),
            trans=trans_t_hat
        ).to_tensor_7()
        t = torch.ones(num_batch, device=device)[..., None] * t_hat
        batch["t"] = t
        batch["trans_t"] = t
        batch["rot_t"] = t

        scores_and_vfs, denoiser_out = self.model.get_scores_and_vfs(
            batch,
            aux_inputs=aux_inputs,
            self_condition=self_conditioning
        )

        # Process model output.
        pred_rigids = denoiser_out['denoised_rigids']
        pred_trans_1 = pred_rigids.get_trans()
        pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()

        clean_traj_point = (
            pred_trans_1,
            pred_rotmats_1,
            denoiser_out["pred_seq"].detach().cpu(),
        )

        trans_t_2 = self.diffeq.trans_step(
            trans_t_hat,
            t_hat,
            d_t_hat,
            scores_and_vfs,
            rigids_noising_mask
        )
        rotmats_t_2 = self.diffeq.rot_step(
            rotmats_t_hat,
            t_hat,
            d_t_hat,
            scores_and_vfs,
            rigids_noising_mask
        )

        prot_traj_point = (
            trans_t_2,
            rotmats_t_2,
            denoiser_out["pred_seq"].detach().cpu(),
        )

        return denoiser_out, prot_traj_point, clean_traj_point #, trans_t_hat, rotmats_t_hat, scores_and_vfs


class EulerIntegrator(Integrator):
    def __init__(self, *, wrapped_model, diffeq, no_rot_sampling=False, **kwargs):
        super().__init__(wrapped_model=wrapped_model, diffeq=diffeq, **kwargs)
        self.no_rot_sampling = no_rot_sampling

    def integration_step(
        self,
        batch,
        trans_t_1,
        rotmats_t_1,
        t_1,
        t_2,
        aux_inputs=None,
        self_conditioning=None
    ):
        d_t = t_2 - t_1
        rigids_data = batch['rigids']
        token_data = batch['token']
        rigids_noising_mask = rigids_data['rigids_noising_mask']
        seq_noising_mask = token_data['seq_noising_mask']
        num_batch, num_res = seq_noising_mask.shape
        device = rigids_noising_mask.device

        # Run model.
        rigids_data["trans_t"] = trans_t_1
        rigids_data["rotmats_t"] = rotmats_t_1
        rigids_data['rigids_t'] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t_1),
            trans=trans_t_1
        ).to_tensor_7()
        t = torch.ones(num_batch, device=device)[..., None] * t_1
        batch["t"] = t
        batch["trans_t"] = t
        batch["rot_t"] = t

        scores_and_vfs, denoiser_out = self.model.get_scores_and_vfs(
            batch,
            aux_inputs=aux_inputs,
            self_condition=self_conditioning
        )

        # Process model output.
        pred_rigids = denoiser_out['denoised_rigids']
        pred_trans_1 = pred_rigids.get_trans()
        pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()

        clean_traj_point = (
            pred_trans_1,
            pred_rotmats_1,
            denoiser_out["pred_seq"].detach().cpu(),
        )

        trans_t_2 = self.diffeq.trans_step(
            trans_t_1,
            t,
            d_t,
            scores_and_vfs,
            rigids_noising_mask
        )
        if self.no_rot_sampling:
            rotmats_t_2 = rotmats_t_1
        else:
            rotmats_t_2 = self.diffeq.rot_step(
                rotmats_t_1,
                t,
                d_t,
                scores_and_vfs,
                rigids_noising_mask
            )

        prot_traj_point = (
            trans_t_2,
            rotmats_t_2,
            denoiser_out["pred_seq"].detach().cpu(),
        )

        return denoiser_out, prot_traj_point, clean_traj_point


class PnPIntegrator(Integrator):
    def __init__(
        self,
        *,
        wrapped_model: ModelForwardWrapper,
        diffeq: DifferentialEquation,
        pnp_cycles: int = 2,
        pnp_cutoff: float = 0.5,
        use_aligned_noise: bool=False,
        **kwargs
    ):
        super().__init__(wrapped_model=wrapped_model, diffeq=diffeq)
        self.pnp_cycles = pnp_cycles
        self.pnp_cutoff = pnp_cutoff
        sigma_grid = torch.linspace(0.1, 1.5, 1000)
        self._igso3 = so3_utils.SampleIGSO3(1000, sigma_grid, cache_dir=".cache")
        self.use_aligned_noise = use_aligned_noise

    def _corrupt_trans(self, trans_1, trans_0, t, rigids_mask, diffuse_mask):
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * rigids_mask[..., None]

    def _corrupt_rotmats(self, rotmats_1, rotmats_0, t, rigids_mask, diffuse_mask):
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=t.device)
        rotmats_t = rotmats_t * rigids_mask[..., None, None] + identity[None, None] * (
            ~rigids_mask[..., None, None]
        )

        return _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask)

    def _sample_rotmats_0(self, rotmats_1):
        # num_rigids = rotmats_1.shape[0] * rotmats_1.shape[1]
        # noisy_rotmats = self._igso3.sample(torch.tensor([1.5]), num_rigids).to(rotmats_1.device)
        # noisy_rotmats = noisy_rotmats.view(*rotmats_1.shape[:2], 3, 3).float()
        noisy_rotmats = _uniform_so3(rotmats_1.shape[0], rotmats_1.shape[1], device=rotmats_1.device)
        rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        return rotmats_0

    def encode(
        self,
        batch,
        pred_trans_1,
        pred_rotmats_1,
        t
    ):
        rigids_data = batch["rigids"]

        # [N]
        rigids_mask = rigids_data["rigids_mask"]
        rigids_noising_mask = rigids_data["rigids_noising_mask"]

        # get encoding noise
        rotmats_0 = self._sample_rotmats_0(pred_rotmats_1)
        trans_0 = torch.randn_like(pred_trans_1) * 16
        trans_0 = trans_0 - trans_0.mean(dim=1)[..., None, :]

        if self.use_aligned_noise:
            # get COM
            global_center = (pred_trans_1 * rigids_mask[..., None]).sum(dim=1) / rigids_mask.long().sum(dim=1)[..., None].clip(min=1)

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
                pred_trans_1[align_mask]
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
            trans_0 = torch.einsum("bni,bij->bnj", trans_0, align_rot_mats_safe)

            stoch_center = torch.randn_like(global_center)
            trans_0 = trans_0 + stoch_center[..., None, :]

        trans_t = self._corrupt_trans(
            pred_trans_1,
            trans_0,
            t,
            rigids_mask,
            rigids_noising_mask.bool(),
        )
        rotmats_t = self._corrupt_rotmats(
            pred_rotmats_1,
            rotmats_0,
            t,
            rigids_mask,
            rigids_noising_mask.bool(),
        )
        return trans_t, rotmats_t

    def integration_step(
        self,
        batch,
        trans_t_1,
        rotmats_t_1,
        t_1,
        t_2,
        aux_inputs=None,
        self_conditioning=None
    ):
        d_t = t_2 - t_1
        rigids_data = batch['rigids']
        token_data = batch['token']
        rigids_noising_mask = rigids_data['rigids_noising_mask']
        seq_noising_mask = token_data['seq_noising_mask']
        num_batch, num_res = seq_noising_mask.shape
        device = rigids_noising_mask.device

        # Run model.
        rigids_data["trans_t"] = trans_t_1
        rigids_data["rotmats_t"] = rotmats_t_1
        rigids_data['rigids_t'] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t_1),
            trans=trans_t_1
        ).to_tensor_7()
        t = torch.ones(num_batch, device=device)[..., None] * t_1
        batch["t"] = t
        batch["trans_t"] = t
        batch["rot_t"] = t

        scores_and_vfs, denoiser_out = self.model.get_scores_and_vfs(
            batch,
            aux_inputs=aux_inputs,
            self_condition=self_conditioning
        )

        if t_1 < self.pnp_cutoff:
            for k in range(self.pnp_cycles):
                # Process model output.
                pred_rigids = denoiser_out['denoised_rigids']
                pred_trans_1 = pred_rigids.get_trans()
                pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()

                trans_t_1_k, rotmats_t_1_k = self.encode(
                    batch, pred_trans_1, pred_rotmats_1, t_1.to(pred_trans_1.device)
                )

                # Run model.
                rigids_data["trans_t"] = trans_t_1_k
                rigids_data["rotmats_t"] = rotmats_t_1_k
                rigids_data['rigids_t'] = ru.Rigid(
                    rots=ru.Rotation(rot_mats=rotmats_t_1_k),
                    trans=trans_t_1_k
                ).to_tensor_7()
                t = torch.ones(num_batch, device=device)[..., None] * t_1
                batch["t"] = t
                batch["trans_t"] = t
                batch["rot_t"] = t
                scores_and_vfs, denoiser_out = self.model.get_scores_and_vfs(
                    batch,
                    aux_inputs=aux_inputs,
                    self_condition=self_conditioning
                )
            trans_t_2 = self.diffeq.trans_step(
                trans_t_1_k,
                t,
                d_t,
                scores_and_vfs,
                rigids_noising_mask
            )
            rotmats_t_2 = self.diffeq.rot_step(
                rotmats_t_1_k,
                t,
                d_t,
                scores_and_vfs,
                rigids_noising_mask
            )
        else:
            trans_t_2 = self.diffeq.trans_step(
                trans_t_1,
                t,
                d_t,
                scores_and_vfs,
                rigids_noising_mask
            )
            rotmats_t_2 = self.diffeq.rot_step(
                rotmats_t_1,
                t,
                d_t,
                scores_and_vfs,
                rigids_noising_mask
            )


        # Process model output.
        pred_rigids = denoiser_out['denoised_rigids']
        pred_trans_1 = pred_rigids.get_trans()
        pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()

        clean_traj_point = (
            pred_trans_1,
            pred_rotmats_1,
            denoiser_out["pred_seq"].detach().cpu(),
        )

        prot_traj_point = (
            trans_t_2,
            rotmats_t_2,
            denoiser_out["pred_seq"].detach().cpu(),
        )

        return denoiser_out, prot_traj_point, clean_traj_point


class EDMIntegrator2(Integrator):
    def __post_init__(self):
        assert isinstance(self.diffeq, EDMEulerSDEStep)

    def integration_step(
        self,
        batch,
        trans_t_1,
        rotmats_t_1,
        t_1,
        t_2,
        aux_inputs=None,
        self_conditioning=None,
        seed=None
    ):
        d_t = t_2 - t_1
        rigids_data = batch['rigids']
        token_data = batch['token']
        rigids_noising_mask = rigids_data['rigids_noising_mask']
        seq_noising_mask = token_data['seq_noising_mask']
        num_batch, num_res = seq_noising_mask.shape
        device = rigids_noising_mask.device
        denoiser_out = None

        t_1 = t_1.to(device=device)
        d_t = d_t.to(device=device)

        t_hat, d_t_hat, trans_t_hat = self.diffeq.trans_churn(
            trans_t_1,
            t_1,
            d_t,
            rigids_noising_mask,
            seed=seed,
            noise_scale_override=(
                1.0 if t_1 < 0.15
                else None
            )
        )
        _, _, rotmats_t_hat = self.diffeq.rot_churn(
            rotmats_t_1,
            t_1,
            d_t,
            rigids_noising_mask,
            seed=seed,
            noise_scale_override=(
                1.0 if t_1 < 0.15
                else None
            )
        )

        # Run model.
        rigids_data["trans_t"] = trans_t_hat
        rigids_data["rotmats_t"] = rotmats_t_hat
        rigids_data['rigids_t'] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t_hat),
            trans=trans_t_hat
        ).to_tensor_7()
        t = torch.ones(num_batch, device=device)[..., None] * t_hat
        batch["t"] = t
        batch["trans_t"] = t
        batch["rot_t"] = t

        scores_and_vfs, denoiser_out = self.model.get_scores_and_vfs(
            batch,
            aux_inputs=aux_inputs,
            self_condition=self_conditioning
        )

        # Process model output.
        pred_rigids = denoiser_out['denoised_rigids']
        pred_trans_1 = pred_rigids.get_trans()
        pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()

        clean_traj_point = (
            pred_trans_1,
            pred_rotmats_1,
            denoiser_out["pred_seq"].detach().cpu(),
        )

        trans_t_2 = self.diffeq.trans_step(
            trans_t_hat,
            t_hat,
            d_t_hat,
            scores_and_vfs,
            rigids_noising_mask
        )
        rotmats_t_2 = self.diffeq.rot_step(
            rotmats_t_hat,
            t_hat,
            d_t_hat,
            scores_and_vfs,
            rigids_noising_mask
        )

        prot_traj_point = (
            trans_t_2,
            rotmats_t_2,
            denoiser_out["pred_seq"].detach().cpu(),
        )

        return denoiser_out, prot_traj_point, clean_traj_point #, trans_t_hat, rotmats_t_hat, scores_and_vfs


class IntervalEquilibrationIntegrator(Integrator):
    def __init__(
        self,
        *,
        wrapped_model: ModelForwardWrapper,
        diffeq: DifferentialEquation,
        t_start: float,
        t_end: float,
        t_step: float,
        n_equil: int,
        equil_factor_exponent: int
    ):
        self.model = wrapped_model
        self.diffeq = diffeq

        self.t_start = t_start
        self.t_end = t_end
        self.t_step = t_step
        self.n_equil = n_equil
        self.langevin_diffeq = CustomLangevinEquilibrationStep(
            trans_psi=1,
            rot_psi=(2 ** equil_factor_exponent)
        )

    def sample(
        self,
        batch,
        ts,
    ):
        rigids_data = batch['rigids']
        rigids_data['rigids_t'] = rigids_data['rigids_1']
        rigids_0 = ru.Rigid.from_tensor_7(rigids_data['rigids_t'])
        trans_0 = rigids_0.get_trans()
        rotmats_0 = rigids_0.get_rots().get_rot_mats()

        prot_traj = [(
            trans_0,
            rotmats_0,
            None
        )]

        clean_traj = []

        # set up initial integration conditions
        t_1 = ts[0]
        prev_denoiser_out = None
        for t_2 in tqdm.tqdm(ts[1:]):
            if t_2 > self.t_start and t_2 < self.t_end:
                for _ in range(self.n_equil):
                    trans_t, rotmats_t, _ = prot_traj[-1]
                    prev_denoiser_out, prot_traj_point, clean_traj_point = self.integration_step(
                        self.langevin_diffeq,
                        batch,
                        trans_t,
                        rotmats_t,
                        t_1,
                        t_1 + self.t_step,
                        self_conditioning=prev_denoiser_out
                    )
                    prot_traj.append(prot_traj_point)
                    clean_traj.append(clean_traj_point)
            trans_t, rotmats_t, _ = prot_traj[-1]
            prev_denoiser_out, prot_traj_point, clean_traj_point = self.integration_step(
                self.diffeq,
                batch,
                trans_t,
                rotmats_t,
                t_1,
                t_2,
                self_conditioning=prev_denoiser_out
            )
            prot_traj.append(prot_traj_point)
            clean_traj.append(clean_traj_point)
            t_1 = t_2

        trans_t, rotmats_t, _ = prot_traj[-1]
        final_denoiser_out, _, _ = self.integration_step(
            self.diffeq,
            batch,
            trans_t,
            rotmats_t,
            ts[-1],
            ts[-1],
            self_conditioning=prev_denoiser_out,
        )

        # Process model output.
        prot_traj = prot_traj[1:]

        return clean_traj, prot_traj, final_denoiser_out

    def integration_step(
        self,
        diffeq,
        batch,
        trans_t_1,
        rotmats_t_1,
        t_1,
        t_2,
        self_conditioning=None
    ):
        d_t = t_2 - t_1
        rigids_data = batch['rigids']
        token_data = batch['token']
        rigids_noising_mask = rigids_data['rigids_noising_mask']
        seq_noising_mask = token_data['seq_noising_mask']
        num_batch, num_res = seq_noising_mask.shape
        device = rigids_noising_mask.device

        # Run model.
        rigids_data["trans_t"] = trans_t_1
        rigids_data["rotmats_t"] = rotmats_t_1
        rigids_data['rigids_t'] = ru.Rigid(
            rots=ru.Rotation(rot_mats=rotmats_t_1),
            trans=trans_t_1
        ).to_tensor_7()
        t = torch.ones(num_batch, device=device)[..., None] * t_1
        batch["t"] = t
        batch["trans_t"] = t
        batch["rot_t"] = t

        scores_and_vfs, denoiser_out = self.model.get_scores_and_vfs(
            batch,
            aux_inputs=None,
            self_condition=self_conditioning
        )

        # Process model output.
        pred_rigids = denoiser_out['denoised_rigids']
        pred_trans_1 = pred_rigids.get_trans()
        pred_rotmats_1 = pred_rigids.get_rots().get_rot_mats()

        clean_traj_point = (
            pred_trans_1,
            pred_rotmats_1,
            denoiser_out["pred_seq"].detach().cpu(),
        )

        trans_t_2 = diffeq.trans_step(
            trans_t_1,
            t,
            d_t,
            scores_and_vfs,
            rigids_noising_mask
        )
        rotmats_t_2 = diffeq.rot_step(
            rotmats_t_1,
            t,
            d_t,
            scores_and_vfs,
            rigids_noising_mask
        )

        prot_traj_point = (
            trans_t_2,
            rotmats_t_2,
            denoiser_out["pred_seq"].detach().cpu(),
        )

        return denoiser_out, prot_traj_point, clean_traj_point