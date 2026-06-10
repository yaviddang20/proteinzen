import abc
import functools as fn

import torch
import torch.nn.functional as F
from tree import map_structure, map_structure_with_path

from proteinzen.stoch_interp import so3_utils
from proteinzen.openfold.utils.rigid_utils import Rigid, Rotation
from proteinzen.model.denoiser import IpaMultiRigidDenoiser
from proteinzen.model.utils import gather_helper


class ModelForwardWrapper(abc.ABC):
    def __init__(
        self,
        model: IpaMultiRigidDenoiser,
        rot_noise_mode: str = "igso3_prior",
        compute_scores: bool = False,
        input_transform = None,
        output_transform = None,
    ):
        assert rot_noise_mode in ["uniform_prior", "igso3_prior", "heuristic"]
        for p in model.parameters():
            p.requires_grad = False
        self.model = model
        self.rot_noise_mode = rot_noise_mode
        self.compute_scores = compute_scores
        self.input_transform = input_transform
        self.output_transform = output_transform

    def compute_base_trans_vf(self, t, trans_1, trans_t):
        return (trans_1 - trans_t) / (1 - t)[..., None]

    def trans_vf_to_trans_score(self, t, trans_vf, trans_t):
        return (t[..., None] * trans_vf - trans_t) / (1-t)[..., None]

    def trans_score_to_trans_vf(self, t, trans_score, trans_t):
        # trans_score = (t * trans_vf - trans_t) / (1-t)
        # thus trans_vf = [(1-t) * trans_score + trans_t] / t
        # TODO: this is likely numerically unstable around t close to 0
        trans_vf = ((1-t)[..., None] * trans_score + trans_t) / t[..., None].clip(min=0.01)
        return trans_vf

    def compute_base_rot_vf(self, t, rotmats_1, rotmats_t):
        rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1) / (1 - t[..., None])
        rot_vf = rot_vf.float()
        return rot_vf

    def _get_rot_score_prefactor(self, t, rotmats_1, rotmats_t):
        if self.rot_noise_mode == "heuristic":
            ls = torch.arange(1000, device=rotmats_t.device)
            rel_rotmat = torch.einsum("...ij,...jk->...ik", rotmats_t.transpose(-1, -2), rotmats_1)
            omega, _, _ = so3_utils.angle_from_rotmat(rel_rotmat)
            sigma = (
                ((1-t) * 1.5).square()
                + (t * 0.1).square()
            ).sqrt()
            sigma = sigma.expand(omega.shape).to(omega.device)
            omega = omega.view(-1)
            sigma = sigma.contiguous().view(-1)
            prefactor = so3_utils.dlog_igso3_expansion(omega, sigma, ls)
            prefactor = prefactor.view(rotmats_t.shape[:-2])
            omega = omega.view(rotmats_t.shape[:-2])
            prefactor = prefactor / omega
        else:
            rel_rotmat = torch.einsum("...ij,...jk->...ik", rotmats_t.transpose(-1, -2), rotmats_1)
            theta_t, _, _ = so3_utils.angle_from_rotmat(rel_rotmat)

            def calc_cot_prefactor(theta_t):
                small_angle = theta_t < 0.01
                small_angle_approx = 1 / 6 * (1 / (1-t)**2 - 1)
                small_angle_approx = small_angle_approx * torch.ones_like(theta_t)
                cot = lambda x: torch.cos(x) / torch.sin(x)
                factor = cot(theta_t/2) - 1/(1-t) * cot(theta_t / (2 * (1-t)))
                factor[small_angle] = small_angle_approx[small_angle]
                return factor

            prefactor = calc_cot_prefactor(theta_t)

            if self.rot_noise_mode == "igso3_prior":
                ls = torch.arange(1000, device=rotmats_t.device)
                omega = theta_t.view(-1)
                sigma = torch.ones_like(omega) * 1.5
                igso3_prefactor = so3_utils.dlog_igso3_expansion(omega, sigma, ls)
                igso3_prefactor = igso3_prefactor.view(rotmats_t.shape[:-2])
                omega = omega.view(rotmats_t.shape[:-2])
                prefactor = prefactor - (igso3_prefactor / omega.clip(min=0.01))

        return prefactor

    def rot_vf_to_rot_score(self, t, rot_vf, rotmats_1, rotmats_t):
        prefactor = self._get_rot_score_prefactor(t, rotmats_1, rotmats_t)
        if self.rot_noise_mode == "heuristic":
            rot_score = prefactor[..., None] * so3_utils.calc_rot_vf(rotmats_1, rotmats_t)
        else:
            rot_score = (prefactor * (1 - t))[..., None] * rot_vf
        return rot_score

    def rot_score_to_rot_vf(self, t, rot_score, rotmats_1, rotmats_t):
        prefactor = self._get_rot_score_prefactor(t, rotmats_1, rotmats_t)
        # rot_score = prefactor[..., None] * log_vec
        # rot_vf = log_vec / (1 - t)
        # so rot_vf = (1-t) / prefactor * rot_score
        rot_vf = rot_score / ((1 - t) * prefactor)[..., None]
        return rot_vf

    def extract_score_from_tensor7_grad(self, tensor7):
        tensor7_grad = tensor7.grad
        quat = tensor7[..., :4]
        quat_grad, trans_grad = torch.split(tensor7_grad, [4, 3], dim=-1)
        rotvec_grad = so3_utils.quat_grad_to_so3(quat, quat_grad)
        return rotvec_grad, trans_grad

    @abc.abstractmethod
    def compute_gradient(self, model_input, aux_inputs=None, self_condition=None):
        raise NotImplementedError()

    @torch.no_grad()
    def get_scores_and_vfs(self, model_input, aux_inputs=None, self_condition=None):
        # run the model and compute any necessary gradients
        if self.input_transform is not None:
            model_input = self.input_transform(model_input)

        model_input, model_outputs = self.compute_gradient(
            model_input,
            aux_inputs,
            self_condition
        )
        if self.output_transform is not None:
            model_outputs = self.output_transform(model_input, model_outputs)

        # extract necessary inputs and outputs
        trans_time = model_input['trans_t']
        rot_time = model_input['rot_t']
        pred_rigids_1 = model_outputs['denoised_rigids']
        pred_rotmats_1 = pred_rigids_1.get_rots().get_rot_mats()
        pred_trans_1 = pred_rigids_1.get_trans()
        # NOTE! to maintain numerical equivalence with
        # the previous sampler
        # we have to use the original rotmats
        # not the rotmats we get from rigids_t
        # it's not clear to me whether the deviation is practically significant
        # but we carry around the original rotmats anyways so might as well
        trans_t = model_input['rigids']['trans_t']
        rotmats_t = model_input['rigids']['rotmats_t']
        rigids_t_tensor7 = model_input['rigids']['rigids_t']

        # compute base scores and vfs
        base_rot_vf = self.compute_base_rot_vf(rot_time, pred_rotmats_1, rotmats_t)
        base_trans_vf = self.compute_base_trans_vf(trans_time, pred_trans_1, trans_t)

        scores_and_vfs = {
            "base_rot_vf": base_rot_vf.detach(),
            "base_trans_vf": base_trans_vf.detach(),
        }

        if self.compute_scores:
            base_rot_score = self.rot_vf_to_rot_score(rot_time, base_rot_vf, pred_rotmats_1, rotmats_t)
            base_trans_score = self.trans_vf_to_trans_score(trans_time, base_trans_vf, trans_t)
            scores_and_vfs.update({
                "base_rot_score": base_rot_score.detach(),
                "base_trans_score": base_trans_score.detach()
            })


        # if we computed gradients, compute scores and vfs from those and record them
        if rigids_t_tensor7.grad is not None:
            assert self.rot_noise_mode != "heuristic", "we can't do clean conversions under the heuristic score function"
            aux_rot_score, aux_trans_score = self.extract_score_from_tensor7_grad(rigids_t_tensor7)
            model_input['rigids']['rigids_t'] = rigids_t_tensor7.detach()

            aux_rot_vf = self.rot_score_to_rot_vf(
                rot_time,
                aux_rot_score,
                pred_rotmats_1,
                rotmats_t
            )
            aux_trans_vf = self.trans_score_to_trans_vf(
                trans_time,
                aux_trans_score,
                trans_t
            )

            scores_and_vfs.update({
                "gradient_rot_vf": aux_rot_vf.detach(),
                "gradient_rot_score": aux_rot_score.detach(),
                "gradient_trans_vf": aux_trans_vf.detach(),
                "gradient_trans_score": aux_trans_score.detach(),
            })

        # remove all grads from outputs
        for key, value in scores_and_vfs.items():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                scores_and_vfs[key] = value.detach()
        for key, value in model_outputs.items():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                model_outputs[key] = value.detach()

        print({
            key: value.nan_to_num().norm(dim=-1).mean()
            for key, value in scores_and_vfs.items()
        })
        print({
            key: value.nan_to_num().norm(dim=-1).var()
            for key, value in scores_and_vfs.items()
        })

        return scores_and_vfs, model_outputs


class BaseModelForward(ModelForwardWrapper):
    def compute_gradient(self, model_input, aux_inputs=None, self_condition=None):
        model_outputs = self.model(
            model_input,
            self_condition=self_condition
        )
        return model_input, model_outputs

class RSTFMForward(ModelForwardWrapper):
    def brownian_sigma_t(self, t):
        sigma_max = 1.5
        return torch.sqrt((0.1 ** 2) * t**2  + (sigma_max ** 2) * (1 - t) **2)

    def brownian_g_t(self, t):
        g_t = torch.sqrt(
            torch.clip(4.5-4.52 * t, min=0)
        )
        return g_t

    def _get_rot_score_prefactor(self, t, rotmats_1, rotmats_t):
        rel_rotmat = so3_utils.rot_mult(rotmats_t.transpose(-1, -2), rotmats_1)
        omega, _, _ = so3_utils.angle_from_rotmat(rel_rotmat)
        sigma = self.brownian_sigma_t(t)
        # Generate grid of expansion orders.
        l_max = 1000
        l_grid = torch.arange(l_max + 1, device=omega.device).to(omega.dtype)
        pred_rot_score_scaling = - so3_utils.batched_dlog_igso3_expansion(
            omega, sigma, l_grid
        )
        return pred_rot_score_scaling

    def compute_base_rot_vf(self, t, rotmats_1, rotmats_t):
        rot_vf_axis = so3_utils.calc_rot_vf(rotmats_t, rotmats_1)
        score_prefactor = self._get_rot_score_prefactor(t, rotmats_1, rotmats_t)
        g_t = self.brownian_g_t(t)
        rot_vf = (0.5 * g_t ** 2 * score_prefactor)[..., None] * F.normalize(
            rot_vf_axis, dim=-1
        )
        return rot_vf

    def rot_vf_to_rot_score(self, t, rot_vf, rotmats_1, rotmats_t):
        g_t = self.brownian_g_t(t)
        rot_score = rot_vf / (0.5 * g_t**2)[..., None]
        return rot_score

    def rot_score_to_rot_vf(self, t, rot_score, rotmats_1, rotmats_t):
        g_t = self.brownian_g_t(t)
        rot_vf = (0.5 * g_t ** 2)[..., None] * rot_score
        return rot_vf

    def compute_gradient(self, model_input, aux_inputs=None, self_condition=None):
        model_outputs = self.model(
            model_input,
            self_condition=self_condition
        )
        return model_input, model_outputs

class FoldingGradient(ModelForwardWrapper):
    def compute_gradient(self, model_input, aux_inputs, self_condition=None):
        assert "folding_seq" in aux_inputs
        with torch.enable_grad():
            model_input['rigids']['rigids_t'].requires_grad = True
            model_input['rigids']['rigids_t'].grad = None

            model_outputs = self.model(
                model_input,
                self_condition=self_condition
            )
            log_p_seq = F.cross_entropy(
                model_outputs["decoded_seq_logits"],
                aux_inputs["folding_seq"],
                reduction="none"
            )
            log_p_seq.backward(gradient=torch.ones_like(log_p_seq) * model_input['token']['token_mask'])

        return model_input, model_outputs


class SequenceEntropyGradient(ModelForwardWrapper):
    def compute_gradient(self, model_input, aux_inputs, self_condition=None):
        aux_inputs = {}
        # aux_inputs['seq_entropy_mask'] = model_input['token']['seq_noising_mask']
        aux_inputs['seq_entropy_mask'] = model_input['token']['token_mask']
        # assert "seq_entropy_mask" in aux_inputs

        def maybe_detach(t):
            if isinstance(t, (torch.Tensor, Rigid)):
                return t.detach()
            else:
                return t

        model_input = map_structure(maybe_detach, model_input)
        if self_condition is not None:
            self_condition = map_structure(maybe_detach, self_condition)

        with torch.enable_grad():
            assert not torch.is_inference_mode_enabled(), "inference mode is on but we need gradient calcuations for this"
            model_input['rigids']['rigids_t'] = model_input['rigids']['rigids_t'].detach()
            model_input['rigids']['rigids_t'].requires_grad = True
            model_input['rigids']['rigids_t'].grad = None

            seq_entropy_mask = aux_inputs['seq_entropy_mask'] * model_input['token']['token_mask']

            # print(model_input['rigids']['rigids_t'])
            model_outputs = self.model(
                model_input,
                self_condition=self_condition
            )
            log_p_seq = torch.log_softmax(model_outputs["decoded_seq_logits"], dim=-1)
            p_seq = torch.exp(log_p_seq)
            H_seq = -torch.sum(log_p_seq * p_seq, dim=-1) * seq_entropy_mask
            H_seq = H_seq.sum()
            H_seq.backward()
            # H_seq.backward(gradient=torch.ones_like(H_seq) * seq_entropy_mask)

        return model_input, model_outputs


class DistogramCompactGradient(ModelForwardWrapper):
    def compute_gradient(self, model_input, aux_inputs, self_condition=None):
        def maybe_detach(t):
            if isinstance(t, (torch.Tensor, Rigid)):
                return t.detach()
            else:
                return t

        model_input = map_structure(maybe_detach, model_input)
        if self_condition is not None:
            self_condition = map_structure(maybe_detach, self_condition)

        asym_id = model_input['token']['asym_id']
        same_chain_mask = asym_id[..., None] == asym_id[..., None, :]
        rigids_to_nodes = fn.partial(gather_helper, token_gather_idx=model_input['token']['token_to_rep_rigid'])
        token_noising_mask = rigids_to_nodes(
            model_input['rigids']['rigids_noising_mask'].float()[..., None]
        ).squeeze(-1).bool()

        loss_mask = (token_noising_mask[..., None] | token_noising_mask[..., None, :])
        loss_mask = loss_mask * same_chain_mask

        with torch.enable_grad():
            assert not torch.is_inference_mode_enabled(), "inference mode is on but we need gradient calcuations for this"
            model_input['rigids']['rigids_t'] = model_input['rigids']['rigids_t'].detach()
            model_input['rigids']['rigids_t'].requires_grad = True
            model_input['rigids']['rigids_t'].grad = None

            # print(model_input['rigids']['rigids_t'])
            model_outputs = self.model(
                model_input,
                self_condition=self_condition
            )

            with torch.autocast("cuda", dtype=torch.float32):
                log_p_dist = torch.log_softmax(model_outputs["distogram_logits"], dim=-1)
                select_log_p = model_outputs['distogram_bin_upper'] < 14
                log_p_dist = log_p_dist * select_log_p
                p_dist = torch.softmax(model_outputs["distogram_logits"] -  (1 - select_log_p.float()) * 1e7, dim=-1)
                H_low_dist_bins = - torch.sum(p_dist * log_p_dist, dim=-1)
                loss = (H_low_dist_bins * loss_mask).sum()
            loss.backward()

            # H_seq.backward(gradient=torch.ones_like(H_seq) * seq_entropy_mask)

        return model_input, model_outputs


class PredLocalFafeGradient(ModelForwardWrapper):
    def compute_gradient(self, model_input, aux_inputs, self_condition=None):
        def maybe_detach(t):
            if isinstance(t, (torch.Tensor, Rigid)):
                return t.detach()
            else:
                return t

        model_input = map_structure(maybe_detach, model_input)
        if self_condition is not None:
            self_condition = map_structure(maybe_detach, self_condition)

        with torch.enable_grad():
            assert not torch.is_inference_mode_enabled(), "inference mode is on but we need gradient calcuations for this"
            model_input['rigids']['rigids_t'] = model_input['rigids']['rigids_t'].detach()
            model_input['rigids']['rigids_t'].requires_grad = True
            model_input['rigids']['rigids_t'].grad = None

            # print(model_input['rigids']['rigids_t'])
            model_outputs = self.model(
                model_input,
                self_condition=self_condition
            )
            log_p_local_trans_fafe = torch.log_softmax(model_outputs["local_trans_fafe_logits"], dim=-1)
            log_p_local_rot_fafe = torch.log_softmax(model_outputs["local_rot_fafe_logits"], dim=-1)
            log_p_local_trans_fafe = torch.logsumexp(log_p_local_trans_fafe[..., 40:], dim=-1)
            log_p_local_rot_fafe = torch.logsumexp(log_p_local_rot_fafe[..., 40:], dim=-1)
            log_p_local_fafe = 0.05 * log_p_local_trans_fafe + log_p_local_rot_fafe
            log_p_local_fafe = log_p_local_fafe * model_input['rigids']['rigids_mask']
            log_p_local_fafe = log_p_local_fafe.sum()

            loss = log_p_local_fafe * model_input['t'].view(-1)[0]
            loss.backward()

        return model_input, model_outputs