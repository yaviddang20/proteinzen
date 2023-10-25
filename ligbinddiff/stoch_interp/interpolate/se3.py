from typing import Optional
import numpy as np
from . import so3_utils
from . import so3, r3
from scipy.spatial.transform import Rotation
from ligbinddiff.utils.openfold import rigid_utils as ru
from .so3_flow_matching import SO3ConditionalFlowMatcher
import torch

def _extract_trans_rots(rigid: ru.Rigid):
    rot = rigid.get_rots().get_rot_mats().cpu().numpy()
    rot_shape = rot.shape
    num_rots = np.cumprod(rot_shape[:-2])[-1]
    rot = rot.reshape((num_rots, 3, 3))
    rot = Rotation.from_matrix(rot).as_rotvec().reshape(rot_shape[:-2] +(3,))
    tran = rigid.get_trans().cpu().numpy()
    return tran, rot

def _torch_extract_trans_rots(rigid: ru.Rigid):
    rot = rigid.get_rots().get_rot_mats()
    rot = so3_utils.Log(rot)
    tran = rigid.get_trans()
    return tran, rot

def _assemble_rigid(rotvec, trans):
    rotvec_shape = rotvec.shape
    num_rotvecs = np.cumprod(rotvec_shape[:-1])[-1]
    rotvec = rotvec.reshape((num_rotvecs, 3))
    rotmat = Rotation.from_rotvec(rotvec).as_matrix().reshape(
        rotvec_shape[:-1] + (3, 3))
    return ru.Rigid(
            rots=ru.Rotation(
                rot_mats=torch.Tensor(rotmat)),
            trans=torch.tensor(trans))

def _torch_assemble_rigid(rotvec, trans):
    rotmat = so3_utils.Exp(rotvec)
    return ru.Rigid(
        rots=ru.Rotation(rot_mats=rotmat),
        trans=trans
    )


class SE3FlowMatcher:
    def __init__(self,
                 trans_noise_scale=10,
                 diffuse_rot=True,
                 diffuse_trans=True,
                 ):
        self.trans_noise_scale = trans_noise_scale
        self.so3_cfm = SO3ConditionalFlowMatcher()
        self.diffuse_rot = diffuse_rot
        self.diffuse_trans = diffuse_trans

    def forward_marginal(
            self,
            rigids_0: ru.Rigid,
            t: torch.Tensor,
            diffuse_mask: Optional[torch.Tensor] = None,
            as_tensor_7: bool=True,
            batch=None,
        ):
        """
        Args:
            rigids_0: [..., N] openfold Rigid objects
            t: continuous time in [0, 1].

        Returns:
            rigids_t: [..., N] noised rigid. [..., N, 7] if as_tensor_7 is true.
            trans_score: [..., N, 3] translation score
            rot_score: [..., N, 3] rotation score
            trans_score_norm: [...] translation score norm
            rot_score_norm: [...] rotation score norm
        """
        if batch is not None:
            center = ru.batchwise_center(rigids_0, batch)
            rigids_0 = rigids_0.translate(-center)

        t = torch.ones(rigids_0.shape[0]) * t

        trans_0 = rigids_0.get_trans()
        # rot_0 = rigids_0.get_rots()
        rot_0 = rigids_0.get_rots().get_rot_mats().float()
        rot_1, trans_1 = self.sample_ref(
            rigids_0.shape[0],
            impute=rigids_0,
            diffuse_mask=diffuse_mask)
        trans_1 = trans_1.to(trans_0.device)
        rot_1 = rot_1.to(rot_0.device, dtype=rot_0.dtype)

        if self.diffuse_rot:
            # rot_t = so3.geodesic_interpolant(rot_0, rot_1, t)
            # rot_cond_v = so3.cond_v_field(rot_t, rot_0, t)
            rot_t = self.so3_cfm.sample_xt(rot_0, rot_1, t, if_matrix_format=True)
            rot_t = torch.as_tensor(rot_t).float()
            rot_cond_v = self.so3_cfm.compute_conditional_flow(rot_t, rot_0, t)
            # print(rot_t.shape, rot_cond_v.shape)
        else:
            rot_t = rot_0.clone()
            rot_cond_v = torch.zeros_like(rot_0)

        if self.diffuse_trans:
            trans_t = r3.linear_interpolant(trans_0, trans_1, t)
            trans_cond_v = r3.cond_v_field(trans_0, trans_1, torch.ones_like(t))
        else:
            trans_t = trans_0.clone()
            trans_cond_v = torch.zeros_like(trans_0)

        if diffuse_mask is not None:
            rot_t = self._apply_mask(
                rot_t, rot_0, diffuse_mask[..., None])
            trans_t = self._apply_mask(
                trans_t, trans_0, diffuse_mask[..., None])

            trans_cond_v = self._apply_mask(
                trans_cond_v,
                torch.zeros_like(trans_cond_v),
                diffuse_mask[..., None])
            rot_cond_v = self._apply_mask(
                rot_cond_v,
                torch.zeros_like(rot_cond_v),
                diffuse_mask[..., None])

        rigids_t = ru.Rigid(rots=ru.Rotation(rot_mats=rot_t), trans=trans_t)

        if batch is not None:
            rigids_t = rigids_t.translate(center)

        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
        return {
            'rigids_t': rigids_t,
            'trans_cond_v': trans_cond_v,
            'rot_cond_v': rot_cond_v,
        }

    def calc_trans_cond_v(self, trans_t, trans_0, t, batch):
        return r3.cond_v_field(trans_0, trans_t, t, batch=batch, t_clip=0)

    def calc_rot_cond_v(self, rots_t, rots_0, t):
        # return so3.cond_v_field(rots_t, rots_0, t)
        return self.so3_cfm.compute_conditional_flow(rots_t.get_rot_mats(), rots_0.get_rot_mats(), t)

    def _apply_mask(self, x_diff, x_fixed, diff_mask):
        return diff_mask * x_diff + (1 - diff_mask) * x_fixed

    def reverse(
            self,
            rigid_t: ru.Rigid,
            rot_cond_v: np.ndarray,
            trans_cond_v: np.ndarray,
            t: float,
            dt: float,
            diffuse_mask: np.ndarray = None,
            anneal=10,
        ):
        """Reverse sampling function from (t) to (t-1).

        Args:
            rigid_t: [..., N] protein rigid objects at time t.
            rot_score: [..., N, 3] rotation score.
            trans_score: [..., N, 3] translation score.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: [..., N] which residues to update.
            center: true to set center of mass to zero after step

        Returns:
            rigid_t_1: [..., N] protein rigid objects at time t-1.
        """
        rot_t = rigid_t.get_rots()
        trans_t = rigid_t.get_trans()

        if self.diffuse_rot:
            # rot_t_1 = so3.step(
            #     rot_t,
            #     rot_cond_v,
            #     t,
            #     dt
            #     )
            if anneal is None:
                anneal = 1
            else:
                anneal = anneal * t
            rot_t_1 = so3_utils.expmap(rot_t.get_rot_mats(), rot_cond_v * dt * anneal)
            rot_t_1 = ru.Rotation(rot_mats=rot_t_1)
        else:
            rot_t_1 = rot_t

        if self.diffuse_trans:
            trans_t_1 = r3.step(
                trans_t,
                trans_cond_v,
                t=t,
                dt=dt
                )
        else:
            trans_t_1 = trans_t

        if diffuse_mask is not None:
            trans_t_1 = self._apply_mask(
                trans_t_1, trans_t, diffuse_mask[..., None])
            rot_t = Rotation.from_matrix(rot_t.get_rot_mats().numpy(force=True)).as_rotvec()
            rot_t_1 = Rotation.from_matrix(rot_t_1.get_rot_mats().numpy(force=True)).as_rotvec()
            rot_t_1 = self._apply_mask(
                rot_t_1, rot_t, diffuse_mask[..., None])
            return _assemble_rigid(rot_t_1, trans_t_1)
        else:
            return ru.Rigid(rots=rot_t_1, trans=trans_t_1)

    def sample_ref(
            self,
            n_samples: int,
            impute: Optional[ru.Rigid]=None,
            diffuse_mask: Optional[torch.Tensor]=None,
            as_tensor_7: bool=False,
            align_trans_noise=True,
            couple_rot=False
        ):
        """Samples rigids from reference distribution.

        Args:
            n_samples: Number of samples.
            impute: Rigid objects to use as imputation values if either
                translations or rotations are not diffused.
        """
        rot_ref = so3.sample_uniform(
            n_samples=n_samples).numpy(force=True)
        trans_ref = r3.sample_prior(
            n_samples=n_samples
        )
        trans_ref = trans_ref * self.trans_noise_scale

        if impute is not None:
            assert impute.shape[0] == n_samples
            trans_impute, rot_impute = _extract_trans_rots(impute)
            trans_impute = trans_impute.reshape((n_samples, 3))
            rot_impute = rot_impute.reshape((n_samples, 3))
            if align_trans_noise:
                kabsh_rot, _ = Rotation.align_vectors(
                    trans_impute,
                    trans_ref,
                    diffuse_mask
                )
                trans_ref = kabsh_rot.apply(trans_ref)

        if diffuse_mask is not None:
            assert impute is not None
            if self.diffuse_rot:
                rot_ref = self._apply_mask(
                    rot_ref, rot_impute, diffuse_mask[..., None])
            else:
                rot_ref = rot_impute
            if self.diffuse_trans:
                trans_ref = self._apply_mask(
                    trans_ref, trans_impute, diffuse_mask[..., None])
            else:
                trans_ref = trans_impute

        rotvec = rot_ref
        rotvec_shape = rotvec.shape
        num_rotvecs = np.cumprod(rotvec_shape[:-1])[-1]
        rotvec = rotvec.reshape((num_rotvecs, 3))
        rotmat = Rotation.from_rotvec(rotvec).as_matrix().reshape(
            rotvec_shape[:-1] + (3, 3))
        # rot_ref = ru.Rotation(rot_mats=torch.as_tensor(rotmat))
        rot_ref = torch.as_tensor(rotmat).float()


        return rot_ref, torch.as_tensor(trans_ref)
