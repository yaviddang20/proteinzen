import torch
import numpy as np

from ligbinddiff.utils.openfold import rigid_utils as ru

from . import so3_utils
from . import so3_helpers


def exp_r_tangent(r_tangent: ru.Rotation, r: torch.Tensor) -> ru.Rotation:
    """ Compute the Riemannian exp via parallel transport tricks """
    r_tangent_inv = r_tangent.invert()
    r_rel = r_tangent_inv.compose_r(ru.Rotation(rot_mats=r))
    r_rel_SO3 = so3_utils.exp(r_rel.get_rot_mats())
    r_rel_SO3 = ru.Rotation(rot_mats=r_rel_SO3)
    ret = r_tangent.compose_r(r_rel_SO3)
    return ret


def log_r_tangent(r_tangent: ru.Rotation, r: ru.Rotation) -> torch.Tensor:
    """ Compute the Riemannian log via parallel transport tricks """
    r_tangent_inv = r_tangent.invert()
    r_rel = r_tangent_inv.compose_r(r)
    r_rel_so3 = so3_utils.log(r_rel.get_rot_mats())
    ret = r_tangent.compose_r(ru.Rotation(rot_mats=r_rel_so3))
    return ret.get_rot_mats()


def geodesic_interpolant(r0: ru.Rotation, r1: ru.Rotation, t: torch.Tensor) -> ru.Rotation:
    log_r0_r1 = log_r_tangent(r0, r1)
    return exp_r_tangent(r0, t[..., None, None] * log_r0_r1)


def cond_v_field(rt: ru.Rotation, r0: ru.Rotation, t, t_clip=1e-3):
    return log_r_tangent(rt, r0) / t[..., None, None].clip(min=t_clip)


def sample_uniform(n_samples, M=1000):
    N = n_samples
    omega_grid = np.linspace(0, np.pi, M)
    cdf = np.cumsum(np.pi**-1 * (1-np.cos(omega_grid)), 0)/(M/np.pi)
    omegas = np.interp(np.random.rand(N), cdf, omega_grid)
    axes = np.random.randn(N, 3)
    axes = omegas[..., None]* axes/np.linalg.norm(axes, axis=-1, keepdims=True)
    return torch.as_tensor(axes)


def step(rot_t, cond_v_field, t, dt, anneal=10):
    if anneal is None:
        anneal = 1
    else:
        anneal = anneal * t
    rot_dt = exp_r_tangent(rot_t, dt * cond_v_field * anneal)
    rot_t_m_dt = rot_t.compose_r(rot_dt) #rot_dt.compose_r(rot_t)
    # rot_t_m_dt = rot_dt.compose_r(rot_t)
    return rot_t_m_dt
