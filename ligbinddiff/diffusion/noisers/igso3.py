"""
Operations re: IGSO(3)

Taken from or adapted from FrameDiff, https://github.com/jasonkyuyim/se3_diffusion/blob/master/data/so3_diffuser.py and
https://github.com/jasonkyuyim/se3_diffusion/blob/unsupported_refactor/data/igso3.py """

import torch
import numpy as np

from . import so3_utils


def f_igso3(omega, t, L=500):
    """Truncated sum of IGSO(3) distribution.

    This function approximates the power series in equation 5 of
    "DENOISING DIFFUSION PROBABILISTIC MODELS ON SO(3) FOR ROTATIONAL
    ALIGNMENT"
    Leach et al. 2022

    This expression diverges from the expression in Leach in that here, sigma =
    sqrt(2) * eps, if eps_leach were the scale parameter of the IGSO(3).

    With this reparameterization, IGSO(3) agrees with the Brownian motion on
    SO(3) with t=sigma^2 when defined for the canonical inner product on SO3,
    <u, v>_SO3 = Trace(u v^T)/2

    Args:
        omega: i.e. the angle of rotation associated with rotation matrix
        t: variance parameter of IGSO(3), maps onto time in Brownian motion
        L: Truncation level
    """

    ls = torch.arange(L)  # of shape [1, L]
    approx = ((2*ls + 1) * torch.exp(-ls*(ls+1)*t/2) *
         torch.sin(omega.unsqueeze(-1)*(ls+1/2)) / torch.sin(omega.unsqueeze(-1)/2)).sum(dim=-1)
    return approx


def d_logf_d_omega(omega, t, L=500):
    omega = torch.tensor(omega, requires_grad=True)
    log_f = torch.log(f_igso3(omega, t, L))
    out = torch.autograd.grad(log_f.sum(), omega)[0]
    return out

# IGSO3 density with respect to the volume form on SO(3)
def igso3_density(Rt, t, L=500):
    return f_igso3(so3_utils.Omega(Rt), t, L)

def igso3_density_angle(omega, t, L=500):
    return f_igso3(torch.tensor(omega), t, L).numpy()*(1-np.cos(omega))/np.pi

# grad_R log IGSO3(R; I_3, t)
def igso3_score(R, t, L=500):
    omega = so3_utils.Omega(R)
    unit_vector = torch.einsum('...ij,...jk->...ik', R, so3_utils.log(R))/omega[:, None, None]
    return unit_vector * d_logf_d_omega(omega, t, L)[:, None, None]


def calculate_igso3(*, num_ts=1000, num_omegas=1000, min_t=0.01, max_t=4., L=500):
    """calculate_igso3 pre-computes numerical approximations to the IGSO3 cdfs
    and score norms and expected squared score norms.

    Args:
        num_ts: number of different ts for which to compute igso3
            quantities.
        num_omegas: number of point in the discretization in the angle of
            rotation.
        min_t, max_t: the upper and lower ranges for the angle of
            rotation on which to consider the IGSO3 distribution.  This cannot
            be too low or it will create numerical instability.
    """
    # Discretize omegas for calculating CDFs. Skip omega=0.
    discrete_omegas = np.linspace(0, np.pi, num_omegas+1)[1:]

    # Exponential separation of sigmas.
    discrete_ts = 10 ** np.linspace(np.log10(min_t), np.log10(max_t), num_ts)

    # Compute the pdf and cdf values for the marginal distribution of the angle
    # of rotation (which is needed for sampling)
    pdf_vals = np.asarray(
        [igso3_density_angle(discrete_omegas, t) for t in discrete_ts])
    pdf_vol_form_vals = np.asarray(
        [f_igso3(torch.tensor(discrete_omegas), t).numpy() for t in discrete_ts])
    cdf_vals = np.asarray(
        [pdf.cumsum() / num_omegas * np.pi for pdf in pdf_vals])

    # Compute the norms of the scores.  This are used to scale the rotation axis when
    # computing the score as a vector.
    d_logf_d_omega_val = np.asarray(
        [d_logf_d_omega(discrete_omegas, t).numpy() for t in discrete_ts])

    return {
        'cdf': cdf_vals, # CDF for angle of rotation -- for sampling
        'pdf_angle': pdf_vals, # PDF for angle of rotation
        'pdf': pdf_vol_form_vals, # PDF for w.r.t. volume form
        'd_logf_d_omega': d_logf_d_omega_val,
        'discrete_omegas': discrete_omegas,
        'discrete_ts': discrete_ts,
    }


class IGSO3:
    def __init__(self, min_t=0.02, max_t=4., L=500, num_ts=500, num_omegas=1000):
        self.min_t = min_t
        self.max_t = max_t
        self.num_ts = num_ts
        self.num_omegas = num_omegas

        # Precompute IGSO3 values.
        print('Computing IGSO3.')
        igso3_vals = calculate_igso3(
            num_ts=num_ts, num_omegas=num_omegas,min_t=min_t, max_t=max_t, L=L)

        igso3_vals = {key: torch.as_tensor(val) for key,val in igso3_vals.items()}

        self._cdf = igso3_vals['cdf']
        self._pdf = igso3_vals['pdf']
        self._pdf_angle = igso3_vals['pdf_angle']
        self._d_logf_d_omega = igso3_vals['d_logf_d_omega']
        self._discrete_ts = igso3_vals['discrete_ts']
        self._discrete_omegas = igso3_vals['discrete_omegas']

        self._argmin_omega_for_d_logf_d_omega = self._discrete_omegas[torch.argmin(self._d_logf_d_omega, dim=1)]

    def t_idx(self, t: torch.Tensor):
        """Calculates the index for discretized t during IGSO(3) initialization."""
        return torch.bucketize(t, self._discrete_ts.to(t.device)) - 1

    def argmin_omega_for_d_logf_d_omega(self, t):
        return self._argmin_omega_for_d_logf_d_omega[self.t_idx(t)]

    def pdf_wrt_uniform(self, R, t):
        omegas = so3_utils.Omega(R)
        return np.interp(omegas, self._discrete_omegas, self._pdf[self.t_idx(t)])

    def sample_angle(self, t: torch.Tensor):
        """Uses the inverse cdf to sample an angle of rotation from IGSO(3).

        Args:
            t: time of Brownian motion
            n_samples: number of samples to draw.

        Returns: [n_samples] angles of rotation.
        """
        unique_t = torch.unique(t).tolist()

        out = np.zeros_like(t.numpy(force=True))
        for _t in unique_t:
            select = (t == _t)
            n_samples = select.long().sum()
            select = select.numpy(force=True)
            x = np.random.rand(n_samples)
            angles = np.interp(x, self._cdf[self.t_idx(torch.tensor(_t))], self._discrete_omegas)
            out[select] = angles

        return out

    def sample(self, t: torch.Tensor):
        """Generates rotation vector(s) from IGSO(3).

        Args:
            t: continuous time in [0, 1].
            n_sample: number of samples to generate.

        Returns:
            [n_samples, 3, 3] rotation matrix sampled from IGSO(3) as torch
                tensor
        """
        n_samples = t.numel()
        x = np.random.randn(n_samples, 3)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        rot_vecs = x * self.sample_angle(t)[:, None]
        rot_vecs = torch.tensor(rot_vecs).float()
        return so3_utils.exp(so3_utils.hat(rot_vecs)).to(t.device)

    def d_logf_d_omega(self, omega: torch.Tensor, t: torch.Tensor):
        omega = omega.numpy(force=True)
        unique_t = torch.unique(t).tolist()

        out = np.zeros_like(omega)
        for _t in unique_t:
            select = (t == _t).numpy(force=True)
            subset_omega = omega[select]
            diff = np.interp(subset_omega, self._discrete_omegas, self._d_logf_d_omega[self.t_idx(torch.tensor(_t))])
            out[select] = diff
        return torch.as_tensor(out, device=t.device)

    def score(self, R: torch.Tensor, t: torch.Tensor, eps: float=1e-6):
        """Computes the score of IGSO(3) density as a rotation vector.

        Same as score function but uses pytorch and performs a look-up.

        Args:
            R: [..., 3, 3] array of rotation matrices in SO(3)
            t: time for Brownian motion
            eps: for stability when dividing by something small

        Returns:
            [..., 3, 3] score vector in the direction of the sampled vector with
            magnitude given by _d_logf_d_omega.
        """
        device = R.device
        omega = so3_utils.Omega(R)
        d_logf_d_omega = self.d_logf_d_omega(omega, t)

        # Unit vector in tangent space
        direction = torch.einsum('...jk,...kl->...jl',
                R, so3_utils.log(R)/(omega[..., None, None] + eps))

        return direction * d_logf_d_omega[..., None, None]
