from .latent import LatentInterpolant
from .se3 import SE3Interpolant, SE3InterpolantConfig
from .dirichlet import DirichletConditionalFlow
from .sidechain_torsion import SidechainTorsionInterpolant, SidechainMultiTorsionInterpolant

class ProteinInterpolant:
    """ Wrapper for SE3Interpolant and LatentInterpolant """
    def __init__(self,
                 se3_cfg: SE3InterpolantConfig,
                 use_batch_ot=False,
                 latent_dim_size=128):
        self._cfg = se3_cfg
        self.se3_noiser = SE3Interpolant(
            se3_cfg,
            use_batch_ot=use_batch_ot)
        self.sidechain_noiser = LatentInterpolant(
            min_t=se3_cfg.min_t,
            self_condition=se3_cfg.self_condition,
            dim_size=latent_dim_size)


class ProteinDirichletInterpolant:
    """ Wrapper for SE3Interpolant and DirichletInterpolant """
    def __init__(self,
                 se3_cfg: SE3InterpolantConfig,
                 use_batch_ot=False,
                 dirichlet_t_max=8):
        self._cfg = se3_cfg
        self.se3_noiser = SE3Interpolant(
            se3_cfg,
            use_batch_ot=use_batch_ot)
        self.sidechain_noiser = DirichletConditionalFlow(
            t_max=dirichlet_t_max
        )


class ProteinDirichletChiInterpolant:
    """ Wrapper for SE3Interpolant and DirichletInterpolant """
    def __init__(self,
                 se3_cfg: SE3InterpolantConfig,
                 use_batch_ot=False,
                 dirichlet_t_max=8,
                 use_uniform_chi_noise=False,
                 chi_noise_sigma=1.5):
        self._cfg = se3_cfg
        self.se3_noiser = SE3Interpolant(
            se3_cfg,
            use_batch_ot=use_batch_ot)
        self.sidechain_noiser = DirichletConditionalFlow(
            t_max=dirichlet_t_max
        )
        self.chi_noiser = SidechainTorsionInterpolant(
            uniform_rot_noise=use_uniform_chi_noise,
            sigma=chi_noise_sigma
        )


class ProteinDirichletMultiChiInterpolant:
    """ Wrapper for SE3Interpolant and DirichletInterpolant """
    def __init__(self,
                 se3_cfg: SE3InterpolantConfig,
                 use_batch_ot=False,
                 dirichlet_t_max=8,
                 use_uniform_chi_noise=False,
                 chi_noise_sigma=1.5):
        self._cfg = se3_cfg
        self.se3_noiser = SE3Interpolant(
            se3_cfg,
            use_batch_ot=use_batch_ot)
        self.sidechain_noiser = DirichletConditionalFlow(
            t_max=dirichlet_t_max
        )
        self.chi_noiser = SidechainMultiTorsionInterpolant(
            uniform_rot_noise=use_uniform_chi_noise,
            sigma=chi_noise_sigma
        )
