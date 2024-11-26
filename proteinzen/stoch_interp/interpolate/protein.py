from .atom10 import Atom10Interpolant
from .latent import LatentInterpolant, DenseLatentInterpolant
from .se3 import SE3Interpolant, SE3InterpolantConfig
from .fisher import FisherFlow
from .dirichlet import DirichletConditionalFlow
from .catflow import CatFlow
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


class DenseProteinInterpolant:
    """ Wrapper for SE3Interpolant and LatentInterpolant """
    def __init__(self,
                 se3_cfg: SE3InterpolantConfig,
                 use_batch_ot=False):
        self._cfg = se3_cfg
        self.se3_noiser = SE3Interpolant(
            se3_cfg,
            use_batch_ot=use_batch_ot)
        self.latent_noiser = DenseLatentInterpolant(
            min_t=se3_cfg.min_t)


class ProteinDirichletInterpolant:
    """ Wrapper for SE3Interpolant and DirichletInterpolant """
    def __init__(self,
                 se3_cfg: SE3InterpolantConfig,
                 use_batch_ot=False,
                 dirichlet_t_max=8,
                 dirichlet_polyn_coeff=1):
        self._cfg = se3_cfg
        self.se3_noiser = SE3Interpolant(
            se3_cfg,
            use_batch_ot=use_batch_ot)
        self.sidechain_noiser = DirichletConditionalFlow(
            t_max=dirichlet_t_max,
            polyn_sched_coeff=dirichlet_polyn_coeff
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


class ProteinFisherInterpolant:
    """ Wrapper for SE3Interpolant and FisherInterpolant """
    def __init__(self,
                 se3_cfg: SE3InterpolantConfig,
                 use_batch_ot=False,
                 prior='hypersphere',
                 train_sched="linear",
                 train_c=1,
                 sample_sched="linear",
                 sample_c=1,
                 dirichlet_bridge_concentration=None):
        self._cfg = se3_cfg
        self.se3_noiser = SE3Interpolant(
            se3_cfg,
            use_batch_ot=use_batch_ot)
        self.sidechain_noiser = FisherFlow(
            prior=prior,
            train_sched=train_sched,
            train_c=train_c,
            sample_sched=sample_sched,
            sample_c=sample_c,
            dirichlet_bridge_conc=dirichlet_bridge_concentration
        )


class ProteinFisherMultiChiInterpolant:
    """ Wrapper for SE3Interpolant and DirichletInterpolant """
    def __init__(self,
                 se3_cfg: SE3InterpolantConfig,
                 use_batch_ot=False,
                 use_uniform_chi_noise=False,
                 chi_noise_sigma=1.5):
        self._cfg = se3_cfg
        self.se3_noiser = SE3Interpolant(
            se3_cfg,
            use_batch_ot=use_batch_ot)
        self.sidechain_noiser = FisherFlow(
            prior="hypersphere"
        )
        self.chi_noiser = SidechainMultiTorsionInterpolant(
            uniform_rot_noise=use_uniform_chi_noise,
            sigma=chi_noise_sigma
        )



class ProteinCatFlowInterpolant:
    """ Wrapper for SE3Interpolant and FisherInterpolant """
    def __init__(self,
                 se3_cfg: SE3InterpolantConfig,
                 use_batch_ot=False,
    ):
        self._cfg = se3_cfg
        self.se3_noiser = SE3Interpolant(
            se3_cfg,
            use_batch_ot=use_batch_ot)
        self.sidechain_noiser = CatFlow()


class ProteinAtom10Interpolant:
    """ Wrapper for SE3Interpolant and Atom10 """
    def __init__(self,
                 se3_cfg: SE3InterpolantConfig,
                 use_batch_ot=False,
                 atom10_one_m_exp_c=None,
                 atom10_sigmoid_c=None,
                 atom10_emperical_mean_offset=False,
                 atom10_smarter_prior=False,
                 atom10_smarter_prior_std=3,
                 atom10_prior_std=1,
                 atom10_nonlocal_prior=False,
                 trans_preconditioning=False,
                 trans_preconditioning_std=16
    ):
        self._cfg = se3_cfg
        self.se3_noiser = SE3Interpolant(
            se3_cfg,
            use_batch_ot=use_batch_ot,
            trans_preconditioning=trans_preconditioning,
            trans_preconditioning_std=trans_preconditioning_std
        )
        self.sidechain_noiser = Atom10Interpolant(
            one_m_exp_c=atom10_one_m_exp_c,
            sigmoid_c=atom10_sigmoid_c,
            emperical_mean_offset=atom10_emperical_mean_offset,
            smarter_prior=atom10_smarter_prior,
            smarter_prior_std=atom10_smarter_prior_std,
            prior_std=atom10_prior_std,
            nonlocal_prior=atom10_nonlocal_prior
        )