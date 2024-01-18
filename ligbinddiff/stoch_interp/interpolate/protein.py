from .latent import LatentInterpolant
from .se3 import SE3Interpolant, SE3InterpolantConfig

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
