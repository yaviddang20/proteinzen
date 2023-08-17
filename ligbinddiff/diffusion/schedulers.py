""" Noising schedulers """
import abc

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist


class Scheduler(abc.ABC):
    def __init__(self, T, discrete):
        super().__init__()
        self.T = T
        self.discrete = discrete

    @abc.abstractmethod
    def alphabar(self, t):
        raise NotImplemented

    @abc.abstractmethod
    def beta(self, t):
        raise NotImplemented

    @abc.abstractmethod
    def sample_t(self):
        raise NotImplemented

    @abc.abstractmethod
    def weight(self, t):
        raise NotImplemented


class DiscreteLinearScheduler(Scheduler):
    def __init__(self,
                 beta_min=0.1,
                 beta_max=20,
                 N=1000):
        super().__init__(T=N, discrete=True)

        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.t_dist = dist.Categorical(torch.ones(N) / N)

    def alphabar(self, t):
        assert int(t) == t
        return self.alphas_cumprod[int(t)]

    def beta(self, t):
        assert int(t) == t
        return self.discrete_betas[int(t)]

    def noising_coeffs(self, t):
        assert int(t) == t
        return self.sqrt_alphas_cumprod[int(t)], self.sqrt_1m_alphas_cumprod[int(t)]

    def sample_t(self):
        return self.t_dist.sample()

    def weight(self, t):
        assert int(t) == t
        return 0.5 / self.discrete_betas[int(t)]


class PositiveLinear(nn.Module):
    def forward(self, weight):
        return torch.abs(weight)


class LearnedScheduler(Scheduler, nn.Module):
    def __init__(self,
                 gamma_0=-7,
                 gamma_1=13.5,
                 h_dim=1024):
        super().__init__(T=1, discrete=False)

        self.t_dist = dist.Uniform(0, self.T)

        ## beta schedule parameters
        self.l1 = nn.Linear(1, 1)
        self.l2 = nn.Linear(1, h_dim)
        self.l3 = nn.Linear(h_dim, 1)
        nn.utils.parametrize.register_parametrization(self.l1, "weight", PositiveLinear())
        nn.utils.parametrize.register_parametrization(self.l1, "bias", PositiveLinear())
        nn.utils.parametrize.register_parametrization(self.l2, "weight", PositiveLinear())
        nn.utils.parametrize.register_parametrization(self.l2, "bias", PositiveLinear())
        nn.utils.parametrize.register_parametrization(self.l3, "weight", PositiveLinear())
        nn.utils.parametrize.register_parametrization(self.l3, "bias", PositiveLinear())

        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1

    def _gamma_tilde(self, t):
        ret = self.l1(t) + self.l3(F.sigmoid(self.l2(self.l1(t))))  # B x 1
        return ret

    def _gamma(self, t):
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * (
                self._gamma_tilde(t) - self._gamma_tilde(torch.zeros_like(t))
            ) / (
                self._gamma_tilde(torch.ones_like(t)) - self._gamma_tilde(torch.zeros_like(t))
            )
        return gamma  # B x 1

    def snr(self, t):
        return torch.expm1(-self._gamma(t)) + 1 # B x 1

    def snr_derivative(self, t):
        snr_t, tau_t = torch.func.jvp(self.snr, (t,), (torch.ones_like(t),)) # B x B
        return snr_t, tau_t  # B x 1, B x 1

    def alphabar(self, t):
        return F.sigmoid(-self._gamma(t))  # B x 1

    def beta(self, t):
        _, tau_t = self.snr_derivative(t)
        alphabar_t = self.alphabar(t)
        beta_t = -tau_t * (1 - alphabar_t) ** 2
        return beta_t

    def sample_t(self):
        return self.t_dist.sample()

    def weight(self, t):
        _, tau_t = torch.func.jvp(self.snr, (t,), (torch.ones_like(t),)) # B x B
        return -0.5 * tau_t  # B x 1, B x 1
