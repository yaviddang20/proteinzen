"""Optimizers

Currently only includes the Noam optimizer,
based on https://github.com/jingraham/neurips19-graph-protein-design
"""

import re
import torch

class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self, **kwargs):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step(**kwargs)

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        optimizer_state = self.optimizer.state_dict()
        return_state = {
            'step': self._step,
            'warmup': self.warmup,
            'factor': self.factor,
            'model_size': self.model_size,
            'rate': self._rate,
            'optimizer_state': optimizer_state
        }
        return return_state

    def load_state_dict(self, state_dict):
        self._step = state_dict['step']
        self.warmup = state_dict['warmup']
        self.factor = state_dict['factor']
        self.model_size = state_dict['model_size']
        self._rate = state_dict['rate']
        self.optimizer.load_state_dict(state_dict['optimizer_state'])

    def __getattr__(self, name):
        return getattr(self.optimizer, name)


def make_adam(model, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def get_std_opt(model, d_model, state=None):
    optim = NoamOpt(d_model, 2, 4000,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    if state is not None:
        optim.load_state_dict(state)
    return optim


# nn.Embedding tables — must go to Adam, not Muon
_EMBEDDING_RE = re.compile(r'rigid_idx_embed|rigid_element_embed|lin_token_bonds')


class MuonWithAdam(torch.optim.Optimizer):
    def __init__(self, muon_params, adam_params, lr_muon=1e-3, lr_adam=1e-4,
                 wd_muon=0.1, wd_adam=0.0, momentum=0.95, nesterov=True,
                 betas_adam=(0.9, 0.999), eps_adam=1e-8):
        self.muon = torch.optim.Muon(muon_params, lr=lr_muon, weight_decay=wd_muon,
                                     momentum=momentum, nesterov=nesterov)
        self.adam = torch.optim.Adam(adam_params, lr=lr_adam, betas=betas_adam,
                                     eps=eps_adam, weight_decay=wd_adam)
        # bypass Optimizer.__init__ and set required attributes directly
        self.defaults = {}
        self.state = {}
        self.param_groups = self.muon.param_groups + self.adam.param_groups

    def step(self, closure=None):
        self.muon.step(closure)
        self.adam.step(closure)

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {'muon': self.muon.state_dict(), 'adam': self.adam.state_dict()}

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict['muon'])
        self.adam.load_state_dict(state_dict['adam'])


def make_muon(model, lr_muon=1e-3, lr_adam=1e-4,
              wd_muon=0.1, wd_adam=0.0,
              momentum=0.95, nesterov=True,
              betas_adam=(0.9, 0.999), eps_adam=1e-8):
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and not _EMBEDDING_RE.search(name):
            muon_params.append(p)
        else:
            adam_params.append(p)

    return MuonWithAdam(muon_params, adam_params, lr_muon=lr_muon, lr_adam=lr_adam,
                        wd_muon=wd_muon, wd_adam=wd_adam, momentum=momentum,
                        nesterov=nesterov, betas_adam=betas_adam, eps_adam=eps_adam)