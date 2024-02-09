import numpy as np
import torch
from torch import nn

import e3nn
from e3nn import o3

class SeparableS2Activation(nn.Module):
    def __init__(self, irreps, grid_res=10):
        super().__init__()
        self.irreps = irreps
        self.num_scalars = self.irreps.count("1e")
        self.scalar_act = torch.nn.SiLU()

        s2_irreps = o3.Irreps((1, (l, 1)) for l in irreps.ls)
        self.s2_act     = e3nn.nn.S2Activation(irreps=s2_irreps, act=self.scalar_act, grid_res)

        multiplicities = [irreps.count(o3.Irrep((l, 1))) for l in irreps.ls]
        assert np.all(np.array(multiplicities) == multiplicities[0])
        self.multiplicity = multiplicities[0]

    def forward(self, input_tensors):
        input_components = [input_tensors[..., s] for s in self.irreps.sort().slices()]

        input_scalars = input_components[0]
        output_scalars = self.scalar_act(input_scalars)

        input_spherical_tensors = torch.cat(
            [
                component.view(-1, self.multiplicity, 2*l+1)
                for component, l in zip(input_components, self.irreps.ls)
            ], dim=-1
        )
        output_spherical_tensors = self.s2_act(input_spherical_tensors)
        output_components = output_spherical_tensors.split(
            [2*l+1 for l in self.irreps.ls],
            dim=-1
        )
        output_tensors = torch.cat(
            [component.view(-1, self.multiplicity) for component in output_components],
            dim=-1
        )
        outputs = torch.cat(
            [output_scalars, output_tensors[..., self.num_scalars:]],
            dim=-1
        )
        return outputs


class FeedForward(nn.Module):
    def __init__(self, in_irreps, h_multiplicity, out_irreps, bypass=True):
        super().__init__()
        self.in_irreps = in_irreps
        self.h_irreps = h_multiplicity

        ls = sorted(set(in_irreps.ls) + set(out_irreps.ls))
        h_irreps = o3.Irreps([(h_multiplicity, (l, 1)) for l in ls])

        self.lin1 = o3.Linear(in_irreps, h_irreps)
        self.lin2 = o3.Linear(h_irreps, h_irreps)
        self.lin3 = o3.Linear(h_irreps, out_irreps)
        self.act = SeparableS2Activation(h_irreps, grid_res=18)

        if bypass:
            self.bypass = o3.Linear(in_irreps, out_irreps)
        self.norm = e3nn.nn.BatchNorm(out_irreps)

    def forward(self, features):
        out = self.lin1(features)
        out = self.act(out)
        out = self.lin2(out)
        out = self.act(out)
        out = self.lin3(out)

        if hasattr(self, "bypass"):
            out = self.bypass(features) + out

        return self.norm(out)