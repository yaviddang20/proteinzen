import math
from proteinzen.model.modules.equiformer_v2.activation import SeparableS2Activation
from proteinzen.model.modules.equiformer_v2.layer_norm import EquivariantRMSNormArraySphericalHarmonicsV2 as NormSO3
from proteinzen.model.modules.equiformer_v2.so3 import SO3_Grid
from proteinzen.model.modules.openfold.layers import Linear
from proteinzen.utils.openfold import rigid_utils as ru


import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelwiseVectorGateUpdate(nn.Module):
    def __init__(self, c_s, c_v, c_hidden):
        super().__init__()

        self.c_s = c_s
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.mlp = nn.Sequential(
            Linear(self.c_s + self.c_v, self.c_hidden, init="relu"),
            nn.ReLU(),
            Linear(self.c_hidden, self.c_hidden, init="relu"),
            nn.ReLU(),
            Linear(self.c_hidden, self.c_s + self.c_v, init="final")
        )

    def forward(self, f_s, f_v, r):
        f_loc = torch.cat([
            f_s,
            torch.linalg.vector_norm(f_v, dim=-1)
        ], dim=-1)
        f_s_v_gate = self.mlp(f_loc)
        f_s, f_v_gate = f_s_v_gate.split([self.c_s, self.c_v], dim=-1)
        f_v = f_v * F.sigmoid(f_v_gate)[..., None]
        return f_s, f_v



class LocalFrameUpdate(nn.Module):
    def __init__(self, c_s, c_v, c_hidden):
        super().__init__()

        self.c_s = c_s
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.mlp = nn.Sequential(
            Linear(self.c_s + self.c_v * 4, self.c_hidden, init="relu"),
            nn.ReLU(),
            Linear(self.c_hidden, self.c_hidden, init="relu"),
            nn.ReLU(),
            Linear(self.c_hidden, self.c_s + 3 * self.c_v, init="final")
        )

    def forward(self, f_s, f_v, r):
        f_vloc = r[..., None].invert_apply(f_v)
        f_loc = torch.cat([
            f_s,
            f_vloc.flatten(start_dim=-2, end_dim=-1),
            torch.linalg.vector_norm(f_v, dim=-1)
        ], dim=-1)
        f_s_vloc = self.mlp(f_loc)
        f_s, f_vloc = f_s_vloc.split([self.c_s, 3*self.c_v], dim=-1)
        f_v = r[..., None].apply(f_vloc.view([-1, self.c_v, 3]))
        return f_s, f_v


class BackboneUpdateVectorBias(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s, c_v):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super().__init__()

        self.c_s = c_s
        self.c_v = c_v

        self.linear_s = Linear(self.c_s, 3, init="final")
        self.linear_v = Linear(self.c_v, 1, init="final")

    def forward(self, s: torch.Tensor, v: torch.Tensor, r: ru.Rigid):
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        # [*, 6]
        update_s = self.linear_s(s)
        update_v = self.linear_v(v.transpose(-1, -2)).squeeze(-1)
        update_v = r.get_rots().invert_apply(update_v)
        update = torch.cat([update_s, update_v], dim=-1)

        return update


class Lmax1_LinearV2(torch.nn.Module):
    def __init__(self, in_features, out_features):
        '''
            1. Use `torch.einsum` to prevent slicing and concatenation
            2. Need to specify some behaviors in `no_weight_decay` and weight initialization.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = 1

        self.weight = torch.nn.Parameter(torch.randn((self.lmax + 1), out_features, in_features))
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

        expand_index = torch.zeros([(self.lmax + 1) ** 2]).long()
        for l in range(self.lmax + 1):
            start_idx = l ** 2
            length = 2 * l + 1
            expand_index[start_idx : (start_idx + length)] = l
        self.register_buffer('expand_index', expand_index)


    def forward(self, input_embedding):

        weight = torch.index_select(self.weight, dim=0, index=self.expand_index) # [(L_max + 1) ** 2, C_out, C_in]
        out = torch.einsum('bmi, moi -> bmo', input_embedding, weight) # [N, (L_max + 1) ** 2, C_out]
        bias = self.bias.view(1, 1, self.out_features)
        out[:, 0:1, :] = out.narrow(1, 0, 1) + bias

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax})"


class VectorLayerNorm(nn.Module):
    def __init__(self, c_v):
        super().__init__()
        self.c_v = c_v
        self.ln_v = NormSO3(lmax=1, num_channels=self.c_v)

    def forward(self, v):
        in_v = v.transpose(-1, -2)
        v_dummy_scalars = torch.zeros([in_v.shape[0], 1, self.c_v], device=in_v.device)
        out_v = self.ln_v(
            torch.cat([v_dummy_scalars, in_v], dim=-2)
        )[..., 1:, :].transpose(-1, -2)
        return out_v


class NodeTransition(nn.Module):
    def __init__(self, c_s, c_v):
        super().__init__()

        self.c_s = c_s
        self.c_v = c_v
        self.c_hidden = max(c_s, c_v)

        lmax = 1

        self.embed_s = Linear(self.c_s, self.c_hidden)
        self.embed_v = Linear(self.c_v, self.c_hidden, bias=False)

        self.linear2 = Lmax1_LinearV2(self.c_hidden, self.c_hidden)

        self.act = SeparableS2Activation(1, 1)
        self.SO3_grid_list = nn.ModuleList()
        for l in range(lmax+1):
            SO3_m_grid = nn.ModuleList()
            for m in range(lmax+1):
                SO3_m_grid.append(
                    SO3_Grid(l, m)
                )
            self.SO3_grid_list.append(SO3_m_grid)


        self.out_s = Linear(self.c_hidden, self.c_s, init='final')
        self.out_v = Linear(self.c_hidden, self.c_v, bias=False, init='final')

        self.ln_s = nn.LayerNorm(self.c_s)
        self.ln_v = VectorLayerNorm(self.c_v)

    def forward(self, s, v):

        # "linear 1"
        embed_s = self.embed_s(s)
        embed_v = self.embed_v(v.transpose(-1, -2))
        embedding = torch.cat([embed_s[..., None, :], embed_v], dim=-2)

        embedding = self.act(embedding[:, 0], embedding, self.SO3_grid_list)
        embedding = self.linear2(embedding)
        embedding = self.act(embedding[:, 0], embedding, self.SO3_grid_list)

        # "linear 3"
        out_s = self.out_s(embedding[:, 0])
        out_v = self.out_v(embedding[:, 1:])

        out_s = self.ln_s(s + out_s)
        out_v = v + out_v.transpose(-1, -2)
        out_v = self.ln_v(out_v)

        return out_s, out_v
