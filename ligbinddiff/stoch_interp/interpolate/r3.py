import torch
import numpy as np


def linear_interpolant(trans_0, trans_1, t):
    return (1-t)[..., None] * trans_0 + t[..., None] * trans_1


def cond_v_field(trans_0, trans_t, t, t_clip=1e-1, batch=None):
    vel = (trans_0 - trans_t) / t[..., None].clip(min=t_clip)
    if batch is None:
        center_vel = vel.mean(dim=0)[None]
    else:
        center_vel = []
        for i in range(batch.max().item() + 1):
            select = (batch == i)
            num_nodes = select.long().sum().item()
            center_vel.append(
                vel[select].mean(dim=0)[None].expand(num_nodes, -1)
            )
        center_vel = torch.cat(center_vel, dim=0)

    return vel - center_vel


def sample_prior(n_samples):
    return np.random.randn(n_samples, 3)


def step(trans_t, cond_v_field, t, dt):
    return trans_t + cond_v_field * dt
