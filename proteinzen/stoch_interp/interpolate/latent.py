import tqdm
import torch
from . import so3_utils
from . import utils as du
from scipy.spatial.transform import Rotation
import copy
from scipy.optimize import linear_sum_assignment

from proteinzen.utils.framediff import all_atom
from proteinzen.utils.openfold import rigid_utils as ru
from proteinzen.model.utils.graph import batchwise_to_nodewise

from torch_geometric.data import HeteroData, Batch
from torch_geometric.utils import scatter


def _centered_gaussian(batch, dim_size, device):
    noise = torch.randn(batch.shape[0], dim_size, device=device)
    center = scatter(
        noise,
        index=batch,
        dim=0,
        reduce='mean'
    )
    return noise - center[batch]


def _diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (~diffuse_mask[..., None])


class LatentInterpolant:
    def __init__(self,
                 min_t=1e-2,
                 self_condition=True,
                 dim_size=128):
        self.min_t = min_t
        self.self_condition = self_condition
        self.dim_size = dim_size

    def set_device(self, device):
        self._device = device

    def _corrupt_x(self, x_1, t, res_mask, batch):
        x_0 = torch.randn_like(x_1)
        x_t = (1 - t[..., None]) * x_0 + t[..., None] * x_1
        x_t = _diffuse_mask(x_t, x_1, res_mask)
        return x_t * res_mask[..., None]

    @torch.no_grad()
    def corrupt_batch(self, batch: HeteroData, intermediates):
        res_data = batch["residue"]

        # [N]
        res_mask = res_data["res_mask"]
        noising_mask = res_data["noising_mask"]
        mask = res_mask & noising_mask

        # [B]
        t = batch["t"]
        nodewise_t = batchwise_to_nodewise(t, res_data.batch)

        # Apply corruptions
        latent_sidechain = intermediates['latent_sidechain']
        noised_latent = self._corrupt_x(latent_sidechain, nodewise_t, mask, res_data.batch)

        return {
            "noised_latent_sidechain": noised_latent
        }

    def _euler_step(self, d_t, t, x_1, x_t):
        x_vf = (x_1 - x_t) / (1 - t)
        return x_t + x_vf * d_t


class DenseLatentInterpolant:
    def __init__(self,
                 min_t=1e-2):
        self.min_t = min_t

    def _corrupt_x(self, x_1, t, res_mask):
        t = t.view(-1, *[1 for _ in range(x_1.dim()-1)])
        x_0 = torch.randn_like(x_1)
        x_t = (1 - t) * x_0 + t * x_1
        x_t = _diffuse_mask(x_t, x_1, res_mask)
        return x_t * res_mask[..., None]

    @torch.no_grad()
    def corrupt_batch(self, batch: HeteroData, intermediates):
        res_data = batch["residue"]

        # [N]
        res_mask = res_data["res_mask"]
        noising_mask = res_data["noising_mask"]
        mask = res_mask & noising_mask
        dense_mask = mask.view(batch.num_graphs, -1)
        edge_mask = dense_mask[..., None] & dense_mask[..., None, :]

        # [B]
        t = batch["t"]
        nodewise_t = t[res_data.batch]
        # Apply corruptions
        latent_sidechain = intermediates['latent_sidechain']
        latent_edge = intermediates['latent_edge']
        noised_latent = self._corrupt_x(latent_sidechain, t, dense_mask)
        noised_edge = self._corrupt_x(latent_edge, t, edge_mask)

        return {
            "noised_latent_sidechain": noised_latent,
            "noised_latent_edge": noised_edge
        }

    def _euler_step(self, d_t, t, x_1, x_t):
        x_vf = (x_1 - x_t) / (1 - t)
        return x_t + x_vf * d_t