import torch
from torch_geometric.data import HeteroData

from ligbinddiff.model.utils.graph import batchwise_to_nodewise



def diagonalize(N, edges=[], antiedges=[], a=1, b=0.3, lamb=0., ptr=None):
    J = torch.zeros((N, N), device=edges.device)  # temporary fix
    for i, j in edges:
        J[i, i] += a
        J[j, j] += a
        J[i, j] = J[j, i] = -a
    for i, j in antiedges:
        J[i, i] -= b
        J[j, j] -= b
        J[i, j] = J[j, i] = b
    J += torch.diag(lamb)
    if ptr is None:
        return torch.linalg.eigh(J)

    Ds, Ps = [], []
    for start, end in zip(ptr[:-1], ptr[1:]):
        D, P = torch.linalg.eigh(J[start:end, start:end])
        Ds.append(D)
        Ps.append(P)
    return torch.cat(Ds), torch.block_diag(*Ps)

@torch.no_grad()
def sample_harmonic_prior(num_nodes, edge_index, sigma=1, ptr=None):
    edge_index = edge_index[:, edge_index[0] < edge_index[1]]
    lamb = 1 / sigma ** 2
    D, P = diagonalize(
        N=num_nodes,
        edges=edge_index.T,
        ptr=ptr,
        lamb=lamb
    )
    noise = torch.randn((num_nodes, 3), device=edge_index.device)
    prior = P @ (noise / torch.sqrt(D)[:, None])
    return prior


def _diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (~diffuse_mask[..., None])


class HarmonicPriorInterpolant:
    def __init__(self,
                 min_t=1e-2):
        self.min_t = min_t

    def set_device(self, device):
        self._device = device

    def _corrupt_pos(self, pos_1, t, edge_index, pos_mask, ptr=None):
        pos_0 = sample_harmonic_prior(
            pos_1.shape[0],
            edge_index,
            ptr=ptr)
        pos_t = (1 - t[..., None]) * pos_0 + t[..., None] * pos_1
        pos_t = _diffuse_mask(pos_t, pos_1, pos_mask)
        return pos_t * pos_mask[..., None]

    def sample_t(self, num_batch):
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2 * self.min_t) + self.min_t

    @torch.no_grad()
    def corrupt_batch(self, batch: HeteroData):
        atom_data = batch["ligand"]

        # [N]
        atom_mask = atom_data["res_mask"]
        noising_mask = atom_data["noising_mask"]
        mask = atom_mask & noising_mask

        # [B]
        if "t" in batch.keys():
            t = batch["t"]
        else:
            t = self.sample_t(batch.num_graphs)
            batch["t"] = t
        nodewise_t = batchwise_to_nodewise(t, atom_data.batch)

        # Apply corruptions
        atom_pos = atom_data['atom_pos']
        noised_atoms = self._corrupt_pos(atom_pos, nodewise_t, mask, atom_data.ptr)

        return {
            "noised_atom_pos": noised_atoms
        }

    def _euler_step(self, d_t, t, x_1, x_t):
        x_vf = (x_1 - x_t) / (1 - t)
        return x_t + x_vf * d_t