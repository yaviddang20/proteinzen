""" Train a model """

import logging
import tqdm

import torch
import torch_geometric.utils as pygu

from proteinzen.stoch_interp.interpolate.se3 import SE3InterpolantConfig
from proteinzen.data.datasets.datamodule import FramediffDataModule
from proteinzen.tasks.task import single_task_sampler
from proteinzen.tasks.fm.atom10 import ProteinAtom10Interpolant, ProteinAtom10Interpolation
from proteinzen.openfold.utils.rigid_utils import Rigid


# A logger for this file
log = logging.getLogger(__name__)


if __name__ == '__main__':
    device = "cuda:0"
    task_sampler = single_task_sampler(
        ProteinAtom10Interpolation(
            ProteinAtom10Interpolant(SE3InterpolantConfig())
        )
    )
    datamodule_inst = FramediffDataModule(
        task_sampler=task_sampler,
        data_dir="/wynton/home/kortemme/alexjli/projects/proteinzen/data/afdb_128",
        batch_size=10000,
        num_workers=4
    )

    running_var = 0
    running_mu = 0
    total_num_res = 0

    for batch in tqdm.tqdm(datamodule_inst.train_dataloader()):
        batch = batch.to(device)
        res_data = batch['residue']
        atom14 = res_data['atom14']
        atom14_mask = res_data['atom14_mask']
        rigids = Rigid.from_tensor_7(res_data['rigids_1'])
        atom14 = rigids[..., None].invert_apply(atom14)
        atom14 *= atom14_mask[..., None]
        num_res = torch.sum(atom14_mask, dim=0)[..., None]
        # atoms = atom14[res_data['res_mask'], 4:, :]
        # atoms = atom14[atom14_mask.bool()]
        # var, mu = torch.var_mean(atoms, dim=0, correction=0)
        # var = var.double()
        # mu = mu.double()

        var, mu = [], []
        for i in range(14):
            atoms = atom14[:, i][atom14_mask.bool()[:, i]]
            _var, _mu = torch.var_mean(atoms, dim=0, correction=0)
            var.append(_var.double())
            mu.append(_mu.double())

        var = torch.stack(var)
        mu = torch.stack(mu)

        running_var = (running_var * total_num_res + var * num_res) / (total_num_res + num_res)
        running_mu = (running_mu * total_num_res + mu * num_res) / (total_num_res + num_res)
        total_num_res = total_num_res + num_res
        print(running_var, running_mu, total_num_res)


