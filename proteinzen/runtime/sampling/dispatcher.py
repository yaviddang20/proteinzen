
from hydra_zen import load_from_yaml, instantiate
from torch.utils.data import Dataset, BatchSampler
from torch_geometric.data import Batch

from .task import SamplingTask
from .unconditional import UnconditionalSampling
from .unconditional_smiles import UnconditionalSamplingFromSMILES
from .unconditional_smiles import UnconditionalSamplingFromMol
from .motif_scaffolding import MotifScaffoldingTask
from .protein_pocket import ProteinPocketConditionedSampling, LigandPocketConditionedSampling



class BiomoleculeTaskDispatcher(Dataset):

    def __init__(
        self,
        tasks_yaml,
        batch_size,
        batching_mode="optimal",
        use_collate_for_pad=False,
        trans_std: float = 3,
        include_h: bool = False,
    ):
        super().__init__()
        assert batching_mode in ["lazy", "optimal"]
        self.batch_size = batch_size
        self.batching_mode = batching_mode
        self.use_collate_for_pad = use_collate_for_pad
        self.task_objs = []
        self.task_configs = []

        self.config = load_from_yaml(tasks_yaml)

        task_list = instantiate(self.config)
        for task in task_list:
            self.task_objs.append((None, task))

        self.batches = self._optimal_batching()

    def __iter__(self):
        return iter(self.batches)

    def __getitem__(self, index):
        return self.batches[index]

    def __len__(self):
        return len(self.batches)

    def _optimal_batching(self):
        all_samples = []
        for _, task in self.task_objs:
            for sample in task.sample_data():
                all_samples.append(sample)
        all_samples = sorted(all_samples, key=lambda data: data['token']['token_idx'].numel())

        # batches = []
        # current_batch = []
        # max_sample_size = 0
        # for sample in all_samples:
        #     sample_size = (sample['residue']['num_nodes']) ** 2
        #     _max_sample_size = max(sample_size, max_sample_size)

        #     if _max_sample_size * len(current_batch) <= self.batch_size:
        #         current_batch.append(sample)
        #         max_sample_size = _max_sample_size
        #     else:
        #         batches.append(current_batch)
        #         current_batch = [sample]
        #         max_sample_size = sample_size
        # if len(current_batch) > 0:
        #     batches.append(current_batch)

        return all_samples


class TaskBatchSampler:
    def __init__(
        self,
        dataset: BiomoleculeTaskDispatcher,
        batch_size: int
    ):
        self.dataset = dataset

        samples_per_task = {}
        for i, sample in enumerate(dataset.batches):
            task = sample['task']
            if task in samples_per_task:
                samples_per_task[task].append(i)
            else:
                samples_per_task[task] = [i]

        self.batch_idxs = []

        for task, task_samples in samples_per_task.items():
            batch = []
            for i in task_samples:
                if len(batch) < batch_size:
                    batch.append(i)
                else:
                    self.batch_idxs.append(batch)
                    batch = [i]
            if len(batch) > 0:
                self.batch_idxs.append(batch)

    def __iter__(self):
        for batch in self.batch_idxs:
            yield batch

    def __len__(self):
        return len(self.batch_idxs)
