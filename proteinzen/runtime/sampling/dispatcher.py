
from hydra_zen import load_from_yaml
from torch.utils.data import Dataset # IterableDataset
from torch_geometric.data import Batch

from proteinzen.data.datasets.collate import collate

from .task import SamplingTask
from .unconditional import UnconditionalSampling
from .unconditional_v2 import UnconditionalSamplingV2
from .motif_scaffolding import MotifScaffoldingTask
from .motif_scaffolding_v2 import MotifScaffoldingTaskV2


class TaskDispatcher(Dataset):
    name_to_task_class = {
        "motif_scaffolding": MotifScaffoldingTask,
        "motif_scaffolding_v2": MotifScaffoldingTaskV2,
        "unconditional": UnconditionalSampling,
        "unconditional_v2": UnconditionalSamplingV2
    }

    def __init__(
        self,
        tasks_yaml,
        batch_size,
        batching_mode="optimal",
        use_collate_for_pad=False
    ):
        super().__init__()
        assert batching_mode in ["lazy", "optimal"]
        self.batch_size = batch_size
        self.batching_mode = batching_mode
        self.use_collate_for_pad = use_collate_for_pad
        self.task_objs = []
        self.task_configs = []

        self.config = load_from_yaml(tasks_yaml)

        for task_dict in self.config['tasks']:
            task_class = self.name_to_task_class[task_dict['task']]
            task = task_class(**task_dict)
            self.task_objs.append((task_dict, task))

        if batching_mode == 'optimal':
            self.batches = self._optimal_batching()
        else:
            self.batches = None

    def __iter__(self):
        if self.batching_mode == "lazy":
            for batch in self._lazy_batching():
                yield Batch.from_data_list(self._pad_batch(batch))

        elif self.batching_mode == "optimal":
            assert self.batches is not None
            for batch in self.batches:
                yield Batch.from_data_list(self._pad_batch(batch))

    def __getitem__(self, idx):
        assert self.batching_mode == "optimal"
        assert self.batches is not None
        if self.use_collate_for_pad:
            return self._pad_batch(self.batches[idx])
        else:
            return Batch.from_data_list(self._pad_batch(self.batches[idx]))

    def __len__(self):
        if self.batching_mode == 'optimal':
            return len(self.batches)
        else:
            return sum([t['num_samples'] for t, _ in self.task_objs])

    def _pad_batch(self, batch):
        if self.use_collate_for_pad:
            return collate(batch)
        else:
            max_len = max([sample['residue']['num_nodes'] for sample in batch])
            padded_batch = []
            for sample in batch:
                n_padding = max_len - sample['residue']['num_nodes']
                if n_padding > 0:
                    # TODO: this is a little clunky...
                    padded_batch.append(
                        sample.task.pad_data(sample, n_padding)
                    )
                else:
                    padded_batch.append(sample)
            return padded_batch


    def _lazy_batching(self):
        current_batch = []
        max_sample_size = 0
        for _, task in self.task_objs:
            for sample in task.sample_data():
                sample_size = (sample['residue']['num_nodes']) ** 2
                _max_sample_size = max(sample_size, max_sample_size)

                if _max_sample_size * len(current_batch) <= self.batch_size:
                    current_batch.append(sample)
                    max_sample_size = _max_sample_size
                else:
                    yield current_batch
                    current_batch = [sample]
                    max_sample_size = sample_size

        if len(current_batch) > 0:
            yield current_batch


    def _optimal_batching(self):
        all_samples = []
        for _, task in self.task_objs:
            for sample in task.sample_data():
                all_samples.append(sample)
        all_samples = sorted(all_samples, key=lambda x: x['residue']['num_nodes'])

        batches = []
        current_batch = []
        max_sample_size = 0
        for sample in all_samples:
            sample_size = (sample['residue']['num_nodes']) ** 2
            _max_sample_size = max(sample_size, max_sample_size)

            if _max_sample_size * len(current_batch) <= self.batch_size:
                current_batch.append(sample)
                max_sample_size = _max_sample_size
            else:
                batches.append(current_batch)
                current_batch = [sample]
                max_sample_size = sample_size
        if len(current_batch) > 0:
            batches.append(current_batch)

        return batches


class BiomoleculeTaskDispatcher(Dataset):
    name_to_task_class = {
        "motif_scaffolding": MotifScaffoldingTask,
        "motif_scaffolding_v2": MotifScaffoldingTaskV2,
        "unconditional": UnconditionalSampling,
        "unconditional_v2": UnconditionalSamplingV2
    }

    def __init__(
        self,
        tasks_yaml,
        batch_size,
        batching_mode="optimal",
        use_collate_for_pad=False
    ):
        super().__init__()
        assert batching_mode in ["lazy", "optimal"]
        self.batch_size = batch_size
        self.batching_mode = batching_mode
        self.use_collate_for_pad = use_collate_for_pad
        self.task_objs = []
        self.task_configs = []

        self.config = load_from_yaml(tasks_yaml)

        for task_dict in self.config['tasks']:
            task_class = self.name_to_task_class[task_dict['task']]
            task = task_class(**task_dict)
            self.task_objs.append((task_dict, task))

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