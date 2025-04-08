
from hydra_zen import load_from_yaml
from torch.utils.data import IterableDataset
from torch_geometric.data import Batch

from .task import SamplingTask
from .unconditional import UnconditionalSampling
from .motif_scaffolding import MotifScaffoldingTask


class TaskDispatcher(IterableDataset):
    name_to_task_class = {
        "motif_scaffolding": MotifScaffoldingTask,
        "unconditional": UnconditionalSampling
    }

    def __init__(
        self,
        tasks_yaml,
        batch_size,
        batching_mode="lazy"
    ):
        super().__init__()
        assert batching_mode in ["lazy", "optimal"]
        self.batch_size = batch_size
        self.batching_mode = batching_mode
        self.task_objs = []
        self.task_configs = []

        self.config = load_from_yaml(tasks_yaml)

        for task_dict in self.config['tasks']:
            task_class = self.name_to_task_class[task_dict['task']]
            task = task_class(**task_dict)
            self.task_objs.append((task_dict, task))

    def __iter__(self):
        if self.batching_mode == "lazy":
            for batch in self._lazy_batching():
                yield Batch.from_data_list(self._pad_batch(batch))

        elif self.batching_mode == "optimal":
            raise NotImplementedError()

    def __len__(self):
        return sum([t['num_samples'] for t, _ in self.task_objs])

    def _pad_batch(self, batch):
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