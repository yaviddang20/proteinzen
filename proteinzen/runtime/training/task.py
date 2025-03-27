import abc
from typing import Tuple, List
import numpy as np
from torch import Tensor


class TrainingTask(abc.ABC):
    name: str = "abc"  # override this
    def sample_t_and_mask(self, batch) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError


class TaskSampler:
    def __init__(
        self,
        task_list: List[TrainingTask],
        task_probs: List[float]
    ):
        assert np.isclose(sum(task_probs), 1.0), sum(task_probs)
        keep = [(p > 0.0) for p in task_probs]

        self.task_list = [task for i, task in enumerate(task_list) if keep[i]]
        self.task_probs = [task for i, task in enumerate(task_probs) if keep[i]]

    def sample_task(self) -> TrainingTask:
        choice = np.random.choice(a=len(self.task_probs), p=self.task_probs)
        return self.task_list[choice]