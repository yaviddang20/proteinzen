import abc
from typing import Tuple, List, Dict
import numpy as np
from torch import Tensor


class TrainingTask(abc.ABC):
    name: str = "abc"  # override this
    prob: float

    def sample_t_and_mask(self, batch) -> Dict[str, Tensor]:
        raise NotImplementedError


class TaskSampler:
    def __init__(
        self,
        task_list: List[TrainingTask],
    ):
        task_probs = [t.prob for t in task_list]
        assert np.isclose(sum(task_probs), 1.0), sum(task_probs)

        self.task_list = task_list
        self.task_probs = task_probs

    def sample_task(self) -> TrainingTask:
        choice = np.random.choice(a=len(self.task_probs), p=self.task_probs)
        return self.task_list[choice]