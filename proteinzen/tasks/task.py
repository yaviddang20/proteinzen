import abc
from typing import Sequence, Dict, Union

import numpy as np
from torch_geometric.data import HeteroData



class Task(abc.ABC):
    loss_weight: float=1

    @abc.abstractmethod
    def process_input(self, inputs: HeteroData):
        raise NotImplementedError

    @abc.abstractmethod
    def run_eval(self, model, inputs: HeteroData):
        raise NotImplementedError

    @abc.abstractmethod
    def run_predict(self, model, inputs: HeteroData):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, inputs: HeteroData, outputs: Dict):
        raise NotImplementedError


class TaskList:
    def __init__(self, task_list: Sequence[Task], aux_losses=None):
        self.task_list = task_list
        self.aux_losses = aux_losses

    def compile_task_inputs(self, inputs: HeteroData):
        for task in self.task_list:
            inputs = task.process_input(inputs)
        return inputs

    def run_evals(self, model, inputs: HeteroData):
        outputs = {}
        for task in self.task_list:
            outputs.update(task.run_eval(model, inputs))

        return outputs

    def run_predicts(self, model, inputs: HeteroData):
        outputs = {}
        for task in self.task_list:
            outputs.update(task.run_predict(model, inputs))

        return outputs

    def compile_task_losses(self, inputs: HeteroData, outputs: Dict):
        loss = 0
        loss_dict = {}
        for task in self.task_list:
            task_loss_dict = task.compute_loss(inputs, outputs)
            loss_dict.update(task_loss_dict)
            loss += task_loss_dict['loss'] * task.loss_weight
        if self.aux_losses is not None:
            for loss_fn in self.aux_losses:
                aux_loss_dict = loss_fn(outputs)
                loss_dict.update(aux_loss_dict)
                loss += aux_loss_dict['loss'] * loss_fn.weight

        loss_dict['loss'] = loss
        return loss_dict


class TaskSampler:
    def __init__(self,
                 tasklists: Sequence[Union[TaskList, Task]],
                 task_rates: Sequence[float]):
        assert len(tasklists) == len(task_rates)
        self.num_tasks = len(tasklists)
        assert np.isclose(sum(task_rates), 1).all(), "Task rates must sum to 1"

        # if you feed in a single task, turn it into a tasklist for convenience
        self.tasklists = [
            TaskList([task]) if isinstance(task, Task) else task for task in tasklists
        ]
        self.task_rates = task_rates


    def sample_task(self):
        selected_task = np.random.choice(self.num_tasks)
        return self.tasklists[selected_task]


def single_task_sampler(tasklist):
    return TaskSampler([tasklist], [1.0])