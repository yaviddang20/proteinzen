import abc
from typing import Sequence, Dict, Union

import numpy as np
from torch_geometric.data import HeteroData


class TrainingHarness(abc.ABC):

    @abc.abstractmethod
    def process_input(self, inputs: HeteroData):
        raise NotImplementedError

    @abc.abstractmethod
    def run_eval(self, model, inputs: HeteroData):
        raise NotImplementedError

    @abc.abstractmethod
    def run_predict(self, model, inputs: HeteroData, device: str):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, inputs: HeteroData, outputs: Dict):
        raise NotImplementedError
