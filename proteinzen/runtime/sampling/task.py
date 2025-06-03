"""

classes to specify sampling conditions

"""
import abc


class SamplingTask(abc.ABC):
    task_name: str = "abc"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abc.abstractmethod
    def sample_data(self):
        raise NotImplementedError

    @abc.abstractmethod
    def pad_data(self, data, n_padding):
        raise NotImplementedError