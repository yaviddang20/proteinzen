import abc

class Biomolecule(abc.ABC):
    @abc.abstractmethod
    def sample(self):
        raise NotImplemented