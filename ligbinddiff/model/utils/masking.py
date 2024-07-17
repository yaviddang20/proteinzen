import abc
from typing import List

import torch


class MaskingUnit(abc.ABC):
    @property
    @abc.abstractmethod
    def applies_to(self):
        return NotImplemented

    @property
    @abc.abstractmethod
    def subgraph(self):
        return NotImplemented

    @abc.abstractmethod
    def gen_info_masks(self, data):
        return NotImplemented


class BackboneFrameUniformNoiser(MaskingUnit):
    _applies_to = ['res_mask', 'atom14_mask', 'chi_mask']
    _subgraph = 'residue'

    @property
    def applies_to(self):
        return self._applies_to

    @property
    def subgraph(self):
        return self._subgraph

    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def gen_info_masks(self, data):
        res_data = data['residue']
        res_mask = res_data['res_mask']
        noising_mask = (torch.rand_like(res_mask, dtype=torch.float32) < self.rate)
        atom14_noising_mask = res_data['atom14_mask'] & noising_mask[..., None]
        chi_noising_mask = res_data['chi_mask'] & noising_mask[..., None]
        return {
            "res_noising_mask": noising_mask & res_mask,
            "atom14_noising_mask": atom14_noising_mask,
            "chi_noising_mask": chi_noising_mask
        }


class SidechainUniformNoiser(MaskingUnit):
    _applies_to = ['atom14_mask', 'chi_mask']
    _subgraph = 'residue'

    @property
    def applies_to(self):
        return self._applies_to

    @property
    def subgraph(self):
        return self._subgraph

    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def gen_info_mask(self, data):
        res_data = data['residue']
        atom14_mask = res_data['atom14_mask']
        noising_mask = (torch.rand(atom14_mask.shape[0], dtype=torch.float32) < self.rate)
        atom14_noising_mask = atom14_mask.clone()
        atom14_noising_mask[noising_mask, 4:] = False
        chi_noising_mask = res_data['chi_mask'] & noising_mask[..., None]
        return {
            "atom14_noising_mask": atom14_noising_mask,
            "chi_noising_mask": chi_noising_mask
        }


class SequenceUniformNoiser(MaskingUnit):
    _applies_to = ['seq_mask', 'atom14_mask', 'chi_mask']
    _subgraph = 'residue'

    @property
    def applies_to(self):
        return self._applies_to

    @property
    def subgraph(self):
        return self._subgraph

    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def gen_info_mask(self, data):
        res_data = data['residue']
        seq_mask = res_data['seq_mask']
        noising_mask = (torch.rand_like(seq_mask.shape[0], dtype=torch.float32) < self.rate)
        atom14_noising_mask = res_data['atom14_mask'].clone()
        atom14_noising_mask[noising_mask, 4:] = False
        chi_noising_mask = res_data['chi_mask'] & noising_mask[..., None]
        return {
            "atom14_noising_mask": atom14_noising_mask,
            "chi_noising_mask": chi_noising_mask,
            "seq_noising_mask": noising_mask & seq_mask,
        }


class MaskingFactory:
    def __init__(self, maskers: List[MaskingUnit]):
        self.maskers = maskers
        self.masking_scope = set()
        for masking_unit in self.maskers:
            self.masking_scope.update([(masking_unit.subgraph, target) for target in masking_unit.applies_to])

    def apply_masks(self, data):
        for masking_unit in self.maskers:
            subgraph = masking_unit.subgraph
            info_mask_dict = masking_unit.gen_info_masks(data)
            for target, mask in info_mask_dict.items():
                if target in data[subgraph]:
                    data[subgraph][target] = data[subgraph][target] & mask
                else:
                    data[subgraph][target] = mask

        return data