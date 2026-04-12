from dataclasses import replace
from typing import Iterator, List

from numpy.random import RandomState

from proteinzen.boltz.data.types import Record, ConformerRecord
from proteinzen.boltz.data.sample.sampler import Sample, Sampler

import numpy as np


class RandomSampler(Sampler):
    """A simple random sampler with replacement."""

    def sample(self, records: List[Record], random: RandomState) -> Iterator[Sample]:
        """Sample a structure from the dataset infinitely.

        Parameters
        ----------
        records : List[Record]
            The records to sample from.
        random : RandomState
            The random state for reproducibility.

        Yields
        ------
        Sample
            A data sample.

        """
        while True:
            # Sample item from the list
            index = random.randint(0, len(records))
            record = records[index]

            # Remove invalid chains and interfaces
            chains = [c for c in record.chains if c.valid]
            interfaces = [i for i in record.interfaces if i.valid]
            record = replace(record, chains=chains, interfaces=interfaces)

            yield Sample(record=record)

class ConformerSampler(Sampler):
    """A simple random sampler with replacement."""

    def sample(self, records: List[ConformerRecord], random: RandomState) -> Iterator[Sample]:
        """Sample a structure from the dataset infinitely.

        Parameters
        ----------
        records : List[Record]
            The records to sample from.
        random : RandomState
            The random state for reproducibility.

        Yields
        ------
        Sample
            A data sample.

        """
        while True:
            # Sample item from the list
            index = random.randint(0, len(records))
            conformer_record = records[index]

            boltzmann_weights = conformer_record.boltzmann_weights

            w = np.asarray(boltzmann_weights, dtype=np.float64)
            w = np.clip(w, 0.0, None)

            Z = w.sum()

            p = w / Z

            conformer_index = np.random.choice(len(boltzmann_weights), p=p)

            structure = conformer_record.structures[conformer_index]
            conformer_id = conformer_record.ids[conformer_index]

            record = Record(
                id=conformer_id,
                structure=structure,
                chains=conformer_record.chains,
                interfaces=conformer_record.interfaces,
                inference_options=conformer_record.inference_options,
                templates=conformer_record.templates,
                md=conformer_record.md,
                affinity=conformer_record.affinity,
            )


            # Remove invalid chains and interfaces
            chains = [c for c in record.chains if c.valid]
            interfaces = [i for i in record.interfaces if i.valid]
            record = replace(record, chains=chains, interfaces=interfaces)

            yield Sample(record=record, e_min=conformer_record.e_min)
