""" Generic dataset for all input data """
from typing import Dict
import logging
import os
import json

import torch
import torch.utils.data as data

from proteinzen.boltz.data import const
from proteinzen.boltz.data.types import Record, ConformerRecord


class MMCIFDataset(data.Dataset):
    def __init__(
            self,
            data_dir,
            data_sampler,
            task_sampler,
            mode,
            min_num_res=30,
            max_num_res=5000,
            min_percent_ordered=0.5,
            max_resolution=3.0,
            subset=None,
            split=None,
            exclude_mol_types=None,
            overfit_num=None,
            count_on_protein_res=False,
            use_conformer_record=True,
            interface_crop=False,
            remove_mol_types=None,
            chain_content_criteria=None,
            min_percent_protein: float | None = None,
            min_interface_size: int | None = None,
            only_valid_multichain: bool = False
        ):
        self._log = logging.getLogger(__name__)
        self.interface_crop = interface_crop
        self.data_dir = data_dir
        self.min_num_res = min_num_res
        self.max_num_res = max_num_res
        self.subset = subset
        self.split = split
        self.min_percent_ordered = min_percent_ordered
        self.max_resolution = max_resolution
        self.task_sampler = task_sampler
        self.data_sampler = data_sampler
        self.overfit_num = overfit_num

        self.use_conformer_record = use_conformer_record
        self.mode = mode
        self.data_dir = f"{self.data_dir}/{self.mode}"
        print("Initializing MMCIFDataset in mode:", mode)

        self.count_on_protein_res = count_on_protein_res
        self.min_percent_protein = min_percent_protein
        self.min_interface_size = min_interface_size
        self.only_valid_multichain = only_valid_multichain

        if self.min_interface_size:
            print("WARNING: the current implementation of min_interface_size currently counts via residue and may mark interfaces with small molecules as invalid")

        if exclude_mol_types is None:
            self.exclude_mol_types = []
        else:
            print("Excluding molecules that contain chains of types:", exclude_mol_types)
            self.exclude_mol_types = [const.chain_types.index(s) for s in exclude_mol_types]

        if remove_mol_types is None:
            self.remove_mol_types = []
        else:
            print("Removing chains which are purely composed of the following types:", remove_mol_types)
            self.remove_mol_types = [const.chain_types.index(s) for s in remove_mol_types]

        if chain_content_criteria:
            print(f"Only training on samples that have at minimum the following chain content: {chain_content_criteria}")
            self.chain_content_criteria = {
                const.chain_types.index(s): v
                for s, v in chain_content_criteria.items()
            }
        else:
            self.chain_content_criteria = {}

        self._init_metadata()

        print(f"Training on {len(self.manifest)} datapoints")

    def _init_metadata(self):
        """Initialize metadata."""
        manifest_path = os.path.join(self.data_dir, "manifest.json")

        # Process CSV with different filtering criterions.
        with open(manifest_path) as fp:
            self.raw_manifest = json.load(fp)

        self.manifest = []
        num_records = 0
        for record in self.raw_manifest:
            if 'id' not in record:
                record['id'] = record['ids'][0]
            # remove records which contain mol types we're excluding
            _exclude_record = False
            for chain in record['chains']:
                if chain['mol_type'] in self.exclude_mol_types:
                    _exclude_record = True
                    break
            if _exclude_record:
                continue
            if self.interface_crop and not record.get('interfaces'):
                continue
            # apply some filtering critera that we might change at train time
            if self.count_on_protein_res:
                num_res = sum(chain['num_residues'] for chain in record['chains'] if chain['mol_type'] == 0)
            else:
                num_res = sum(chain['num_residues'] for chain in record['chains'])
            if num_res > self.max_num_res or num_res < self.min_num_res:
                continue
            if 'structure' in record:
                resolution = record['structure']['resolution']
            elif 'structures' in record:
                resolution = record['structures'][0]['resolution']
            else:
                raise ValueError(f"Record {record['id']} has no structure or structures")

            # remove records which will be empty when we remove mol types we're removing
            _empty_record = True
            for chain in record['chains']:
                if chain['mol_type'] not in self.remove_mol_types and chain['valid']:
                    _empty_record = False
                    break
            if _empty_record:
                continue

            # mark all chains that will be masked as invalid
            for chain in record['chains']:
                if chain['mol_type'] in self.remove_mol_types:
                    chain['valid'] = False
            # mark interfaces with invalid chains as invalid
            is_chain_valid = {
                chain['chain_id']: chain['valid']
                for chain in record['chains']
            }
            for interface in record['interfaces']:
                interface_chains_are_valid = is_chain_valid[interface['chain_1']] and is_chain_valid[interface['chain_2']]
                interface['valid'] = interface['valid'] and interface_chains_are_valid

            # remove records which don't have a specified chain composition
            if self.chain_content_criteria:
                keep_mol = True
                num_chains_per_type = {
                    mol_type: 0
                    for mol_type in const.chain_type_ids.values()
                }
                for chain in record['chains']:
                    remove_chain = (chain['mol_type'] in self.remove_mol_types)
                    exclude_chain = (chain['mol_type'] in self.exclude_mol_types)
                    if not (remove_chain or exclude_chain) and chain['valid']:
                        num_chains_per_type[chain['mol_type']] += 1

                for mol_type, mol_type_criteria in self.chain_content_criteria.items():
                    if num_chains_per_type[mol_type] < mol_type_criteria:
                        keep_mol = False

                if not keep_mol:
                    continue

            # ensure a minimum protein content
            if self.min_percent_protein:
                num_protein_res = sum(chain.get('num_resolved_residues', chain['num_residues']) for chain in record['chains'] if chain['mol_type'] == const.chain_type_ids["PROTEIN"])
                num_res = sum(chain.get('num_resolved_residues', chain['num_residues']) for chain in record['chains'])
                if num_protein_res / num_res < self.min_percent_protein:
                    continue

            # ensure a minimum interface size
            if self.min_interface_size:
                # TODO: there's probably a smarter way of dealing with this wrt making sure we dont exclude small molecule interfaces
                for interface in record['interfaces']:
                    interface_size = min(interface['chain_1_num_res'], interface['chain_2_num_res'])
                    if interface_size < self.min_interface_size:
                        interface['valid'] = False

            # remove records with no valid interfaces
            if self.only_valid_multichain:
                has_valid_interface = False
                for interface in record['interfaces']:
                    if interface['valid']:
                        has_valid_interface = True
                        break
                if not has_valid_interface:
                    continue

            # apply some filtering critera that we might change at train time
            if self.count_on_protein_res:
                num_res = sum(chain.get('num_resolved_residues', chain['num_residues']) for chain in record['chains'] if chain['mol_type'] == const.chain_type_ids['PROTEIN'])
            else:
                num_res = sum(chain.get('num_resolved_residues', chain['num_residues']) for chain in record['chains'])
            if num_res > self.max_num_res or num_res < self.min_num_res:
                continue
            if 'structure' in record:
                resolution = record['structure']['resolution']
            else:
                resolution = record['structures'][0]['resolution']
            if resolution is not None and resolution > self.max_resolution:
                continue
            if self.split is not None and record['id'] not in self.split:
                continue


            record['is_af2_struct'] = record['id'].startswith("AF-")


            if self.use_conformer_record:
                self.manifest.append(ConformerRecord.from_dict(record))
            else:
                self.manifest.append(Record.from_dict(record))

            num_records += 1
            
            if self.overfit_num is not None and num_records >= self.overfit_num:
                print(f"Overfitting on {num_records} records:\n", self.manifest)
                break

        # if self.overfit_num is not None:
        #     self.manifest = self.manifest[-self.overfit_num:]
        #     print("overfit entries:", self.manifest)
            # if record['structure']['coil_percent'] > (1 - self.min_percent_ordered):
            #     continue



        if self.overfit_num is not None:
            self.manifest = self.manifest[-self.overfit_num:]
            print("overfit entries:", self.manifest)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        # Sample data example.
        record = self.manifest[idx]
        return record


class LengthDataset(data.Dataset):
    def __init__(
            self,
            lengths: Dict[int, int],
            batch_size: int = 3000,
            same_length_per_batch=False,
            normalize_cache_path=None,
            batch_by_edge_fn=None,
        ):
        self._log = logging.getLogger(__name__)
        self.lengths = lengths
        self.batch_size = batch_size
        if normalize_cache_path is not None:
            self.cache_stats = torch.load(normalize_cache_path)
        else:
            self.cache_stats = None

        self.batches = []
        self.sample_ids = []
        current_batch = []
        current_sample_ids = []
        sample_id = 0
        for sample_l, num_samples in self.lengths.items():
            for _ in range(num_samples):
                if batch_by_edge_fn is not None:
                    current_batch_size = sum([batch_by_edge_fn(l) for l in current_batch])
                else:
                    current_batch_size = sum(current_batch)
                if current_batch_size >= self.batch_size:
                    self.batches.append(current_batch)
                    self.sample_ids.append(current_sample_ids)
                    current_batch = []
                    current_sample_ids = []
                current_batch.append(sample_l)
                current_sample_ids.append(sample_id)
                sample_id += 1
            if same_length_per_batch:
                self.batches.append(current_batch)
                self.sample_ids.append(current_sample_ids)
                current_batch = []
                current_sample_ids = []

        if len(current_batch) > 0:
            self.batches.append(current_batch)
        if len(current_sample_ids) > 0:
            self.sample_ids.append(current_sample_ids)


    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        ret =  {
            "num_res": self.batches[idx],
            "sample_id": self.sample_ids[idx]
        }
        if self.cache_stats is not None:
            ret['latent_norm_mu'] = self.cache_stats['mu'].clone()
            ret['latent_norm_std'] = self.cache_stats['std'].clone()

        return ret
