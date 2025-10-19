ProteinZen
==============================
[//]: # (Badges)
<!-- [![GitHub Actions Build Status](https://github.com/alexjli/proteinzen/workflows/CI/badge.svg)](https://github.com/alexjli/proteinzen/actions?query=workflow%3ACI) -->
<!-- [//]: # [![codecov](https://codecov.io/gh/alexjli/proteinzen/branch/main/graph/badge.svg)](https://codecov.io/gh/alexjli/proteinzen/branch/main) -->

This repo contains the code for ProteinZen as described in "All-atom protein generation via SE(3) flow matching with ProteinZen".

> [!NOTE]
> This repo and ProteinZen are still under active development. Please reach out or open an issue if you run into any problems, and be on the lookout for future releases!

## Installation
To install, first set up a fresh conda environment
```bash
conda create -n proteinzen python=3.10
conda activate proteinzen
bash install_pt26.sh
```
Alternatively, we provide an `environment.yaml` which you can install via
```bash
conda env create -f environment.yaml
conda activate proteinzen
pip install -e .
```

Finally, update `env_vars.sh` to what is correct for your setup. `REPO_ROOT` should point to the root of this repo, and `ENV_NAME`
should match the name of the conda environment you created (if it is `proteinzen` you won't need to change anything).

## Pretrained weights

We're still working on a more permanent home for data! For now, pretrained weights can be downloaded at
[https://ucsf.box.com/s/nzvilhj9w8zmck5xm1c6fy4c0ztmx1ql](https://ucsf.box.com/s/nzvilhj9w8zmck5xm1c6fy4c0ztmx1ql).

## Training

### Dataset preprocessing
See `scripts/data/README.md` for instructions on how to preprocess data.

### Training
To train ProteinZen, run
```bash
python train.py \
    domain=protein \
    paradigm=multiframefm \
    datamodule.batch_size=<batch_size_per_gpu> \
    datamodule.num_workers=<num_cpu_workers> \
    dataset.config=<path_to_dataset_config> \
    experiment.lightning.devices=<num_gpus> \
    experiment.lightning.strategy=ddp_find_unused_parameters_true \
```
Additional flags can be found in `proteinzen/runtime/config.py` within `config_hydra_store()`.
Example training scripts can be found in `scripts/submit_scripts/train/` and example dataset configs can be found in `configs/train/data/`.

## Sampling
To sample from ProteinZen, run
```bash
python sample.py \
    model_dir=<model_dir> \
    out_dir=<output_folder> \
    sampler.tasks_yaml=<task_yaml_path> \
    sampler.batch_size=<batch_size> \
```
where
- `model_dir` specifies the path to a model checkpoint folder. If you extract `proteinzen_weights` to `REPO_ROOT`,
  then this would be `${REPO_ROOT}/proteinzen_weights/unconditional_model` for unconditional sampling
  and `${REPO_ROOT}/proteinzen_weights/motif_scaffolding_model` for motif scaffolding
- `out_dir` specifies a path to the output folder
- `samples.tasks_yaml` specifies a path to a sampling task YAML file
- `sampler.batch_size` controls the batch size used at inference.

By default ProteinZen will use all visible GPUs. To restrict this behavior, modify `CUDA_VISIBLE_DEVICES` to specify the GPUs you'd like to use.
Additional flags can be found in `proteinzen/runtime/config.py` within `config_sampling_hydra_store()`. Some potentially useful options include:
- `save_traj=<bool, default=False>` which specifies whether to dump per-sample denoising trajectories.
  Note that this currently significantly slows down inference!
- `corrupter.num_timesteps=<int, default=400>` which controls the number of discritization steps to use to integrate
- `corrupter.sampling_noise_mode=<null or "churn", default="churn">` which specifies whether to perform ODE or SDE sampling, respectively

Sampling output directories will generally have the following structure:
```
└── <sampling_run>/
    ├── metadata/
    │   ├── samples_metadata_rank0_batch0.json
    │   └── ...
    ├── samples/
    │   ├── len_100_protein_id0_0_0.pdb
    │   └── ...
    ├── traj/
    │   ├── len_100_protein_id0_0_0_clean_traj.pdb
    │   ├── len_100_protein_id0_0_0_prot_traj.pdb
    │   └── ...
    ├── config.yaml
    ├── pmpnn_fixed_pos_dict.jsonl
    └── samples_metadata.json
```
where
- `config.yaml` is a copy of the tasks YAML file fed in as input
- `pmpnn_fixed_pos_dict.jsonl` is a dictionary formatted for use with ProteinMPNN to specify residues to not redesign the sequence of
- `samples_metadata.json` contains various per-sample metadata information, including
  - `path`: path to the sample
  - `name`: the name of the task used to generate this sample
  - `fixed_bb_res_idx`: list of residue indices whose backbone positions were prespecified (e.g. backbone of input motif)
  - `fixed_bb_chain`: the chain IDs associated with the residues in `fixed_bb_res_idx`
  - `fixed_seq_res_idx`: list of residue indices whose sequence identities were prespecified (e.g. sequence of input motif)
  - `fixed_seq_chain`: the chain IDs associated with the residues in `fixed_seq_res_idx`
- `metadata/` stores per-batch metadata jsons
- `samples/` stores the generated samples in .pdb format
- `traj/` stores trajectories if `save_traj=True`
  - `*_clean_traj.pdb` contains the denoiser predictions over the course of integration
  - `*_prot_traj.pdb` contains the per-timestep state `x_t` over the course of integration

Example sampling scripts can be found in `scripts/submit_scripts/eval/`, which include evaluation pipelines
involving sequence redesign via ProteinMPNN, structure prediction via ESMFold, and RMSD calculations.

### Task YAML schema
Task YAMLs generally follow the following format:
```yaml
tasks:
- task: <name of task 1>
  # parameters for task 1
- task: <name of task 2>
...
  # parameters for task 2
- task: <name of task n>
  # parameters for task n
```
where the `task` field specifies what kind of sampling task you'd like to do. Internally, ProteinZen constructs task objects that correspond to these.
Currently the supported tasks are
- `motif_scaffolding` (which points to `proteinzen.runtime.sampling.motif_scaffolding.MotifScaffoldingTask`)
- `unconditional` (which points to `proteinzen.runtime.sampling.unconditional.UnconditionalSampling`)

Below we briefly describe how to run each of these tasks.
To see detailed documentation for each task, please see the docstrings for these respective files. Example task YAML files can also be found in `configs/sampling/`.

#### Unconditional generation
```yaml
- task: unconditional
  sample_length: <sample length, type=int>
  num_samples: <number of samples at specified sample length, type=int>
```
Unconditional generation task objects have two possible parameters: `sample_length` which specifies how large of a protein to generate,
and `num_samples` which is how many samples to generate of that length. Both will be parsed as ints.
You can optionally specify a string in the `name` field which will be assigned in the metadata for all samples generated from that task.
#### Motif scaffolding
```yaml
- task: motif_scaffolding
  pdb_contigs: <comma separated list specifying residues to scaffold. type=str>
  pdb: <path to source pdb file, type=str>
  num_samples: <number of samples to generate, type=int>
  total_length: <total length of protein to generate, type=str>
  sample_contigs_idx_config: <an RFDiffusion-stype contigs string to sample indices from, type=str, default=None>
  sample_chain_name: <the chain id to sample residues from. type=str, default='A'>
  redesign_contigs: <which residues to redesign the sequence identity of. type=str, default=None>
```
Motif scaffolding tasks have the above set of parameters.
- `pdb_contigs`: comma-separated list which specifies which residues to scaffold. Each entry can specify single residues (e.g. `A35`) or a range of residues (e.g. `A35-40`).
  - Example: `A35,A40-55,A12`.
- `pdb`: path to the motif pdb file.
- `num_samples`: number of scaffolds to generate.
- `total_length`: string specifying the total length of the scaffolds to generate. Can be a single length e.g `50` or a range of lengths from which the scaffold length will be uniformly sampled from e.g. `50-100`.
- `sample_contigs_idx_config`: comma-separated list specifying how to sample the indices and spacing of the motif and scaffold. This format largely follows the RFDiffusion-style contig strings
  where the length of spacers are specified by either a single number (e.g. `50`) or a range of numbers (e.g. `50-100`), while a motif can be specified by either a single residues (e.g. `A35`) or a range of residues (e.g. `A35-50`).
  - Example: `"0-10,A35,10-25,A40-55,10,A12,5-20"`
- `sample_chain_name`: chain ID of the scaffold to generate. This should match the chain ID of the motif in `pdb_contigs`. Defaults to `A`.
- `redesign_contigs`: commay-separated list of residues to redesign the sequence of. Each entry can specify single residues (e.g. `A35`) or a range of residues (e.g. `A35-40`).
  - Example: `A35,A40-55,A12`.

Additionally, you can optionally specify a string in the `name` field which will be assigned in the metadata for all samples generated from that task.


## Known issues
### Multi-chain behavior
The current published set of weights for ProteinZen were trained on monomers only, and hence we do not expect any multi-chain behavior to function properly.
This includes scaffolding multi-chain motifs: we have observed that ProteinZen currently puts all input motif fragments in the same chain
regardless of fragment chain assignment. This will fixed once we have multichain weights.

### Motif scaffolding
Input motifs for both indexed and unindexed motif scaffolding are given to ProteinZen
in the form of additional conditioning tokens, and at the end of the denoising trajectory
the input motif is grafted back into the scaffold based on a motif assignment process.
Although rare, we do observe that ProteinZen sometimes produces designs
where we cannot properly graft the motif back into the structure,
often by assigning multiple scaffold residues to the same motif residue.
This problem occurs more often in unindexed mode
(~0.8% of designs when running the Protpardelle benchmark)
than in indexed mode (~0.08% of designs when running the Protpardelle benchmark).
While we work on reducing this error rate, we include a script
`scripts/analysis/sanitize_motif_idx.py` which will detect and output which samples have such pathologies.


## Acknowledgements
We would like to thank the various open-source repos which have made this work possible!
- FrameDiff: [https://github.com/jasonkyuyim/se3_diffusion](https://github.com/jasonkyuyim/se3_diffusion)
- FrameFlow: [https://github.com/microsoft/protein-frame-flow](https://github.com/microsoft/protein-frame-flow)
- FoldFlow: [https://github.com/DreamFold/FoldFlow](https://github.com/DreamFold/FoldFlow)
- OpenFold: [https://github.com/aqlaboratory/openfold](https://github.com/aqlaboratory/openfold)
- Boltz: [https://github.com/jwohlwend/boltz](https://github.com/jwohlwend/boltz)

All functions adapted from the repos above should be attributed within the code, but if any attributions are
missing please let us know and we will correct it!

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.

## Copyright

Copyright (c) 2025, Alex J Li

