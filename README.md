ProteinZen
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/alexjli/proteinzen/workflows/CI/badge.svg)](https://github.com/alexjli/proteinzen/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/alexjli/proteinzen/branch/main/graph/badge.svg)](https://codecov.io/gh/alexjli/proteinzen/branch/main)

This repo contains the code for ProteinZen as described in "All-atom protein generation via SE(3) flow matching with ProteinZen".

> [!NOTE]
> This repo and ProteinZen are still under active development. Please reach out or open an issue if you run into any problems, and be on the lookout for future releases!

## Installation
To install, first set up a fresh conda environment
```
conda create -n proteinzen python=3.10
```
then run `bash install_pt26.sh`. Alternatively, we provide an `environment.yaml` which you can install via `conda env create -f environment.yaml`.
Then activate the environment and install ProteinZen via `pip install -e .`.

Finally, update `env_vars.sh` to what is correct for your setup. `REPO_ROOT` should point to the root of this repo, and `ENV_NAME`
should match the name of the conda environment you created (if it is `proteinzen` you won't need to change anything).

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
    out_prefix=<output_folder> \
    sampler.tasks_yaml=<task_yaml_path> \
    sampler.batch_size=<batch_size> \
```
where
- `model_dir` specifies the path to a model checkpoint folder
- `out_prefix` specifies a path to the output folder
- `samples.tasks_yaml` specifies a path to a sampling task YAML file
- `sampler.batch_size` controls the batch size used at inference.
By default ProteinZen will use all visible GPUs. To restrict this behavior, modify `CUDA_VISIBLE_DEVICES` to specify the GPUs you'd like to use.
Additional flags can be found in `proteinzen/runtime/config.py` within `config_sampling_hydra_store()`.
Example sampling scripts can be found in `scripts/submit_scripts/eval/`.
### Task YAML schema
Task YAMLs generally follow the following format:
```yaml
tasks:
- task: <name of task 1>
  # parameters for task 1
- task: <name of task 2>
  # parameters for task 2
...
- task: <name of task n>
  # parameters for task n
```
where the `task` field specifies what kind of sampling task you'd like to do. Internally, ProteinZen constructs task objects that correspond to these
Currently the supported tasks are
- `motif_scaffolding` (which points to `proteinzen.runtime.sampling.motif_scaffolding.MotifScaffoldingTask`)
- `unconditional` (which points to `proteinzen.runtime.sampling.unconditional.UnconditionalSampling`)
To see what parameters are necessary to specify for each, please see the docstrings for these respective files.
Example task YAML files can also be found in `configs/sampling/`.

## Known issues
### Multi-chain behavior
The published set of weights for ProteinZen was trained on monomers only, and hence we do not expect any multi-chain behavior to function properly.
This includes scaffolding multi-chain motifs: we have observed that ProteinZen currently puts all input motif fragments in the same chain
regardless of fragment chain assignment.

### Motif scaffolding
Input motifs for both indexed and unindexed motif scaffolding are given to ProteinZen
in the form of additional conditioning tokens, and at the end of the denoising trajectory
the input motif is grafted back into the scaffold based on a motif assignment process.
Although rare, we do observe that ProteinZen sometimes produces designs
where we cannot properly graft the motif back into the structure, often either in the form of
(a) not maintaining the chain constraint within a motif fragment, or
(b) assigning multiple scaffold residues to the same motif residue.
These problems occurs more often in unindexed mode
(~0.8% of designs when running the Protpardelle benchmark)
than in indexed mode (~0.08% of designs when running the Protpardelle benchmark).
While we work on reducing this error rate, we include a script
`scripts/analysis/sanitize_motif_idx.py` which will detect and output which samples have such pathologies.


## Acknowledgements
We would like to thank the various open-source repos which have made this work possible!
- FrameDiff
- FrameFlow
- FoldFlow
- OpenFold
- Boltz

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.

## Copyright

Copyright (c) 2025, Alex J Li

