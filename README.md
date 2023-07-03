ligbinddiff
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/alexjli/ligbinddiff/workflows/CI/badge.svg)](https://github.com/alexjli/ligbinddiff/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/alexjli/ligbinddiff/branch/main/graph/badge.svg)](https://codecov.io/gh/alexjli/ligbinddiff/branch/main)


Substrate-conditioned diffusion for ligand-binding protein generation

### Requirements
- pytorch (2.0.1)
- torch_geometric
- torch_cluster
- pytorch-lightning (specifically, lightning fabric)
- biopython
- jaxtyping
- hydra-zen
- pytest
- e3nn
- dgl
- [SE(3)-Transformers as implemented by NVIDIA](https://github.com/NVIDIA/DeepLearningExamples/tree/master/DGLPyTorch/DrugDiscovery/SE3Transformer)

For code-style we use
- pylint
- black
- isort
- darglint
- autoflake


### Copyright

Copyright (c) 2023, Alex J Li


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
