# conda env create -n proteinzen
# conda activate proteinzen
pip install torch==2.6 torchvision torchaudio "numpy<1.25"
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install lightning torchmetrics pandas
conda install -y -c https://conda.rosettacommons.org \
    pyrosetta
# conda install -y -c conda-forge cxx-compiler

pip install \
    mdtraj \
    biopython \
    black \
    darglint \
    deepspeed \
    dill \
    dm-tree \
    e3nn \
    geomstats \
    hydra-zen \
    isort \
    mypy \
    ninja \
    opt-einsum \
    opt-einsum-fx \
    pylint \
    pytest \
    rdkit \
    seaborn \
    tmtools \
    wandb

pip install -e .
