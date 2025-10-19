# conda env create -n proteinzen
# conda activate proteinzen
pip install torch==2.6 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126 \

pip install cuequivariance cuequivariance-torch \
    cuequivariance-ops-torch-cu12 \

pip install \
    torch_geometric \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    lightning torchmetrics pandas \
    -f https://data.pyg.org/whl/torch-2.6.0+cu126.html

conda install -y -c https://conda.rosettacommons.org \
    pyrosetta

pip install \
    mdtraj \
    biopython \
    black \
    darglint \
    dill \
    dm-tree \
    e3nn \
    geomstats \
    hydra-zen \
    isort \
    mypy \
    ninja \
    pylint \
    pytest \
    rdkit \
    seaborn \
    wandb \
    mashumaro \
    p_tqdm

pip install -e .
