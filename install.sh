# conda env create -n proteinzen
# conda activate proteinzen
conda install -y -c pytorch -c nvidia \
    pytorch=2.4 \
    pytorch-cuda=12.4

conda install -y -c https://conda.rosettacommons.org \
    pyrosetta

conda install -y -c pyg \
    pyg \
    pytorch-cluster \
    pytorch-scatter

conda install -y -c conda-forge \
    lightning \
    torchmetrics

conda install -y -c conda-forge \
    scipy \
    pandas

# conda install -y -c conda-forge \
#     mdtraj \
#     numpy
#
conda install -y nvidia/label/cuda-12.4.0::cuda-toolkit

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


# conda env create -n proteinzen
# conda activate proteinzen
# conda install -c pytorch -c nvidia \
#     pytorch \
#     pytorch-cuda=12.1
#
# conda install -c conda-forge \
#     lightning
#
# conda install -c https://conda.rosettacommons.org \
#     pyrosetta
#
# conda install -c pytorch -c nvidia -c pyg \
#     pyg \
#     pytorch-cluster \
#     pytorch-scatter
#
# conda install \
#     scipy \
#     numpy \
#     matplotlib \
#     mdtraj \
#     pandas \
#     torchmetrics
#
# conda install nvidia/label/cuda-12.4.0::cuda-toolkit
#
# pip install \
#     biopython \
#     black \
#     darglint \
#     deepspeed \
#     dill \
#     dm-tree \
#     e3nn \
#     hydra-zen \
#     isort \
#     mypy \
#     ninja \
#     opt-einsum \
#     opt-einsum-fx \
#     pylint \
#     pytest \
#     rdkit \
#     seaborn \
#     tmtools \
#     wandb
#
#