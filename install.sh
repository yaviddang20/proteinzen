# conda env create -n proteinzen
# conda activate proteinzen
conda install -y -c pytorch -c nvidia -c https://conda.rosettacommons.org -c conda-forge \
    pytorch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    pytorch-cuda=12.4 \
    nvidia/label/cuda-12.4.1::cuda-toolkit 

conda install -y -c pyg -c conda-forge\
    pyg \
    pytorch-cluster \
    pytorch-scatter \
    lightning \
    torchmetrics

conda install -y -c conda-forge \
    scipy \
    pandas

conda install -y -c https://conda.rosettacommons.org \
    pyrosetta

conda install -y -c conda-forge cxx-compiler


# you might need these
conda install nvidia/label/cuda-12.4.1::cuda-cccl nvidia/label/cuda-12.4.1::libcurand

# conda install -y -c conda-forge \
#     mdtraj \
#     numpy
#

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