# conda env create -n proteinzen
# conda activate proteinzen
pip install torch==2.6 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126 \

pip install cuequivariance cuequivariance-torch \
    cuequivariance-ops-torch-cu12 \

pip install \
    torch_geometric \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    lightning==2.5.6 torchmetrics pandas \
    -f https://data.pyg.org/whl/torch-2.6.0+cu126.html

mamba install -y -c https://conda.rosettacommons.org \
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

pip install flash-attn --no-build-isolation

pip install -e .

# install steering reward models
pip install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1 alphafold-colabfold==2.3.13
pip install jaxlib==0.4.29+cuda12.cudnn91 "jax[cuda12]==0.4.29" flax==0.9.0 \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# pip install flax==0.9.0 --no-deps
pip install "git+https://github.com/uw-ipd/tmol.git@d8a6f7f9649d36e74440bca25246ee7c467ce490"
