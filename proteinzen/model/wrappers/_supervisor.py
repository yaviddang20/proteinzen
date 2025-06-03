import torch
from torch import nn

class LatentNodeSupervisor(nn.Module):
    def __init__(self, c_latent, c_hidden, n_aa=20, n_torsion_bins=36, n_torsions=4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(c_latent, c_hidden),
            nn.ReLU(),
            nn.Linear(c_hidden, c_hidden),
            nn.ReLU(),
            nn.LayerNorm(c_hidden)
        )
        self.seq_head = nn.Linear(c_hidden, n_aa)
        self.torsion_head = nn.Linear(c_hidden, n_torsion_bins * n_torsions)
        self.n_torsion_bins = n_torsion_bins
        self.n_torsions = n_torsions

    def forward(self, latent_features):
        x = self.trunk(latent_features)
        seq_logits = self.seq_head(x)
        torsion_logits = self.torsion_head(x)
        torsion_logits = torsion_logits.view(*torsion_logits.shape[:-1], self.n_torsions, self.n_torsion_bins)
        return seq_logits, torsion_logits