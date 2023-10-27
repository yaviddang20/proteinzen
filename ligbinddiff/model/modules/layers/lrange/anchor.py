import torch
from torch import nn

class FindAnchors(nn.Module):
    def __init__(self, c_s, c_hidden):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(c_s, c_hidden),
            nn.ReLU(),
            nn.Linear(c_hidden, 1))

    def forward(self, node_features, edge_features):
        score = self.score(node_features)
