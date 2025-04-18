import torch
from torch import nn

class NodeFeaturizer(nn.Module):
    def __init__(self, c_s):
        super().__init__()