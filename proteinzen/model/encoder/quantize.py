import torch
from torch import nn

from torch_cluster import nearest

class Quantizer(nn.Module):
    def __init__(self, h_dim, n_codebook):
        super().__init__()
        self.h_dim = h_dim
        self.codebook = torch.Parameter(
            torch.randn(h_codebook, h_dim)
        )

    def forward(
        self,
        features,
        features_mask,
        use_torch_cluster=True,
    ):
        features_shape = features.shape
        features = features.view(-1, self.h_dim)
        if use_torch_cluster:
            codebook_idx = nearest(features, self.codebook)
        else:
            dist_mat = torch.cdist(features[None], self.codebook[None]).squeeze(0)
            codebook_idx = torch.argmax(dist_mat, dim=-1)

        quantized_features = self.codebook[codebook_idx]
        quantized_features = quantized_features.view(features_shape)

        all_other_dims = [i for _ in range(features.dim()) if i != 0]
        quant_loss = torch.square(features.detach() - quantized_features) * features_mask[..., None]
        quant_loss = quant_loss.sum(dim=all_other_dims) / features_mask.sum(dim=all_other_dims)

        commit_loss = torch.square(features - quantized_features.detach()) * features_mask[..., None]
        commit_loss = commit_loss.sum(dim=all_other_dims) / features_mask.sum(dim=all_other_dims)

        # "copy" gradients
        quantized_features = features + (quantized_features - features).detach()

        return quantized_features, quant_loss, commit_loss