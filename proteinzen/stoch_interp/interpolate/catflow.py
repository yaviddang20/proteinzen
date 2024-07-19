import torch
import torch.nn.functional as F

# implemented from https://arxiv.org/abs/2406.04843 
class CatFlow:
    def __init__(self,
                 D=20):
        self.D = D

    def sample_prior(self, num_nodes):
        return torch.randn((num_nodes, self.D))

    def _corrupt_seq(self, seq, t):
        seq_1 = F.one_hot(seq, num_classes=self.D)
        seq_0 = self.sample_prior(seq_1.shape[0]).to(seq_1.device)
        seq_t = t[..., None] * seq_0 + (1 - t[..., None]) * seq_1

        return seq_t

    def corrupt_batch(self, batch):
        res_data = batch["residue"]
        t = batch['t']
        nodewise_t = t[res_data.batch]

        # [N]
        res_mask = res_data["res_mask"]
        seq_mask = res_data['seq_mask']
        noising_mask = res_data["noising_mask"]
        mask = res_mask & seq_mask

        seq = res_data['seq']
        seq[~seq_mask] = 0
        seq_t = self._corrupt_seq(seq, nodewise_t)
        seq_t[~mask] = 0

        # TODO: this is a hack
        res_data['seq_probs_t'] = seq_t 
        return batch

    def euler_step(self, d_t, t, seq_t, seq_1):
        vf_per_aa = (torch.eye(self.D, device=seq_t.device)[None] - seq_t[..., None, :]) / (1 - t[..., None, None])
        seq_probs_1 = F.softmax(seq_1, dim=-1)
        vf = (seq_probs_1[..., None, :] @ vf_per_aa).squeeze(-2)
        seq_t_1 = seq_t + d_t * vf
        return seq_t_1