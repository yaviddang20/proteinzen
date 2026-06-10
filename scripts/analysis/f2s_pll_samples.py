import glob
import numpy as np
import os
from tqdm import tqdm
import time
import torch
from torch.nn import functional as F

from frame2seq import Frame2seqRunner
from frame2seq.utils import residue_constants
from frame2seq.utils.util import get_neg_pll, read_fasta_file
from frame2seq.utils.pdb2input import get_inference_inputs
from frame2seq.utils.pred2output import output_csv, output_indiv_csv


def score(self, pdb_file, chain_id, fasta_file, out_path):
    temperature = 1.0
    seq_mask, aatype, X = get_inference_inputs(pdb_file, chain_id)
    seq_mask = seq_mask.to(self.device)
    aatype = aatype.to(self.device)
    X = X.to(self.device)
    input_seqs = read_fasta_file(fasta_file)
    input_seqs_strs = input_seqs
    input_seqs = [
        torch.tensor([residue_constants.AA_TO_ID[aa]
                        for aa in seq]).long() for seq in input_seqs
    ]
    input_seqs = torch.stack(input_seqs, dim=0).to(self.device)

    neg_pll = []

    seq_len = X.shape[1]
    idxs = np.array_split(list(range(seq_len)), np.ceil(seq_len / (4 * (128/seq_len)**2)))
    for i_list in idxs:
        # start = time.time()
        one_hots = []
        for i in i_list:
            onehot = F.one_hot(input_seqs, num_classes=21).float()
            onehot[:, i] = 0
            onehot[:, i, 20] = 1
            one_hots.append(onehot)

        onehot = torch.cat(one_hots, dim=0)
        _X = X.expand(onehot.shape[0], *X.shape[1:])
        # print("prep", time.time()-start)

        with torch.no_grad():
            # pred_seq1 = self.models[0].forward(_X, seq_mask, onehot)
            # pred_seq2 = self.models[1].forward(_X, seq_mask, onehot)
            pred_seq3 = self.models[2].forward(_X, seq_mask, onehot)
            # print("forward", time.time()-start)
            # pred_seq = (pred_seq1 + pred_seq2 + pred_seq3) / 3  # ensemble
            pred_seq = pred_seq3
            pred_seq = pred_seq / temperature
            pred_seq = torch.nn.functional.softmax(pred_seq, dim=-1)
            # print("logits", time.time()-start)
            # pred_seq = pred_seq[seq_mask[..., None].expand(input_seqs.shape[0], *seq_mask.shape[1:], -1)]
            batch_neg_pll, _ = get_neg_pll(pred_seq, input_seqs.tile(len(i_list), *[1 for _ in range(input_seqs.dim()-1)]))
            # print("pll", time.time()-start)
            neg_plls = batch_neg_pll.split([t.shape[0] for t in one_hots])
            for i, neg_pll_i in zip(i_list, neg_plls):
                neg_pll.append(neg_pll_i[:, [i]])
        # print("end", time.time()-start)

    neg_pll = torch.cat(neg_pll, dim=1)
    torch.save(
        {
            "scores": neg_pll,
            "seqs": input_seqs_strs
        },
    out_path)


runner = Frame2seqRunner()
out_dir = "f2s_pll"
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

chain_id = "A"
for pdb_file in tqdm(sorted(glob.glob("samples/*.pdb"), key=lambda x: int(x.split("_")[-3]))):
    fasta_file = os.path.join(
        os.getcwd(),
        "seqs",
        os.path.basename(pdb_file).split(".")[0] + ".fa"
    )
    out_path = os.path.join(
        out_dir,
        os.path.basename(pdb_file).split(".")[0] + ".pt"
    )
    score(runner, pdb_file, chain_id, fasta_file, out_path)