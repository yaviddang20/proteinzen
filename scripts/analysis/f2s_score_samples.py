import glob
import os
from tqdm import tqdm
import torch

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
    str_form = [residue_constants.ID_TO_AA[int(i)] for i in aatype[0]]
    input_aatype_onehot = residue_constants.sequence_to_onehot(
        sequence=str_form,
        mapping=residue_constants.AA_TO_ID,
    )
    input_aatype_onehot = torch.from_numpy(input_aatype_onehot).float()
    input_aatype_onehot = input_aatype_onehot.unsqueeze(0)
    input_aatype_onehot = input_aatype_onehot.to(self.device)
    input_aatype_onehot = torch.zeros_like(input_aatype_onehot)
    input_aatype_onehot[:, :,
                        20] = 1  # all positions are masked (set to unknown)
    with torch.no_grad():
        pred_seq1 = self.models[0].forward(X, seq_mask, input_aatype_onehot)
        pred_seq2 = self.models[1].forward(X, seq_mask, input_aatype_onehot)
        pred_seq3 = self.models[2].forward(X, seq_mask, input_aatype_onehot)
        pred_seq = (pred_seq1 + pred_seq2 + pred_seq3) / 3  # ensemble
        pred_seq = pred_seq / temperature
        pred_seq = torch.nn.functional.softmax(pred_seq, dim=-1)
        pred_seq = pred_seq[seq_mask]
        input_seqs = read_fasta_file(fasta_file)
        input_seqs_strs = input_seqs
        input_seqs = [
            torch.tensor([residue_constants.AA_TO_ID[aa]
                            for aa in seq]).long() for seq in input_seqs
        ]
        input_seqs = torch.stack(input_seqs, dim=0)
        neg_pll, _ = get_neg_pll(pred_seq[None].expand(input_seqs.shape[0], *pred_seq.shape), input_seqs)

        torch.save(
            {
                "scores": neg_pll,
                "seqs": input_seqs_strs,
                "key": residue_constants.AA_TO_ID
            },
        out_path)


runner = Frame2seqRunner()
out_dir = "f2s_scores"
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

chain_id = "A"
for pdb_file in tqdm(glob.glob("not_consistent_samples/*.pdb")):
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