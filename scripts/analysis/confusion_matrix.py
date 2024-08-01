import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Bio import SeqIO

y_true = []
y_pred = []

designable_samples = [path.split("/")[-1].split(".")[0] for path in glob.glob("designable_samples/*.pdb")]

for fasta_file in tqdm.tqdm(glob.glob("seqs/*")):
    name = fasta_file.split("/")[-1].split(".")[0]
    if name in designable_samples:
        seqs = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
        design_seq = seqs[0]
        pmpnn_seqs = seqs[1:]
        for seq in pmpnn_seqs:
            y_true.append(np.array([c for c in seq]))
            y_pred.append(np.array([c for c in design_seq]))

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

fig = plt.figure(figsize=(15, 15))
cmd = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize="true", ax=fig.gca())
# cmd.plot(ax=fig.gca())
plt.savefig("confusion_matrix.png")