import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--min_epoch", default=0, type=int)
parser.add_argument("--max_epoch", default=-1, type=int)
args = parser.parse_args()


df = pd.read_csv("metrics.csv")
df = df[df['train/loss_epoch'].astype(str).str.len() > 0]
df = df[df.epoch.astype(int) >= args.min_epoch]
if args.max_epoch != -1:
    df = df[df.epoch.astype(int) <= args.max_epoch]


fig, axs = plt.subplots(5, 3)
fig.set_size_inches(12, 20)

get_ax_idx = lambda x: (x // 3, x % 3)

loss_terms = [
    'train/loss_epoch',
    'train/frame_vf_loss_epoch',
    'train/rot_vf_loss_epoch',
    'train/trans_vf_loss_epoch',
    'train/seq_loss_epoch',
    'train/smooth_lddt_epoch',
    'task/motif_scaffold_unindexed_frame_vf_loss_epoch',
    'task/motif_scaffold_indexed_frame_vf_loss_epoch',
    'task/unconditional_frame_vf_loss_epoch',
]

for idx, l in enumerate(loss_terms):
    if l in df.columns:
        sns.lineplot(df, x='epoch', y=l, ax=axs[get_ax_idx(idx)])

# plt.legend()
plt.savefig("loss_curve.png")
