import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--min_epoch", default=0, type=int)
parser.add_argument("--max_epoch", default=-1, type=int)
args = parser.parse_args()


df = pd.read_csv("metrics.csv")
df = df.iloc[1:]
df = df[df['train/loss_epoch'].astype(str).str.len() > 0]
df = df[~df['train/loss_epoch'].isna()]
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
    'train/scaled_fafe_epoch',
    'grad_norm_epoch',
    # ('task/motif_scaffold_unindexed_frame_vf_loss_epoch', 'task/motif_scaffold_indexed_frame_vf_loss_epoch', 'task/unconditional_frame_vf_loss_epoch')
    ('task/motif_scaffold_unindexed_frame_vf_loss_epoch', 'task/unconditional_frame_vf_loss_epoch')

]

def moving_average(x, w):
    ret = np.convolve(x, np.ones(w), 'valid') / w
    ret = np.concatenate([
        ret[0] * np.ones(w-1),
        ret,
    ])
    return ret

for idx, l in enumerate(loss_terms):
    if isinstance(l, str):
        if l in df.columns:
            df[l] = moving_average(df[l].to_numpy(), 10)
            sns.lineplot(df, x='epoch', y=l, ax=axs[get_ax_idx(idx)])
            # axs[get_ax_idx(idx)].set_ylim(bottom=1.2, top=1.45)
    elif isinstance(l, tuple):
        plot = True
        for term in l:
            if term not in df.columns:
                plot = False
        if plot:
            sns.lineplot(df[list(l)], ax=axs[get_ax_idx(idx)])

# plt.legend()
plt.savefig("loss_curve.png")
