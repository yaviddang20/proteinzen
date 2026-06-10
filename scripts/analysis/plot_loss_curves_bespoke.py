import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--min_epoch", default=0, type=int)
parser.add_argument("--max_epoch", default=-1, type=int)
args = parser.parse_args()

df_paths = [
    '/wynton/home/kortemme/alexjli/projects/proteinzen-clone/outputs/pretrain/phase1_1/lightning_logs/version_0/metrics.csv',
    '/wynton/home/kortemme/alexjli/projects/proteinzen-clone/outputs/pretrain/phase1_2/lightning_logs/version_0/metrics.csv',
]

df_list = [pd.read_csv(path) for path in df_paths]
df_list = reversed(df_list)
start_epoch = 1e6
new_df_list = []
for i, df in enumerate(df_list):
    new_df_list.append(df[df.epoch < start_epoch])
    start_epoch = df.epoch.min()


df = pd.concat(reversed(new_df_list))

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

for idx, l in enumerate(loss_terms):
    if isinstance(l, str):
        if l in df.columns:
            sns.lineplot(df, x='epoch', y=l, ax=axs[get_ax_idx(idx)])
    elif isinstance(l, tuple):
        plot = True
        for term in l:
            if term not in df.columns:
                plot = False
        if plot:
            sns.lineplot(df[list(l)], ax=axs[get_ax_idx(idx)])

# plt.legend()
plt.savefig("loss_curve.png")
