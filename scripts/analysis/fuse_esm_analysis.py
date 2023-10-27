import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

cwd = os.getcwd()
os.chdir("/wynton/home/kortemme/alexjli/projects/ligbinddiff/prelim_experiments/")
tail = "samples_step_230k/esmfold/best_sc_rmsd.csv"

df = pd.read_csv(os.path.join("framediff", tail))
df['condition'] = ["framediff"] * len(df)
df['length'] = [int(name.split("_")[1]) for name in df['name'].tolist()]
print((df['sc_rmsd'] < 2).mean())

df_mod = pd.read_csv(os.path.join("framediff_mod", tail))
df_mod['condition'] = ["+sparse+batching"] * len(df_mod)
df_mod['length'] = [int(name.split("_")[1]) for name in df_mod['name'].tolist()]
print((df_mod['sc_rmsd'] < 2).mean())

df_mod_time_batch = pd.read_csv(os.path.join("framediff_mod_time_batch", tail))
df_mod_time_batch['condition'] = ["+sparse"] * len(df_mod)
df_mod_time_batch['length'] = [int(name.split("_")[1]) for name in df_mod_time_batch['name'].tolist()]
print((df_mod_time_batch['sc_rmsd'] < 2).mean())

df_mod_hbond = pd.read_csv(os.path.join("framediff_mod_bb_hbond", tail))
df_mod_hbond['condition'] = ["+sparse+batching-local+hbond"] * len(df_mod)
df_mod_hbond['length'] = [int(name.split("_")[1]) for name in df_mod_hbond['name'].tolist()]
print((df_mod_hbond['sc_rmsd'] < 2).mean())

df_mod_both = pd.read_csv(os.path.join("framediff_mod_both", tail))
df_mod_both['condition'] = ["+sparse+batching+hbond"] * len(df_mod)
df_mod_both['length'] = [int(name.split("_")[1]) for name in df_mod_both['name'].tolist()]
print((df_mod_both['sc_rmsd'] < 2).mean())

full_df = pd.concat([df, df_mod_time_batch, df_mod, df_mod_hbond, df_mod_both])
ax = sns.violinplot(full_df, x="length", y="sc_rmsd", hue="condition")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
os.chdir(cwd)
plt.plot([-0.5, 4.5], [2, 2], color='red')
plt.savefig("out.png", bbox_inches='tight')
