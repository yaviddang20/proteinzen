import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

df = pd.read_csv("esmfold/folding_rmsd.csv")
df["seq_len"] = pd.to_numeric(df["name"].str.split("_", expand=True)[1], errors="coerce").astype(np.int64)

designable = pd.to_numeric(df["sc_rmsd"] < 2).astype(np.float32)
percent_designable = designable.rolling(window=50).mean()
df["percent_designable"] = percent_designable

ax = plt.gca()

sns.scatterplot(df, x="seq_len", y="sc_rmsd", hue="plddt", ax=ax, palette="viridis", legend=False)
ax2 = ax.twinx()
sns.lineplot(df, x="seq_len", y="percent_designable", color="orange", ax=ax2, errorbar=None)
ax2.set_ylim(0, 1)
ax2.yaxis.label.set_color('orange')

# add a colorbar instead of the sampled legend
cmap = sns.color_palette('viridis', as_cmap=True)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize())
cbar = plt.colorbar(sm, ax=ax2)
cbar.set_label('plddt')
pos = cbar.ax.get_position()
pos.x0 += 0.10
pos.x1 += 0.10
cbar.ax.set_position(pos)

ax.axhline(y=2.0, color='red', linestyle='--')
plt.savefig("folding_over_len.png")
