import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("filtered_metadata.csv")
print(df[df["strand_percent"] < 0.05])
sns.jointplot(df, x="helix_percent", y="strand_percent")
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig("ss_comp_unclustered.png")

cluster_df = df.groupby("cluster").sample(1)
print(cluster_df[cluster_df["strand_percent"] < 0.05])
sns.jointplot(cluster_df, x="helix_percent", y="strand_percent")
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig("ss_comp_clustered.png")

less_ordered = df[(df["coil_percent"] > 0.5) & (df["coil_percent"] + df["helix_percent"] + df["strand_percent"] <= 1)]
less_ordered["ordered_percent"] = less_ordered["helix_percent"] + less_ordered["strand_percent"]
plt.clf()
sns.histplot(less_ordered, x="ordered_percent")
plt.savefig("less_ordered_framediff.png")

print(less_ordered[(less_ordered["ordered_percent"] > 0.2) & (less_ordered["ordered_percent"] < 0.3)])
print(less_ordered[(less_ordered["ordered_percent"] > 0.3) & (less_ordered["ordered_percent"] < 0.4)])
print(less_ordered[(less_ordered["ordered_percent"] > 0.4) & (less_ordered["ordered_percent"] < 0.5)])

less_ordered_clustered = less_ordered.groupby("cluster").sample(1)
plt.clf()
sns.histplot(less_ordered_clustered, x="ordered_percent")
plt.savefig("less_ordered_framediff_clustered.png")
print(less_ordered_clustered[(less_ordered_clustered["ordered_percent"] > 0.2) & (less_ordered_clustered["ordered_percent"] < 0.3)])
print(less_ordered_clustered[(less_ordered_clustered["ordered_percent"] > 0.3) & (less_ordered_clustered["ordered_percent"] < 0.4)])
print(less_ordered_clustered[(less_ordered_clustered["ordered_percent"] > 0.4) & (less_ordered_clustered["ordered_percent"] < 0.5)])