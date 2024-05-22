import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("filtered_metadata.csv")
df = df[df["modeled_seq_len"] <= 128]
sns.jointplot(df, x="helix_percent", y="strand_percent")
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig("ss_comp_scope128.png")
less_ordered = df[(df["coil_percent"] > 0.5) & (df["coil_percent"] + df["helix_percent"] + df["strand_percent"] <= 1)]
less_ordered["ordered_percent"] = less_ordered["helix_percent"] + less_ordered["strand_percent"]
plt.clf()
sns.histplot(less_ordered, x="ordered_percent")
plt.savefig("less_ordered_scope128.png")

print(less_ordered[(less_ordered["ordered_percent"] > 0.2) & (less_ordered["ordered_percent"] < 0.3)])
print(less_ordered[(less_ordered["ordered_percent"] > 0.3) & (less_ordered["ordered_percent"] < 0.4)])
print(less_ordered[(less_ordered["ordered_percent"] > 0.4) & (less_ordered["ordered_percent"] < 0.5)])