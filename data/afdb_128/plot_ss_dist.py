import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams.update({
    'font.size': 18,
})

df = pd.read_csv("filtered_metadata.csv")
df = df.groupby("cluster").sample(1)
df = df[(df["coil_percent"] < 0.5) & (df["helix_percent"] + df["strand_percent"] >= 0.5)]
jointplot = sns.jointplot(df, x="helix_percent", y="strand_percent", kind='hist')
jointplot.ax_joint.set(xlabel="Fraction Helix", ylabel="Fraction Strand")
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig("ss_comp.png")

less_ordered = df[(df["coil_percent"] > 0.5) & (df["coil_percent"] + df["helix_percent"] + df["strand_percent"] <= 1)]
less_ordered["ordered_percent"] = less_ordered["helix_percent"] + less_ordered["strand_percent"]
plt.clf()
sns.histplot(less_ordered, x="ordered_percent")
plt.savefig("less_ordered_scope128.png")

print(less_ordered[(less_ordered["ordered_percent"] > 0.2) & (less_ordered["ordered_percent"] < 0.3)])
print(less_ordered[(less_ordered["ordered_percent"] > 0.3) & (less_ordered["ordered_percent"] < 0.4)])
print(less_ordered[(less_ordered["ordered_percent"] > 0.4) & (less_ordered["ordered_percent"] < 0.5)])