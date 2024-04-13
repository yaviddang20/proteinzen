import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("filtered_metadata.csv")

sns.histplot(data=df, x="modeled_seq_len")
plt.savefig("len_dist.png")
plt.clf()

sns.ecdfplot(data=df, x="modeled_seq_len", stat="count")
for i in range(100, 300+1, 50):
    plt.axvline(i, 0., 1.)
plt.savefig("ecdf_len_dist.png")
