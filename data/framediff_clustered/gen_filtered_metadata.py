import pandas as pd

# df = pd.read_csv("framediff_monomers.csv")
df = pd.read_csv("metadata.csv")
df = df[df["oligomeric_detail"] == "monomeric"]

pdb_to_cluster = {}
with open("clusters-by-entity-30.txt", "r") as f:
    for i, line in enumerate(f):
        for chain in line.split(' '):
            pdb = chain.split('_')[0]
            pdb_to_cluster[pdb.lower()] = i

next_cluster = max(pdb_to_cluster.values()) + 1
print(next_cluster)

monomer_cluster = []
for x in df.pdb_name:
    if x in pdb_to_cluster:
        monomer_cluster.append(pdb_to_cluster[x])
    else:
        monomer_cluster.append(next_cluster)
        next_cluster += 1

df['cluster'] = monomer_cluster

df.to_csv("filtered_metadata.csv")