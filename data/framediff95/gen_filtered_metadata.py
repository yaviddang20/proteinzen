import pandas as pd
import pickle
import tqdm
import multiprocessing as mp

# df = pd.read_csv("framediff_monomers.csv")
df = pd.read_csv("metadata.csv")
df = df[df["oligomeric_detail"] == "monomeric"]

# ca_only = []
# for path in tqdm.tqdm(df['processed_path'].tolist()):
#     with open(path, 'rb') as fp:
#         data = pickle.load(fp)
#     ca_only_struct = ~(data['atom_mask'][:, 0].any() & data['atom_mask'][:, 2:].any())
#     if ca_only_struct:
#         ca_only.append(True)
#     else:
#         ca_only.append(False)
def is_ca_only(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
    ca_only_struct = ~(data['atom_mask'][:, 0].any() & data['atom_mask'][:, 2:].any())
    return ca_only_struct

pbar = tqdm.tqdm(total=len(df))

def update(x):
    pbar.update(1)

with mp.Pool(32) as pool:
    ca_only = [pool.apply_async(is_ca_only, (path,), callback=update) for path in df['processed_path']]
    pool.close()
    pool.join()
not_ca_only = [~(r.get()) for r in ca_only]
print(len(df) - sum(not_ca_only))
df = df[not_ca_only]

pdb_to_cluster = {}
with open("clusters-by-entity-95.txt", "r") as f:
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