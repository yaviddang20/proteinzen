import pandas as pd
import json

df = pd.read_csv("/wynton/group/kortemme/afdb/clusters/1-AFDBClusters-entryId_repId_taxId.tsv", sep='\t', header=None)
# yea its code golf but it works LOL
mapping_df = df.groupby(1)[0].apply(list)
mapping = mapping_df.to_dict()

df[1] = "AF-".lower() + df[1].str.lower() + "-F1-model_v4_1".lower()
df[0] = "AF-".lower() + df[0].str.lower() + "-F1-model_v4_1".lower()
mapping_df = df.groupby(1)[0].apply(list)
mapping = mapping_df.to_dict()

clustering = {}
for key, values in mapping.items():
    for item in values:
        clustering[item] = key

with open("clustering.json", 'w') as fp:
    json.dump(clustering, fp)