import json
import glob

samples_metadata = {}
for metadata_path in glob.glob("samples_metadata_*.json"):
    with open(metadata_path) as fp:
        samples_metadata.update(json.load(fp))
with open("samples_metadata.json", 'w') as fp:
    json.dump(samples_metadata, fp)

fixed_pos = {}
for fixed_pos_path in glob.glob("pmpnn_fixed_pos_dict_*.jsonl"):
    with open(fixed_pos_path) as fp:
        fixed_pos.update(json.load(fp))
with open("pmpnn_fixed_pos_dict.jsonl", 'w') as fp:
    json.dump(fixed_pos, fp)