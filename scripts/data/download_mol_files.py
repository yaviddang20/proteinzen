from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
import multiprocessing as mp
import os
import json
import glob
import requests
import tqdm
import time

from ligbinddiff.data.io.mmcif import mmcif_loop_to_list

DB = "/wynton/group/kortemme/alexjli/databases/PDB/mmCIF"
OUT_PATH = "/wynton/group/kortemme/alexjli/databases/ligandmpnn"

def download_lig_files_from_pdb(pdb_code,
                                nonpoly_dict,
                                out_path,
                                exclude=["HOH", "NA", "CL", "K", "BR"]):
    errors = []
    for entry in nonpoly_dict:
        if entry['_pdbx_nonpoly_scheme.mon_id'].upper() in exclude:
            continue
        name = entry['_pdbx_nonpoly_scheme.mon_id']
        asym_id = entry['_pdbx_nonpoly_scheme.asym_id']
        seq_id = entry['_pdbx_nonpoly_scheme.pdb_seq_num']
        sdf_url = f"https://models.rcsb.org/v1/{pdb_code}/ligand?label_asym_id={asym_id}&auth_seq_id={seq_id}&encoding=sdf"
        mol2_url = f"https://models.rcsb.org/v1/{pdb_code}/ligand?label_asym_id={asym_id}&auth_seq_id={seq_id}&encoding=mol2"
        repeat = True
        while repeat:
            response = requests.get(sdf_url)
            if str(response.status_code).startswith("2"):
                with open(os.path.join(out_path, f"{pdb_code}_{name}_{asym_id}_{seq_id}.sdf"), 'w') as fp:
                    fp.write(response.text)
                repeat = False
            elif response.status_code == 429:
                time.sleep(5)
            elif response.status_code in [503, 504]:
                time.sleep(5)
            else:
                repeat = False
                errors.append(f"{pdb_code}_{name}_{asym_id}_{seq_id}.sdf {response.status_code}")
        repeat = True
        while repeat:
            response = requests.get(mol2_url)
            if str(response.status_code).startswith("2"):
                with open(os.path.join(out_path, f"{pdb_code}_{name}_{asym_id}_{seq_id}.mol2"), 'w') as fp:
                    fp.write(response.text)
                repeat = False
            elif response.status_code == 429:
                time.sleep(5)
            elif response.status_code in [503, 504]:
                time.sleep(5)
            else:
                repeat = False
                errors.append(f"{pdb_code}_{name}_{asym_id}_{seq_id}.mol2 {response.status_code}")
    return errors

def download_ligands_and_split_polymers(pdb_code):
    mid = pdb_code.lower()[1:3]
    os.makedirs(os.path.join(OUT_PATH, mid), exist_ok=True)
    os.makedirs(os.path.join(OUT_PATH, mid, pdb_code), exist_ok=True)
    out_path = os.path.join(OUT_PATH, mid, pdb_code)

    if os.path.exists(os.path.join(out_path, pdb_code + "_protein.pdb")):
        return

    parser = MMCIFParser(QUIET=True)
    fp = os.path.join(DB, mid, pdb_code + ".cif")
    if not os.path.exists(fp):
        return [f"{pdb_code}: DNE"]
    full_structure = parser.get_structure('', fp)
    first_model = next(full_structure.get_models())
    # Extract the _mmcif_dict from the parser, which contains useful fields not
    # reflected in the Biopython structure.
    parsed_info = parser._mmcif_dict  # pylint:disable=protected-access
    nonpoly_dict = mmcif_loop_to_list("_pdbx_nonpoly_scheme.", parsed_info)
    errors = download_lig_files_from_pdb(pdb_code, nonpoly_dict, out_path)
    if len(errors) > 0:
        print(errors)
        return errors

    entity_poly_dict = mmcif_loop_to_list("_entity_poly.", parsed_info)
    protein_chains = []
    rna_chains = []
    dna_chains = []
    for entry in entity_poly_dict:
        chains = entry['_entity_poly.pdbx_strand_id'].split(",")
        poly_type = entry['_entity_poly.type']
        if "polypeptide" in poly_type:
            protein_chains += chains
        elif "polyribonucleotide" in poly_type:
            rna_chains += chains
        elif "polydeoxyribonucleotide" in poly_type:
            dna_chains += chains

    protein_model = Model(hash(pdb_code + str(0)))
    if len(rna_chains) > 0:
        rna_model = Model(hash(pdb_code + str(1)))
    else:
        rna_model = None
    if len(dna_chains) > 0:
        dna_model = Model(hash(pdb_code + str(2)))
    else:
        dna_model = None

    for chain in first_model.get_chains():
        chain_id = chain.get_id()
        if chain_id in protein_chains:
            protein_model.add(chain)
        elif chain_id in rna_chains:
            rna_model.add(chain)
        elif chain_id in dna_chains:
            dna_model.add(chain)

    pdb_io = PDBIO()

    try:
        protein_struct = Structure(hash("protein" + pdb_code))
        protein_struct.add(protein_model)
        pdb_io.set_structure(protein_struct)
        pdb_io.save(os.path.join(out_path, pdb_code + "_protein.pdb"))

        if rna_model is not None:
            rna_struct = Structure(hash("rna" + pdb_code))
            rna_struct.add(rna_model)
            pdb_io.set_structure(rna_struct)
            pdb_io.save(os.path.join(out_path, pdb_code + "_rna.pdb"))

        if dna_model is not None:
            dna_struct = Structure(hash("dna" + pdb_code))
            dna_struct.add(dna_model)
            pdb_io.set_structure(dna_struct)
            pdb_io.save(os.path.join(out_path, pdb_code + "_dna.pdb"))
    except Exception as e:
        return [f"{pdb_code}: {str(e)}"]

codes = []
for path in glob.glob("/wynton/home/kortemme/alexjli/projects/ligbinddiff/data/ligandmpnn/*.json"):
    with open(path) as fp:
        split_set = list(json.load(fp))
        print(path, len(split_set))
        codes += split_set


errors = []

pbar = tqdm.tqdm(total=len(codes))
def callback(a):
    pbar.update(1)
def error_callback(a):
    print(a)
    pbar.update(1)

try:
    with mp.Pool(32) as pool:
        ret = [pool.apply_async(download_ligands_and_split_polymers, (code,), callback=callback, error_callback=error_callback) for code in codes]
        pool.close()
        pool.join()
finally:
    for r in ret:
        try:
            r = r.get()
            if r is not None and isinstance(r, list):
                errors += r
        except:
            pass
    with open(os.path.join(OUT_PATH, "errors2.list"), "a") as fp:
        for l in errors:
            fp.write(l + "\n")

# for c in tqdm.tqdm(codes):
#     e = download_ligands_and_split_polymers(c)
#     if e is not None:
#         errors += e
