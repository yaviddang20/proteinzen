""" Compute the bbRMSD between structures """
import argparse
import os
import glob
import json
import dataclasses
import shutil

import torch
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import tqdm
import p_tqdm
from Bio.PDB.Superimposer import Superimposer
from Bio.PDB.PDBIO import PDBIO
import pyrosetta


from pyrosetta.rosetta.core.simple_metrics.metrics import RMSDMetric
from pyrosetta.rosetta.core.simple_metrics.per_residue_metrics import PerResidueRMSDMetric
from pyrosetta.rosetta.core.scoring import superimpose_pose, rms_at_corresponding_atoms_no_super

from proteinzen.openfold.utils.feats import atom14_to_atom37
from proteinzen.openfold.utils.tensor_utils import tree_map, tensor_tree_map

RMSD_PROTEIN_CA_TYPE = pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_protein_bb_ca
RMSD_ALL_HEAVY_TYPE = pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_all_heavy
RMSD_PROTEIN_BB_HEAVY_TYPE = pyrosetta.rosetta.core.scoring.rmsd_atoms.rmsd_protein_bb_heavy


def atom14_to_atom37_np(atom14, batch):
    batch = tree_map(lambda n: torch.tensor(n), batch, np.ndarray)
    out = atom14_to_atom37(torch.as_tensor(atom14), batch)
    return (x.numpy() for x in out)


def parse_esm_log(lines):
    ret = {}
    samples = {}
    name = None
    for l in lines:
        l = l.strip()
        if "Reading sequences" in l:
            if name is not None:
                ret[name] = samples
            name = os.path.splitext(
                l.split("/")[-1]
            )[0]
            samples = {}
        elif "model_name=v_48_020" in l:
            sample_dict = {}
            sample_num = 0

            other_vals = l.split(",")
            for substring in other_vals:
                substring = substring.strip()
                if "pLDDT" in substring:
                    sample_dict["plddt"] = float(substring.split(" ")[1])
                elif "pTM" in substring:
                    sample_dict["ptm"] = float(substring.split(" ")[1])
                elif "pAE" in substring:
                    sample_dict["pae"] = float(substring.split(" ")[1])
            samples[sample_num] = sample_dict

        elif "sample=" in l:
            tail = l.split("sample=")[-1]
            sample_dict = {}
            sample_num = int(tail.split(",")[0])

            other_vals = tail.split(",")
            for substring in other_vals:
                substring = substring.strip()
                if "pLDDT" in substring:
                    sample_dict["plddt"] = float(substring.split(" ")[1])
                elif "pAE" in substring:
                    sample_dict["pae"] = float(substring.split(" ")[1])
            samples[sample_num] = sample_dict
    ret[name] = samples

    return ret

def parse_dict_to_df(d):
    keys = ["name", "sample", "plddt", "ptm"]
    pd_dict = {k: [] for k in keys}
    for name, samples in d.items():
        for sample_num, sample_dict in samples.items():
            pd_dict["name"].append(name)
            pd_dict["sample"].append(sample_num)
            pd_dict["plddt"].append(sample_dict['plddt'])
            pd_dict["ptm"].append(sample_dict['ptm'])

    return pd.DataFrame.from_dict(pd_dict)


# RMSDMetric spazzes out sometimes and I dont know why so here's a numpy version for C-alpha rmsd
def calc_ca_rmsd(
    mod_pose,
    ref_pose,
    residue_selector_ref,
    residue_selector,
    residue_selector_super_ref,
    residue_selector_super,
):
    ref_residues = residue_selector_super_ref.get_residues(ref_pose)
    mod_residues = residue_selector_super.get_residues(mod_pose)
    mod_ca_coords = np.array([mod_pose.residue(res).atom("CA").xyz() for res in mod_residues])
    ref_ca_coords = np.array([ref_pose.residue(res).atom("CA").xyz() for res in ref_residues])

    mod_com = mod_ca_coords.mean(axis=0)
    mod_ca_coords = mod_ca_coords - mod_com[None]
    ref_com = ref_ca_coords.mean(axis=0)
    ref_ca_coords = ref_ca_coords - ref_com[None]

    rot, _ = Rotation.align_vectors(ref_ca_coords, mod_ca_coords)

    ref_residues = residue_selector_ref.get_residues(ref_pose)
    mod_residues = residue_selector.get_residues(mod_pose)
    mod_ca_coords = np.array([mod_pose.residue(res).atom("CA").xyz() for res in mod_residues])
    ref_ca_coords = np.array([ref_pose.residue(res).atom("CA").xyz() for res in ref_residues])

    mod_com = mod_ca_coords.mean(axis=0)
    mod_ca_coords = mod_ca_coords - mod_com[None]
    ref_com = ref_ca_coords.mean(axis=0)
    ref_ca_coords = ref_ca_coords - ref_com[None]

    mod_ca_coords = rot.apply(mod_ca_coords)
    mse = np.mean(
        np.sum((mod_ca_coords - ref_ca_coords) ** 2, axis=-1)
    )
    rms = np.sqrt(mse)
    return rms


def compute_rmsds(
    design_path,
    folded_path,
    pmpnn_paths,
    fixed_seq,
    fixed_seq_chain,
    fixed_bb,
    fixed_bb_chain
):

    design_pose = pyrosetta.pose_from_pdb(design_path)

    all_res_selector = pyrosetta.rosetta.core.select.residue_selector.TrueResidueSelector()
    pdb_resids = [f"{resid}{chain}" for resid, chain in zip(fixed_seq, fixed_seq_chain)]
    res_str = ",".join(pdb_resids)
    bb_resids = [f"{resid}{chain}" for resid, chain in zip(fixed_bb, fixed_bb_chain)]
    bb_str = ",".join(bb_resids)

    # esmfold automatically sets the chain to A
    # so we need a different residue selector
    esmfold_pdb_resids = [f"{resid}A" for resid, chain in zip(fixed_seq, fixed_seq_chain)]
    esmfold_res_str = ",".join(esmfold_pdb_resids)
    esmfold_bb_resids = [f"{resid}A" for resid, chain in zip(fixed_bb, fixed_bb_chain)]
    esmfold_bb_str = ",".join(esmfold_bb_resids)

    fixed_res_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    if len(res_str) == 0 and len(bb_str) > 0:
        fixed_res_selector.set_index(bb_str)
    else:
        fixed_res_selector.set_index(res_str)
    esmfold_fixed_res_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    if len(esmfold_res_str) == 0 and len(esmfold_bb_str) > 0:
        esmfold_fixed_res_selector.set_index(esmfold_bb_str)
    else:
        esmfold_fixed_res_selector.set_index(esmfold_res_str)

    fixed_bb_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    if len(bb_str) > 0:
        fixed_bb_selector.set_index(bb_str)
    else:
        fixed_bb_selector = None
    print(res_str, bb_str)
    esmfold_fixed_bb_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    if len(esmfold_bb_str) > 0:
        esmfold_fixed_bb_selector.set_index(esmfold_bb_str)
    else:
        esmfold_fixed_bb_selector = None

    rmsd_metric = RMSDMetric()
    rmsd_metric.set_comparison_pose(design_pose)
    rmsd_metric.set_run_superimpose(True)

    design_data = {
        "name": os.path.splitext(os.path.basename(design_path))[0],
        "design_path": design_path,
        "folded_path": folded_path,
    }
    # compute all-atom design metrics
    # we have to do this workaround cuz pyrosetta does not like the esmfold/proteinmpnn output file names
    # and its a bit of a pain to rename all the files each time
    with open(folded_path) as fp:
        folded_pose = pyrosetta.Pose()
        pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(folded_pose, fp.read())

    try:
        rmsd_metric.set_rmsd_type(RMSD_ALL_HEAVY_TYPE)
        rmsd_metric.set_residue_selector(esmfold_fixed_res_selector)
        rmsd_metric.set_residue_selector_reference(fixed_res_selector)
        design_data['motif_all_atom_rmsd'] = rmsd_metric.calculate(folded_pose)
        rmsd_metric.set_residue_selector(all_res_selector)
        rmsd_metric.set_residue_selector_reference(all_res_selector)
        design_data['global_all_atom_rmsd'] = rmsd_metric.calculate(folded_pose)

        # rand_aug(folded_pose)

        # rmsd_metric.set_rmsd_type(RMSD_PROTEIN_CA_TYPE)
        if fixed_bb_selector is not None:
            rmsd_metric.set_residue_selector(esmfold_fixed_bb_selector)
            rmsd_metric.set_residue_selector_reference(fixed_bb_selector)
        else:
            rmsd_metric.set_residue_selector(esmfold_fixed_res_selector)
            rmsd_metric.set_residue_selector_reference(fixed_res_selector)
        # design_data['motif_ca_rmsd'] = rmsd_metric.calculate(folded_pose)

        # rand_aug(folded_pose)
        design_data['motif_ca_rmsd'] = calc_ca_rmsd(
            folded_pose,
            design_pose,
            fixed_res_selector if fixed_bb_selector is None else fixed_bb_selector,
            esmfold_fixed_res_selector if fixed_bb_selector is None else esmfold_fixed_bb_selector,
            fixed_res_selector if fixed_bb_selector is None else fixed_bb_selector,
            esmfold_fixed_res_selector if fixed_bb_selector is None else esmfold_fixed_bb_selector,
        )


        rmsd_metric.set_rmsd_type(RMSD_PROTEIN_BB_HEAVY_TYPE)
        design_data['motif_bb_rmsd'] = rmsd_metric.calculate(folded_pose)
        rmsd_metric.set_residue_selector(all_res_selector)
        rmsd_metric.set_residue_selector_reference(all_res_selector)
        design_data['global_bb_rmsd'] = rmsd_metric.calculate(folded_pose)
    except Exception as e:
        print(f"got error {e}, for path {folded_path}, assigning rmsds to 10")
        print(res_str, bb_str, esmfold_res_str, esmfold_bb_str)
        raise e
        design_data['motif_all_atom_rmsd'] = 10.
        design_data['global_all_atom_rmsd'] = 10.
        design_data['motif_bb_rmsd'] = 10.
        design_data['motif_ca_rmsd'] = 10.
        design_data['global_bb_rmsd'] = 10.


    pmpnn_data = []
    # rmsd_metric.set_corresponding_atoms_robust(True)
    for path in pmpnn_paths:
        # print(path)
        _data = {}
        try:
            with open(path) as fp:
                pmpnn_pose = pyrosetta.Pose()
                pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pmpnn_pose, fp.read())
            rmsd_metric.set_rmsd_type(RMSD_ALL_HEAVY_TYPE)
            rmsd_metric.set_residue_selector(esmfold_fixed_res_selector)
            rmsd_metric.set_residue_selector_reference(fixed_res_selector)
            _data['motif_all_atom_rmsd'] = rmsd_metric.calculate(pmpnn_pose)

            # rmsd_metric.set_rmsd_type(RMSD_PROTEIN_CA_TYPE)
            # rmsd_metric.set_rmsd_type(RMSD_PROTEIN_CA_TYPE)
            if fixed_bb_selector is not None:
                rmsd_metric.set_residue_selector(esmfold_fixed_bb_selector)
                rmsd_metric.set_residue_selector_reference(fixed_bb_selector)
            else:
                rmsd_metric.set_residue_selector(esmfold_fixed_res_selector)
                rmsd_metric.set_residue_selector_reference(fixed_res_selector)

            # rand_aug(pmpnn_pose)

            # # _data['motif_ca_rmsd'] = rmsd_metric.calculate(pmpnn_pose)
            _data['motif_ca_rmsd'] = calc_ca_rmsd(
                pmpnn_pose,
                design_pose,
                fixed_res_selector if fixed_bb_selector is None else fixed_bb_selector,
                esmfold_fixed_res_selector if fixed_bb_selector is None else esmfold_fixed_bb_selector,
                fixed_res_selector if fixed_bb_selector is None else fixed_bb_selector,
                esmfold_fixed_res_selector if fixed_bb_selector is None else esmfold_fixed_bb_selector,
            )

            # rand_aug(pmpnn_pose)

            rmsd_metric.set_rmsd_type(RMSD_PROTEIN_BB_HEAVY_TYPE)
            _data['motif_bb_rmsd'] = rmsd_metric.calculate(pmpnn_pose)
            rmsd_metric.set_residue_selector(all_res_selector)
            rmsd_metric.set_residue_selector_reference(all_res_selector)
            _data['global_bb_rmsd'] = rmsd_metric.calculate(pmpnn_pose)
        except Exception as e:
            print(f"got error {e}, for path {path}, assigning rmsds to 10")
            raise e
            _data = {
                'motif_all_atom_rmsd': 10.,
                'motif_bb_rmsd': 10.,
                'motif_ca_rmsd': 10.,
                'global_bb_rmsd': 10.,
            }
        pmpnn_data.append(_data)

    return design_data, pmpnn_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Compute the ")
    parser.add_argument("--esmlog", help="esmfold output log")
    parser.add_argument("--folded_folders", help="emfold folded structures")
    parser.add_argument("--samples", help="gen model samples")
    parser.add_argument("--samples_metadata", help="gen model samples")
    args = parser.parse_args()
    pyrosetta.init()

    with open(args.esmlog) as fp:
        lines = fp.readlines()
    parse_dict = parse_esm_log(lines)

    with open(args.samples_metadata) as fp:
        samples_metadata = json.load(fp)

    sc_df_dict = []
    folding_df_dict = []

    def process(parse_dict_item):
        name, sample_data = parse_dict_item
        ret_sc_df_dicts = []

        sample_path = os.path.join(
            args.samples,
            name + ".pdb"
        )
        folded_path_folder = os.path.join(
            args.folded_folders,
            name
        )
        design_folded_path = None
        pmpnn_paths = []
        pmpnn_paths_idx = []
        for path in glob.glob(os.path.join(folded_path_folder, "*")):
            if "model_name=v_48_020" in path:
                design_folded_path = path
            else:
                sample_num = int(path.split("sample=")[-1][0])
                pmpnn_paths.append(path)
                pmpnn_paths_idx.append(sample_num)

        fixed_seq_idx = samples_metadata[name]['fixed_seq_res_idx']
        fixed_seq_chain = samples_metadata[name]['fixed_seq_chain']
        fixed_bb_res_idx = samples_metadata[name]['fixed_bb_res_idx']
        fixed_bb_chain = samples_metadata[name]['fixed_bb_chain']

        try:
            design_rmsds, pmpnn_rmsds = compute_rmsds(
                sample_path,
                design_folded_path,
                pmpnn_paths,
                fixed_seq_idx,
                fixed_seq_chain,
                fixed_bb_res_idx,
                fixed_bb_chain,
            )
            print(design_rmsds)
            print(pmpnn_rmsds)
        except Exception as e:
            print(f"Error in {name}")
            raise e
        for i, _data in sample_data.items():
            if i == 0:
                folding_dict = {
                    "name": name,
                    "path": os.path.abspath(sample_path),
                    "task": samples_metadata[name]['name']
                }
                folding_dict.update(_data)
                folding_dict.update(design_rmsds)
                ret_folding_df_dict = folding_dict
            else:
                _idx = pmpnn_paths_idx.index(i)
                sc_dict = {
                    "name": name,
                    "path": os.path.abspath(pmpnn_paths[_idx]),
                    "sample": i,
                    "task": samples_metadata[name]['name']
                }
                sc_dict.update(_data)
                sc_dict.update(pmpnn_rmsds[_idx])
                ret_sc_df_dicts.append(sc_dict)

        return ret_folding_df_dict, ret_sc_df_dicts

    processed_data = p_tqdm.p_map(process, parse_dict.items(), num_cpus=16)
    for _data in processed_data:
        folding_df_dict.append(_data[0])
        sc_df_dict.extend(_data[1])

    # for name, sample_data in tqdm.tqdm(parse_dict.items()):
    #     sample_path = os.path.join(
    #         args.samples,
    #         name + ".pdb"
    #     )
    #     folded_path_folder = os.path.join(
    #         args.folded_folders,
    #         name
    #     )
    #     design_folded_path = None
    #     pmpnn_paths = []
    #     pmpnn_paths_idx = []
    #     for path in glob.glob(os.path.join(folded_path_folder, "*")):
    #         if "model_name=v_48_020" in path:
    #             design_folded_path = path
    #         else:
    #             sample_num = int(path.split("sample=")[-1][0])
    #             pmpnn_paths.append(path)
    #             pmpnn_paths_idx.append(sample_num)

    #     fixed_seq_idx = samples_metadata[name]['fixed_seq_res_idx']
    #     fixed_seq_chain = samples_metadata[name]['fixed_seq_chain']
    #     fixed_bb_res_idx = samples_metadata[name]['fixed_bb_res_idx']
    #     fixed_bb_chain = samples_metadata[name]['fixed_bb_chain']

    #     try:
    #         design_rmsds, pmpnn_rmsds = compute_rmsds(
    #             sample_path,
    #             design_folded_path,
    #             pmpnn_paths,
    #             fixed_seq_idx,
    #             fixed_seq_chain,
    #             fixed_bb_res_idx,
    #             fixed_bb_chain,
    #         )
    #         print(design_rmsds)
    #         print(pmpnn_rmsds)
    #     except Exception as e:
    #         print(f"Error in {name}")
    #         raise e
    #     for i, _data in sample_data.items():
    #         if i == 0:
    #             folding_dict = {
    #                 "name": name,
    #                 "path": os.path.abspath(sample_path),
    #                 "task": samples_metadata[name]['name']
    #             }
    #             folding_dict.update(_data)
    #             folding_dict.update(design_rmsds)
    #             folding_df_dict.append(folding_dict)
    #         else:
    #             _idx = pmpnn_paths_idx.index(i)
    #             sc_dict = {
    #                 "name": name,
    #                 "path": os.path.abspath(pmpnn_paths[_idx]),
    #                 "sample": i,
    #                 "task": samples_metadata[name]['name']
    #             }
    #             sc_dict.update(_data)
    #             sc_dict.update(pmpnn_rmsds[_idx])
    #             sc_df_dict.append(sc_dict)

    sc_df = pd.DataFrame(sc_df_dict)
    folding_df = pd.DataFrame(folding_df_dict)
    sc_df.to_csv("sc_rmsd.csv")
    folding_df.to_csv("folding_rmsd.csv")

    collapsed_motif_df = []
    collapsed_global_df = []
    for name in parse_dict.keys():
        sub_df = sc_df[sc_df['name'] == name]
        motif_min_rmsd = sub_df['motif_all_atom_rmsd'].min()
        collapsed_motif_df.append(sub_df[sub_df['motif_all_atom_rmsd'] == motif_min_rmsd])
        global_min_rmsd = sub_df['global_bb_rmsd'].min()
        collapsed_global_df.append(sub_df[sub_df['global_bb_rmsd'] == global_min_rmsd])
    collapsed_motif_df = pd.concat(collapsed_motif_df)
    collapsed_motif_df.to_csv("best_sc_motif_rmsd.csv")
    collapsed_global_df = pd.concat(collapsed_global_df)
    collapsed_global_df.to_csv("best_sc_global_rmsd.csv")

    pmpnn_pass = sc_df[
        (sc_df['motif_all_atom_rmsd'] < 1.5)
        & (sc_df['motif_bb_rmsd'] < 1)
        & (sc_df['global_bb_rmsd'] < 2)
    ]
    pmpnn_pass = pmpnn_pass.groupby("name").sample(1)
    pmpnn_task_pass = pmpnn_pass.groupby("task")
    pz_pass = folding_df[
        (folding_df['motif_all_atom_rmsd'] < 1.5)
        & (folding_df['motif_bb_rmsd'] < 1)
        & (folding_df['global_all_atom_rmsd'] < 2)
    ]
    pz_task_pass = pz_pass.groupby("task")

    with open("../num_designable.txt", 'w') as fp:
        fp.write(f"Num designable: {len(pmpnn_pass)}\n")
        fp.write(f"Num tasks passed: {len(pmpnn_task_pass)}\n")
        fp.write(f"Num consistent: {len(pz_pass)}\n")
        fp.write(f"Num tasks consistent: {len(pz_task_pass)}\n")

