import os
import shutil
import tempfile

import pyrosetta

pyrosetta.init()

def get_hbond_atom_dict(pose: pyrosetta.Pose):
    hbond_set = pyrosetta.rosetta.core.scoring.hbonds.HBondSet(pose, bb_only=False)
    pdb_info = pose.pdb_info()

    ret = {}

    for hbond in hbond_set.hbonds():
        don_res_idx = hbond.don_res()
        don_res = pose.residue(don_res_idx)
        # get the chain and res_idx of the donor
        don_res_chain_name = pdb_info.chain(don_res_idx)
        don_res_res_idx = pdb_info.number(don_res_idx)
        # get the heavy atom bonded to the donor H
        don_heavy_atom_idx = list(don_res.bonded_neighbor(hbond.don_hatm()))[0]
        don_heavy_atom_name = don_res.atom_name(don_heavy_atom_idx).strip()
        don_key = (don_res_chain_name, don_res_res_idx, don_heavy_atom_name)

        acc_res_idx = hbond.acc_res()
        acc_res = pose.residue(acc_res_idx)
        # get the chain and res_idx of the acceptor
        acc_res_chain_name = pdb_info.chain(acc_res_idx)
        acc_res_res_idx = pdb_info.number(acc_res_idx)
        # get the heavy atom of the acceptor
        acc_heavy_atom_name = acc_res.atom_name(hbond.acc_atm()).strip()
        acc_key = (acc_res_chain_name, acc_res_res_idx, acc_heavy_atom_name)

        if don_key not in ret:
            ret[don_key] = {
                'is_donor_to': set(),
                'is_acceptor_of': set()
            }
        if acc_key not in ret:
            ret[acc_key] = {
                'is_donor_to': set(),
                'is_acceptor_of': set()
            }
        ret[don_key]['is_donor_to'].update((acc_key,))
        ret[acc_key]['is_acceptor_of'].update((don_key,))

    for outer_key in ret:
        ret[outer_key] = {
            type_key: sorted(values) #[list(t) for t in sorted(values)]
            for type_key, values in ret[outer_key].items()
        }

    return ret


def get_hbond_atom_dict_from_cif_file(path):
    if 'af-' in path or 'AF-' in path:
        # if we're parsing an AF structure we have to add a dummy entry
        # so that pyrosetta can parse it
        _, tmpfilepath = tempfile.mkstemp(".cif", text=True, dir='/tmp')
        shutil.copyfile(path, tmpfilepath)
        with open(tmpfilepath, 'a') as fp:
            fp.write("\n_dummy.entry AF-structure\n#")
        pose = pyrosetta.pose_from_file(tmpfilepath)
        ret = get_hbond_atom_dict(pose)
        os.remove(tmpfilepath)
    else:
        pose = pyrosetta.pose_from_file(path)
        ret = get_hbond_atom_dict(pose)

    return get_jsonable_dict(ret)


def get_jsonable_dict(hbond_dict):
    tuple_to_str_key = lambda x: "_".join([str(s) for s in x])
    ret = {}
    for outer_key in hbond_dict:
        str_key = tuple_to_str_key(outer_key)
        ret[str_key] = {
            type_key: [tuple_to_str_key(t) for t in values]
            for type_key, values in hbond_dict[outer_key].items()
        }
    return ret



if __name__ == '__main__':
    path = "/wynton/group/kortemme/alexjli/databases/PDB/mmCIF/u4/3u4s.cif"
    hbond_dict = get_hbond_atom_dict_from_cif_file(path)
    print(hbond_dict)
