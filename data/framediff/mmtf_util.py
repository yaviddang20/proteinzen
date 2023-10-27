# adapted from https://github.com/jingraham/neurips19-graph-protein-design

import os, time, gzip, urllib, json
import mmtf
from collections import defaultdict

import mdtraj as md
from Bio.PDB import PDBIO
from Bio.PDB.mmtf import MMTFParser

def download_cached(url, target_location):
    """ Download with caching """
    target_dir = os.path.dirname(target_location)
    if not os.path.isfile(target_location):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Use MMTF for speed
        response = urllib.request.urlopen(url)
        size = int(float(response.headers['Content-Length']) / 1e3)
        # print('Downloading {}, {} KB'.format(target_location, size))
        with open(target_location, 'wb') as f:
            f.write(response.read())
    return target_location


def mmtf_fetch(pdb, cache_dir='cath/mmtf/'):
    """ Retrieve mmtf record from PDB with local caching """
    mmtf_file = cache_dir + pdb + '.mmtf.gz'
    url = 'http://mmtf.rcsb.org/v1.0/full/' + pdb + '.mmtf.gz'
    mmtf_file = download_cached(url, mmtf_file)
    mmtf_record = mmtf.parse_gzip(mmtf_file)
    return mmtf_record


def compute_dssp(mmtf_path, chain):
    pdb_path = None
    try:
        # Workaround for MDtraj not supporting mmcif in their latest release.
        # MDtraj source does support mmcif https://github.com/mdtraj/mdtraj/issues/652
        # We temporarily save the mmcif as a pdb and delete it after running mdtraj.
        p = MMTFParser()
        struc = p.get_structure(mmtf_path)
        io = PDBIO()
        io.set_structure(struc)
        pdb_path = mmtf_path.replace('.mmtf', '.pdb')
        io.save(pdb_path)

        # MDtraj
        traj = md.load(pdb_path)
        # SS calculation
        pdb_ss = md.compute_dssp(traj, simplified=True)
        os.remove(pdb_path)
    except Exception as e:
        raise ValueError(f'Mdtraj failed with error {e}')
    finally:
        if pdb_path is not None and os.path.exists(pdb_path):
            os.remove(pdb_path)

    return pdb_ss


def mmtf_parse(pdb_id, target_atoms = ['N', 'CA', 'C', 'O'], cache_dir='mmtf/'):
    """ Parse mmtf file to extract C-alpha coordinates """
    # MMTF traversal derived from the specification
    # https://github.com/rcsb/mmtf/blob/master/spec.md
    A = mmtf_fetch(pdb_id, cache_dir=cache_dir)


    # Build a dictionary
    mmtf_dict = {}
    mmtf_dict['seq'] = []
    mmtf_dict['coords'] = {code:[] for code in target_atoms}

    chains = A.chain_name_list
    # print(chains)
    # print(A.__dict__.keys())
    # print([entity.keys() for entity in A.entity_list])
    chain = chains[0]
    mmtf_dict['chain'] = chain

    # Get chain of interest from Model 0
    model_ix, chain_ix, group_ix, atom_ix = 0, 0, 0, 0
    target_chain_ix, target_entity = next(
        (i, entity) for entity in A.entity_list for i in entity['chainIndexList']
        if entity['type'] == 'polymer' and A.chain_name_list[i] == chain
    )

    # Traverse chains
    num_chains = A.chains_per_model[model_ix]
    mmtf_dict['num_chains'] = num_chains
    for ii in range(num_chains):
        chain_name = A.chain_name_list[chain_ix]

        # Chain of interest?
        if chain_ix == target_chain_ix:
            mmtf_dict['seq'] = target_entity['sequence']
            coords_null = [[float('nan')] * 3] * len(mmtf_dict['seq'])
            mmtf_dict['coords'] = {code : list(coords_null) for code in target_atoms}

            # Traverse groups, storing data
            chain_group_count = A.groups_per_chain[chain_ix]
            for jj in range(chain_group_count):
                group = A.group_list[A.group_type_list[group_ix]]

                # Extend coordinate data
                seq_ix = A.sequence_index_list[group_ix]
                for code in target_atoms:
                    if code in group['atomNameList']:
                        A_ix = atom_ix + group['atomNameList'].index(code)
                        xyz = [A.x_coord_list[A_ix], A.y_coord_list[A_ix], A.z_coord_list[A_ix]]
                        mmtf_dict['coords'][code][seq_ix] = xyz

                group_atom_count = len(group['atomNameList'])
                atom_ix += group_atom_count
                group_ix += 1
            chain_ix += 1

        else:
            # Traverse groups
            chain_group_count = A.groups_per_chain[chain_ix]
            for jj in range(chain_group_count):
                group = A.group_list[A.group_type_list[group_ix]]
                atom_ix += len(group['atomNameList'])
                group_ix += 1
            chain_ix += 1

    return mmtf_dict
