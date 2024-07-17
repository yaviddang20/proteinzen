from io import StringIO

from rdkit import Chem
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain
from Bio.PDB.Structure import Structure
from Bio.PDB.Polypeptide import is_aa, is_nucleic
from Bio.PDB.PDBIO import PDBIO

from Bio.Data.PDBData import protein_letters_3to1, nucleic_letters_3to1
from rdkit.Chem.rdchem import BondType as BT

aa_struct_smiles_map = {
    "A":"C[C@@H](C(=O)O)N",
    "C":"C([C@@H](C(=O)O)N)S",
    "D":"OC(=O)C[C@@H](C(=O)O)N",
    "E":"C(CC(=O)O)[C@@H](C(=O)O)N",
    "F":"c1ccc(cc1)C[C@@H](C(=O)O)N",
    "G":"NCC(=O)O",
    "H":"C1=C(NC=N1)C[C@@H](C(=O)O)N",
    "I":"CC[C@H](C)[C@@H](C(=O)O)N",
    "K":"NCCCC[C@@H](C(=O)O)N",
    "L":"CC(C)C[C@@H](C(=O)O)N",
    "M":"CSCC[C@@H](C(=O)O)N",
    "N":"C([C@@H](C(=O)O)N)C(=O)N",
    "P":"C1C[C@H](NC1)C(=O)O",
    "Q":"C(CC(=O)N)[C@@H](C(=O)O)N",
    "R":"C(C[C@@H](C(=O)O)N)CN=C(N)N",
    "S":"C([C@@H](C(=O)O)N)O",
    "T":"C[C@H]([C@@H](C(=O)O)N)O",
    "V":"CC(C)[C@@H](C(=O)O)N",
    "W":"C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N",
    "Y":"C1=CC(=CC=C1C[C@@H](C(=O)O)N)O",
}

def atomize_residue(res: Residue, smiles=None):
    if smiles is None:
        if is_aa(res, standard=True):
            smiles = aa_struct_smiles_map[protein_letters_3to1[f"{res.get_resname():<3s}"]]
        elif is_nucleic(res, standard=True):
            raise NotImplementedError("TODO: standard nucleic acids")
        else:
            raise ValueError(f"smiles is required for residue {res} as it is not a standard AA or nucleic acid")
    smiles_mol = Chem.MolFromSmiles(smiles)
    smiles_mol_single_bond = Chem.MolFromSmiles(smiles)
    for bond in smiles_mol_single_bond.GetBonds():
        bond.SetBondType(BT.SINGLE)

    dummy_chain = Chain(0)
    dummy_chain.add(res)
    struct = Structure(0)
    struct.add(dummy_chain)

    pdb_io = PDBIO()
    pdb_io.set_structure(struct)
    str_io = StringIO()
    pdb_io.save(str_io)

    pdb_mol = Chem.MolFromPDBBlock(str_io.getvalue())
    for bond in pdb_mol.GetBonds():
        bond.SetBondType(BT.SINGLE)



    ff = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField()


pdb_str = """
MODEL
ATOM    396  N   PRO A  48      18.606 -12.150  25.891  1.00  0.00         A N
ATOM    397  CA  PRO A  48      17.271 -12.065  25.293  1.00  0.00         A C
ATOM    398  C   PRO A  48      16.761 -10.630  25.189  1.00  0.00         A C
ATOM    399  O   PRO A  48      16.028 -10.314  24.253  1.00  0.00         A O
ATOM    400  CB  PRO A  48      16.403 -12.886  26.250  1.00  0.00         A C
ATOM    401  CG  PRO A  48      17.233 -14.082  26.588  1.00  0.00         A C
ATOM    402  CD  PRO A  48      18.553 -13.610  27.126  1.00  0.00         A C
TER
"""
pdb_str = """
MODEL
ATOM    421  N   ARG A  51      18.483  -9.088  22.301  1.00  0.00         A N
ATOM    422  CA  ARG A  51      17.902  -9.654  21.088  1.00  0.00         A C
ATOM    423  C   ARG A  51      16.649  -8.890  20.673  1.00  0.00         A C
ATOM    424  O   ARG A  51      16.406  -8.600  19.502  1.00  0.00         A O
ATOM    425  CB  ARG A  51      17.570 -11.134  21.290  1.00  0.00         A C
ATOM    426  CG  ARG A  51      17.075 -11.832  20.033  1.00  0.00         A C
ATOM    427  CD  ARG A  51      16.916 -13.331  20.246  1.00  0.00         A C
ATOM    428  NE  ARG A  51      16.217 -13.628  21.492  1.00  0.00         A N
ATOM    429  CZ  ARG A  51      15.362 -14.633  21.661  1.00  0.00         A C
ATOM    430  NH1 ARG A  51      15.083 -15.461  20.661  1.00  0.00         A N
ATOM    431  NH2 ARG A  51      14.781 -14.812  22.838  1.00  0.00         A N
TER
"""

if __name__ == '__main__':
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Chem import AllChem
    from rdkit.Chem import Draw
    # mol = Chem.MolFromPDBBlock(pdb_str)
    mol = Chem.MolFromSmiles("[NH3+]CC(=O)NCC(=O)NCC(=O)O")
    dos = Draw.MolDrawOptions()
    dos.addAtomIndices=True
    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, options=dos)
    img.save("test.png")
    exit()

    # from io import StringIO

    # str_io = StringIO(pdb_str)
    # from Bio.PDB.PDBParser import PDBParser
    # from Bio.PDB.PDBIO import PDBIO

    # parser = PDBParser()
    # struct = parser.get_structure("", str_io)

    # pdb_io = PDBIO()
    # pdb_io.set_structure(struct)

    # str_io = StringIO()
    # pdb_io.save(str_io)
    # mol = Chem.MolFromPDBBlock(str_io.getvalue(), proximityBonding=False)
    # Chem.rdDetermineBonds.DetermineConnectivity(mol)
    # img = Draw.MolToImage(mol, options=dos)
    # img.save("test2.png")

    # # mol_smiles = Chem.MolFromSmiles(aa_struct_smiles_map["P"])
    # mol_smiles = Chem.MolFromSmiles(aa_struct_smiles_map["R"])
    mol_smiles = Chem.MolFromSmiles(aa_struct_smiles_map["R"])
    mol_single_bonds = Chem.MolFromSmiles(aa_struct_smiles_map["R"])
    img = Draw.MolToImage(mol_smiles, options=dos)
    img.save("test3.png")

    # mol = AllChem.AssignBondOrdersFromTemplate(mol_smiles, mol)

    # from rdkit.Chem import AllChem
    # newMol = AllChem.AssignBondOrdersFromTemplate(mol_smiles, mol)
    # img = Draw.MolToImage(newMol, options=dos)
    # img.save("test3p5.png")

    from rdkit.Chem.rdchem import BondType as BT
    for bond in mol_smiles.GetBonds():
        bond.SetBondType(BT.SINGLE)

    match_params = Chem.rdchem.SubstructMatchParameters()

    hit_ats = list(mol_smiles.GetSubstructMatch(mol))
    print(hit_ats)
    hit_bonds = []
    for bond in mol.GetBonds():
        aid1 = hit_ats[bond.GetBeginAtomIdx()]
        aid2 = hit_ats[bond.GetEndAtomIdx()]
        hit_bonds.append(mol_smiles.GetBondBetweenAtoms(aid1,aid2).GetIdx())
    d = Draw.rdMolDraw2D.MolDraw2DCairo(500, 500) # or MolDraw2DCairo to get PNGs
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(d, mol_smiles, highlightAtoms=hit_ats,
                                    highlightBonds=hit_bonds)
    d.WriteDrawingText("test4.png")