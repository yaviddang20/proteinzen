"""
Helper script to extract molecular data from geom_drugs_boltz .npz files.

This script demonstrates how to extract:
- Atom coordinates as a numpy array
- Atom identities (elements) as a list
- Bond matrix as a numpy array
"""

import numpy as np
from pathlib import Path
from typing import Tuple


def load_molecule_data(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load molecular data from a .npz file.
    
    Parameters
    ----------
    npz_path : Path
        Path to the .npz structure file
        
    Returns
    -------
    coordinates : np.ndarray
        Shape (N, 3) array of atom coordinates
    atom_elements : np.ndarray
        Shape (N,) array of atomic numbers (element identities)
    bond_matrix : np.ndarray
        Shape (N, N) binary adjacency matrix where bond_matrix[i, j] = 1 if atoms i and j are bonded
    """
    # Load the .npz file
    data = np.load(npz_path)
    
    # Extract atoms and bonds
    atoms = data['atoms']
    bonds = data['bonds']
    
    # Get coordinates (shape: N x 3)
    coordinates = atoms['coords']
    
    # Get atom elements (atomic numbers)
    atom_elements = atoms['element']
    
    # Create bond matrix
    num_atoms = len(atoms)
    bond_matrix = np.zeros((num_atoms, num_atoms), dtype=np.int32)
    
    # Fill in bonds (bonds are stored as (atom_1, atom_2, bond_type))
    for bond in bonds:
        atom_1 = bond['atom_1']
        atom_2 = bond['atom_2']
        # Make it symmetric (undirected graph)
        bond_matrix[atom_1, atom_2] = 1
        bond_matrix[atom_2, atom_1] = 1
    
    return coordinates, atom_elements, bond_matrix


def load_molecule_data_with_bond_types(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load molecular data with bond type information.
    
    Parameters
    ----------
    npz_path : Path
        Path to the .npz structure file
        
    Returns
    -------
    coordinates : np.ndarray
        Shape (N, 3) array of atom coordinates
    atom_elements : np.ndarray
        Shape (N,) array of atomic numbers (element identities)
    bond_matrix : np.ndarray
        Shape (N, N) matrix where bond_matrix[i, j] = bond_type if atoms i and j are bonded, 0 otherwise
        Bond types: 1=single, 2=double, 3=triple, 4=aromatic, etc.
    """
    # Load the .npz file
    data = np.load(npz_path)
    
    # Extract atoms and bonds
    atoms = data['atoms']
    bonds = data['bonds']
    
    # Get coordinates (shape: N x 3)
    coordinates = atoms['coords']
    
    # Get atom elements (atomic numbers)
    atom_elements = atoms['element']
    
    # Create bond matrix with bond types
    num_atoms = len(atoms)
    bond_matrix = np.zeros((num_atoms, num_atoms), dtype=np.int32)
    
    # Fill in bonds with bond types
    for bond in bonds:
        atom_1 = bond['atom_1']
        atom_2 = bond['atom_2']
        bond_type = bond['type']
        # Make it symmetric (undirected graph)
        bond_matrix[atom_1, atom_2] = bond_type
        bond_matrix[atom_2, atom_1] = bond_type
    
    return coordinates, atom_elements, bond_matrix


def get_atom_symbols(atomic_numbers: np.ndarray) -> list:
    """
    Convert atomic numbers to element symbols.
    
    Parameters
    ----------
    atomic_numbers : np.ndarray
        Array of atomic numbers
        
    Returns
    -------
    list
        List of element symbols (e.g., ['C', 'O', 'N', ...])
    """
    # Common elements mapping (atomic number -> symbol)
    element_map = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
        9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
        16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
        23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
        30: 'Zn', 35: 'Br', 53: 'I'
    }
    return [element_map.get(int(num), f'X{int(num)}') for num in atomic_numbers]


if __name__ == "__main__":
    # Example usage
    example_file = Path("geom_drugs_boltz/structures/00/000a5b33aa52f76ca45411507db1135a595995108e781e12b8327028d9459d85.npz")
    
    if example_file.exists():
        print(f"Loading molecule from: {example_file}")
        print("-" * 60)
        
        # Load basic data
        coords, elements, bond_matrix = load_molecule_data(example_file)
        
        print(f"Number of atoms: {len(coords)}")
        print(f"\nCoordinates shape: {coords.shape}")
        print(f"First 5 atom coordinates:\n{coords[:5]}")
        
        print(f"\nAtom elements (atomic numbers): {elements}")
        atom_symbols = get_atom_symbols(elements)
        print(f"Atom symbols: {atom_symbols}")
        
        print(f"\nBond matrix shape: {bond_matrix.shape}")
        print(f"Number of bonds: {bond_matrix.sum() // 2}")  # Divide by 2 because matrix is symmetric
        print(f"Bond matrix (first 10x10):\n{bond_matrix[:10, :10]}")
        
        # Load with bond types
        print("\n" + "=" * 60)
        print("With bond type information:")
        coords, elements, bond_matrix_typed = load_molecule_data_with_bond_types(example_file)
        print(f"Bond types present: {np.unique(bond_matrix_typed[bond_matrix_typed > 0])}")
        print(f"Bond type matrix (first 10x10):\n{bond_matrix_typed[:10, :10]}")
        
    else:
        print(f"Example file not found: {example_file}")
        print("\nUsage:")
        print("  from extract_molecule_data import load_molecule_data")
        print("  coords, elements, bond_matrix = load_molecule_data(path_to_npz_file)")

