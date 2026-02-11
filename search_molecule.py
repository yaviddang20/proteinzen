#!/usr/bin/env python3
"""
Search for a specific molecule in the geom_drugs_boltz dataset.

Usage:
    python search_molecule.py "SAM"                    # Search by name/keyword
    python search_molecule.py "C[S+]"                  # Search by SMILES pattern
    python search_molecule.py --smiles "C[S+](CC..."  # Search by full SMILES
    python search_molecule.py --id "molecule_id"       # Search by molecule ID
"""

import json
import sys
import argparse
from pathlib import Path


def load_manifest(manifest_path):
    """Load the manifest file."""
    with open(manifest_path) as f:
        return json.load(f)


def search_by_keyword(data, keyword):
    """Search for molecules containing keyword in SMILES or method."""
    keyword_lower = keyword.lower()
    matches = []
    
    for mol_entry in data:
        if isinstance(mol_entry, dict):
            method = str(mol_entry.get('structure', {}).get('method', '')).lower()
            mol_id = mol_entry.get('id', '')
            
            if keyword_lower in method:
                smiles = method.replace('qm9:', '') if method.startswith('qm9:') else method
                matches.append({
                    'id': mol_id,
                    'smiles': smiles,
                    'method': method
                })
    
    return matches


def search_by_smiles_pattern(data, pattern):
    """Search for molecules containing SMILES pattern."""
    matches = []
    
    for mol_entry in data:
        if isinstance(mol_entry, dict):
            method = mol_entry.get('structure', {}).get('method', '')
            mol_id = mol_entry.get('id', '')
            
            if isinstance(method, str) and method.startswith('QM9:'):
                smiles = method[4:]
                if pattern in smiles:
                    matches.append({
                        'id': mol_id,
                        'smiles': smiles
                    })
    
    return matches


def search_by_id(data, mol_id):
    """Search for a specific molecule by ID."""
    for mol_entry in data:
        if isinstance(mol_entry, dict):
            if mol_entry.get('id') == mol_id:
                method = mol_entry.get('structure', {}).get('method', '')
                smiles = method[4:] if isinstance(method, str) and method.startswith('QM9:') else method
                return {
                    'id': mol_id,
                    'smiles': smiles,
                    'method': method
                }
    return None


def main():
    parser = argparse.ArgumentParser(description='Search for molecules in geom_drugs_boltz dataset')
    parser.add_argument('query', nargs='?', help='Search query (keyword, SMILES pattern, or molecule ID)')
    parser.add_argument('--smiles', help='Search by SMILES pattern')
    parser.add_argument('--id', help='Search by molecule ID')
    parser.add_argument('--manifest', default='/datastor1/dy4652/proteinzen/geom_drugs_boltz/manifest.json',
                       help='Path to manifest.json file')
    parser.add_argument('--limit', type=int, default=20, help='Maximum number of results to show')
    
    args = parser.parse_args()
    
    # Load manifest
    print(f"Loading manifest from {args.manifest}...")
    try:
        data = load_manifest(args.manifest)
        print(f"Loaded {len(data)} molecules\n")
    except FileNotFoundError:
        print(f"Error: Manifest file not found at {args.manifest}")
        sys.exit(1)
    
    # Determine search type
    if args.id:
        print(f"Searching for molecule ID: {args.id}")
        result = search_by_id(data, args.id)
        if result:
            print(f"\n✓ Found molecule:")
            print(f"  ID: {result['id']}")
            print(f"  SMILES: {result['smiles'][:200]}...")
        else:
            print("\n✗ Molecule not found.")
    
    elif args.smiles:
        print(f"Searching for SMILES pattern: {args.smiles}")
        matches = search_by_smiles_pattern(data, args.smiles)
        print(f"\nFound {len(matches)} match(es):")
        for i, match in enumerate(matches[:args.limit], 1):
            print(f"\n{i}. ID: {match['id']}")
            print(f"   SMILES: {match['smiles'][:150]}...")
        if len(matches) > args.limit:
            print(f"\n... and {len(matches) - args.limit} more (use --limit to see more)")
    
    elif args.query:
        # Try as keyword first
        print(f"Searching for: {args.query}")
        matches = search_by_keyword(data, args.query)
        
        if not matches:
            # Try as SMILES pattern
            matches = search_by_smiles_pattern(data, args.query)
        
        if matches:
            print(f"\nFound {len(matches)} match(es):")
            for i, match in enumerate(matches[:args.limit], 1):
                print(f"\n{i}. ID: {match['id']}")
                if 'smiles' in match:
                    print(f"   SMILES: {match['smiles'][:150]}...")
                else:
                    print(f"   Method: {match.get('method', 'N/A')[:150]}...")
            if len(matches) > args.limit:
                print(f"\n... and {len(matches) - args.limit} more (use --limit to see more)")
        else:
            print("\n✗ No matches found.")
            print("\nTips:")
            print("  - Try searching by SMILES pattern: python search_molecule.py --smiles 'C[S+]'")
            print("  - Try searching by molecule ID: python search_molecule.py --id 'molecule_id'")
            print("  - The dataset uses QM9 format, so SMILES may be canonicalized")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()






