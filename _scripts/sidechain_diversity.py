"""
Generate a small ensemble of physically-plausible whole-protein side-chain
conformations for a fixed backbone + fixed ligand pose, using Rosetta's
SidechainMCMover (Dunbrack rotamer-based Monte Carlo, no custom sampler).

Packable residues are restricted to protein residues that are NOT the ligand,
NOT within `--cutoff` of the ligand ("first shell"), and NOT within `--cutoff`
of a first-shell residue ("second shell") either — so the pocket-defining
geometry stays exactly as resolved, and only the remaining, non-pocket-critical
side chains get resampled.

Requires PyRosetta (separate install, academic license via pyrosetta-installer;
not a core proteinzen dependency).

Usage:
    python sidechain_diversity.py input.pdb --out-dir ./diversity_out \
        --n-samples 5 --ntrials 10000
"""
import argparse
import json
from pathlib import Path

import pyrosetta
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import PreventRepackingRLT, OperateOnResidueSubset
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
from pyrosetta.rosetta.core.scoring import fa_rep
from pyrosetta.rosetta.protocols.simple_moves.sidechain_moves import SidechainMCMover

_BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O"}


def _residue_atom_coords(pose, res_idx):
    res = pose.residue(res_idx)
    return [res.xyz(a) for a in range(1, res.natoms() + 1)]


def _min_dist(coords_a, coords_b):
    best = float("inf")
    for a in coords_a:
        for b in coords_b:
            d = a.distance(b)
            if d < best:
                best = d
    return best


def find_excluded_residues(pose, ligand_resnums, cutoff):
    """Ligand + first-shell (within cutoff of ligand) + second-shell (within
    cutoff of a first-shell residue) protein residue indices — everything
    that should stay fixed / not be repacked.
    """
    protein_resnums = [i for i in range(1, pose.total_residue() + 1) if pose.residue(i).is_protein()]
    ligand_coords = [c for lr in ligand_resnums for c in _residue_atom_coords(pose, lr)]

    first_shell = {i for i in protein_resnums if _min_dist(_residue_atom_coords(pose, i), ligand_coords) <= cutoff}

    first_shell_coords = [c for i in first_shell for c in _residue_atom_coords(pose, i)]
    second_shell = {
        i for i in protein_resnums
        if i not in first_shell and _min_dist(_residue_atom_coords(pose, i), first_shell_coords) <= cutoff
    }

    return set(ligand_resnums) | first_shell | second_shell, first_shell, second_shell


def build_task(pose, excluded_resnums):
    task = TaskFactory.create_packer_task(pose)
    task.restrict_to_repacking()
    if excluded_resnums:
        sel = ResidueIndexSelector(",".join(str(i) for i in sorted(excluded_resnums)))
        OperateOnResidueSubset(PreventRepackingRLT(), sel).apply(pose, task)
    return task


def sidechain_rmsd(pose_a, pose_b, resnums):
    """Heavy-atom side-chain RMSD (backbone/H excluded) over the given residues."""
    sq_sum, n = 0.0, 0
    for i in resnums:
        ra, rb = pose_a.residue(i), pose_b.residue(i)
        for a in range(1, ra.natoms() + 1):
            name = ra.atom_name(a).strip()
            if name in _BACKBONE_ATOM_NAMES or name.startswith("H"):
                continue
            sq_sum += (ra.xyz(a) - rb.xyz(a)).length_squared()
            n += 1
    return (sq_sum / n) ** 0.5 if n else 0.0


def rotamer_changed(pose_a, pose_b, resnum, chi_tol_deg=20.0):
    ra, rb = pose_a.residue(resnum), pose_b.residue(resnum)
    if ra.nchi() == 0:
        return False
    for c in range(1, ra.nchi() + 1):
        d = abs(ra.chi(c) - rb.chi(c)) % 360.0
        d = min(d, 360.0 - d)
        if d > chi_tol_deg:
            return True
    return False


def generate_ensemble(pdb_path: Path, out_dir: Path, n_samples=5, ntrials=10000,
                       temperature=1.0, cutoff=5.0, max_retries=10,
                       min_frac_changed=0.02, min_sc_rmsd_between_samples=0.2,
                       residue_type_set=None):
    """residue_type_set: optional pyrosetta.rosetta.core.chemical.PoseResidueTypeSet
    (or any ResidueTypeSet) extended with on-the-fly-registered ligand types, for
    structures whose ligand isn't in Rosetta's default fa_standard params library.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if residue_type_set is not None:
        input_pose = pyrosetta.rosetta.core.import_pose.pose_from_file(residue_type_set, str(pdb_path))
    else:
        input_pose = pose_from_pdb(str(pdb_path))
    scorefxn = pyrosetta.get_score_function()
    scorefxn(input_pose)

    ligand_resnums = [i for i in range(1, input_pose.total_residue() + 1) if not input_pose.residue(i).is_protein()]
    excluded, first_shell, second_shell = find_excluded_residues(input_pose, ligand_resnums, cutoff)
    packable = [i for i in range(1, input_pose.total_residue() + 1)
                if input_pose.residue(i).is_protein() and i not in excluded]

    print(f"{pdb_path.name}: {input_pose.total_residue()} residues — "
          f"{len(ligand_resnums)} ligand, {len(first_shell)} first-shell, "
          f"{len(second_shell)} second-shell excluded, {len(packable)} packable")

    task = build_task(input_pose, excluded)

    accepted, accepted_poses = [], []
    attempts = 0
    while len(accepted) < n_samples and attempts < n_samples * max_retries:
        attempts += 1
        pose = input_pose.clone()

        mover = SidechainMCMover()
        mover.set_task(task)
        mover.setup(scorefxn)  # required — populates the mover's internal packed-residue state
        mover.set_ntrials(ntrials)
        mover.set_temperature(temperature)
        mover.set_prob_uniform(0.0)
        mover.set_prob_withinrot(0.0)
        mover.set_prob_random_pert_current(0.0)
        mover.set_preserve_detailed_balance(True)
        mover.apply(pose)

        n_changed = sum(1 for i in packable if rotamer_changed(input_pose, pose, i))
        frac_changed = n_changed / max(len(packable), 1)
        is_dup = any(sidechain_rmsd(pose, other, packable) < min_sc_rmsd_between_samples for other in accepted_poses)
        if is_dup or frac_changed < min_frac_changed:
            continue

        total_score = scorefxn(pose)
        accepted.append({
            "attempt": attempts,
            "total_score": total_score,
            "score_per_residue": total_score / pose.total_residue(),
            "fa_rep": pose.energies().total_energies()[fa_rep],
            "sc_rmsd_from_input": sidechain_rmsd(input_pose, pose, packable),
            "n_changed": n_changed,
            "n_packable": len(packable),
            "frac_changed": frac_changed,
        })
        accepted_poses.append(pose)

    for idx, (rec, pose) in enumerate(zip(accepted, accepted_poses)):
        pose.dump_pdb(str(out_dir / f"sample_{idx:03d}.pdb"))
        rec["sample_index"] = idx

    meta = {
        "input_pdb": str(pdb_path),
        "n_residues": input_pose.total_residue(),
        "ligand_resnums": ligand_resnums,
        "first_shell_resnums": sorted(first_shell),
        "second_shell_resnums": sorted(second_shell),
        "packable_resnums": packable,
        "n_samples_requested": n_samples,
        "n_samples_accepted": len(accepted),
        "attempts": attempts,
        "ntrials": ntrials,
        "temperature": temperature,
        "cutoff": cutoff,
        "samples": accepted,
    }
    with open(out_dir / "ensemble_stats.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Accepted {len(accepted)}/{n_samples} samples in {attempts} attempt(s) -> {out_dir}")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdb_path", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--ntrials", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cutoff", type=float, default=5.0,
                         help="Distance (Å) defining the ligand-contact and second-shell exclusion zones")
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--min-frac-changed", type=float, default=0.02,
                         help="Reject a sample if fewer than this fraction of packable residues changed rotamer")
    args = parser.parse_args()

    pyrosetta.init("-mute all")
    generate_ensemble(args.pdb_path, args.out_dir, args.n_samples, args.ntrials,
                       args.temperature, args.cutoff, args.max_retries, args.min_frac_changed)
