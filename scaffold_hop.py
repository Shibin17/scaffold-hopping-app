"""
scaffold_hop.py — Main entry point for scaffold hopping pipeline.

Usage:
    python scaffold_hop.py \
        --reference "CC(=O)Nc1ccc(O)cc1" \
        --scaffold_smarts "[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1" \
        --drug_db drugs.smi \
        --output results.csv \
        --top_n 20
"""

import argparse
import logging
import sys
from pathlib import Path

from fragment_library import FragmentLibrary
from molecule_splitter import MoleculeSplitter
from scaffold_replacer import ScaffoldReplacer
from scorer import MoleculeScorer
from reporter import Reporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Scaffold Hopping Tool")
    p.add_argument("--reference", required=True, help="Reference molecule SMILES or path to .sdf")
    p.add_argument("--scaffold_smarts", required=True,
                   help="SMARTS pattern defining the replaceable scaffold region")
    p.add_argument("--drug_db", required=True,
                   help="Path to approved drug SMILES file (one SMILES per line, optional name)")
    p.add_argument("--output", default="results.csv", help="Output CSV path")
    p.add_argument("--top_n", type=int, default=50, help="Number of top candidates to return")
    p.add_argument("--logp_max", type=float, default=3.0, help="Maximum allowed logP")
    p.add_argument("--min_attachment_points", type=int, default=1)
    p.add_argument("--max_attachment_points", type=int, default=4)
    p.add_argument("--receptor_pdbqt", default=None,
                   help="Optional: receptor .pdbqt for AutoDock Vina docking")
    p.add_argument("--vina_box", default=None,
                   help="Docking box as 'cx,cy,cz,sx,sy,sz' (center + size in Angstroms)")
    p.add_argument("--n_workers", type=int, default=4, help="Parallel workers for scoring")
    return p.parse_args()


def load_reference(reference: str):
    from rdkit import Chem
    if Path(reference).exists() and reference.endswith(".sdf"):
        suppl = Chem.SDMolSupplier(reference, removeHs=False)
        mol = next(m for m in suppl if m is not None)
        log.info("Loaded reference from SDF: %s", reference)
    else:
        mol = Chem.MolFromSmiles(reference)
        if mol is None:
            raise ValueError(f"Invalid reference SMILES: {reference}")
        log.info("Loaded reference from SMILES")
    return mol


def main():
    args = parse_args()

    log.info("=== Scaffold Hopping Pipeline ===")

    # Step 1: Load reference and split into fixed + scaffold regions
    ref_mol = load_reference(args.reference)
    splitter = MoleculeSplitter(ref_mol, args.scaffold_smarts)
    regions = splitter.split()
    log.info(
        "Fixed atoms: %d | Scaffold atoms: %d | Attachment points: %d",
        len(regions.fixed_atom_indices),
        len(regions.scaffold_atom_indices),
        len(regions.attachment_points),
    )

    # Step 2: Build fragment library from approved drugs
    frag_lib = FragmentLibrary(
        drug_db_path=args.drug_db,
        min_attachments=args.min_attachment_points,
        max_attachments=args.max_attachment_points,
    )
    frag_lib.build()
    log.info("Fragment library: %d fragments from approved drugs", len(frag_lib))

    # Step 3 + 4: Match attachments and replace scaffold
    replacer = ScaffoldReplacer(regions, frag_lib)
    candidates = replacer.enumerate()
    log.info("Generated %d candidate molecules before filtering", len(candidates))

    # Step 5: Score and filter
    scorer = MoleculeScorer(
        reference=ref_mol,
        regions=regions,
        logp_max=args.logp_max,
        receptor_pdbqt=args.receptor_pdbqt,
        vina_box=args.vina_box,
        n_workers=args.n_workers,
    )
    scored = scorer.score_all(candidates)
    scored.sort(key=lambda x: x.total_score, reverse=True)
    top = scored[: args.top_n]
    log.info("Top %d candidates selected", len(top))

    # Step 6: Report
    reporter = Reporter(regions, ref_mol)
    reporter.write_csv(top, args.output)
    reporter.write_sdf(top, args.output.replace(".csv", ".sdf"))
    log.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
