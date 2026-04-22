"""
demo.py — End-to-end demo with paracetamol as reference molecule.

Paracetamol: CC(=O)Nc1ccc(O)cc1
Scaffold to replace: the para-substituted benzene ring
Preserved: the acetamide (NHAc) and hydroxyl (OH) groups

Run:
    python demo.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inline mini drug database (5 approved drugs) for the demo.
# In production, use a full approved drug SMILES file (e.g., DrugBank).
# ---------------------------------------------------------------------------
DEMO_DRUGS = """\
CC(=O)Nc1ccc(O)cc1 Paracetamol
c1ccc2ncccc2c1 Quinoline_scaffold
C1=CC=C2C=CC=NC2=C1 Isoquinoline_scaffold
c1ccncc1 Pyridine
c1ccc2[nH]cccc2c1 Indole
c1ccoc1 Furan
c1ccsc1 Thiophene
C1=CNC=C1 Pyrrole
c1cnccn1 Pyrimidine
c1ccnc2ncccc12 Naphthyridine
O=C1CCCC(=O)N1 Caprolactam
C1CCCCC1 Cyclohexane
C1=CC=NC=C1 Pyridine_alt
c1ccc2c(c1)CCCC2 Tetrahydronaphthalene
C1CCNCC1 Piperidine
C1COCCN1 Morpholine
C1CCNC1 Pyrrolidine
C1CNCCN1 Piperazine
c1cc2ccccc2[nH]1 Indole2
c1ccc2ccncc2c1 Quinoline2
"""

REFERENCE_SMILES = "CC(=O)Nc1ccc(O)cc1"
# Replace the benzene ring; keep NHAc and OH substituents
SCAFFOLD_SMARTS = "c1ccc(cc1)"


def make_demo_drug_db(path: str = "demo_drugs.smi"):
    with open(path, "w") as f:
        f.write(DEMO_DRUGS)
    log.info("Demo drug database written: %s", path)
    return path


def run_demo():
    from rdkit import Chem

    drug_db = make_demo_drug_db()

    # Import pipeline modules
    from molecule_splitter import MoleculeSplitter
    from fragment_library import FragmentLibrary
    from scaffold_replacer import ScaffoldReplacer
    from scorer import MoleculeScorer
    from reporter import Reporter

    log.info("Reference: %s", REFERENCE_SMILES)
    log.info("Scaffold SMARTS: %s", SCAFFOLD_SMARTS)

    ref_mol = Chem.MolFromSmiles(REFERENCE_SMILES)

    # Step 1: Split
    splitter = MoleculeSplitter(ref_mol, SCAFFOLD_SMARTS)
    regions = splitter.split()
    log.info("Attachment points: %d", len(regions.attachment_points))
    for i, ap in enumerate(regions.attachment_points):
        log.info(
            "  AP%d: fixed=%s(%s) --[%s]--> scaffold=%s(%s)",
            i,
            ap.fixed_atom_symbol, ap.fixed_atom_hybridization,
            ap.bond_type,
            ap.scaffold_atom_symbol, ap.scaffold_atom_hybridization,
        )

    # Step 2: Fragment library
    lib = FragmentLibrary(drug_db_path=drug_db, min_attachments=1, max_attachments=4, use_cache=False)
    lib.build()
    log.info("Library size: %d fragments", len(lib))

    # Step 3+4: Enumerate replacements
    replacer = ScaffoldReplacer(regions, lib)
    candidates = replacer.enumerate()
    log.info("Candidates before scoring: %d", len(candidates))

    if not candidates:
        log.warning("No candidates generated — try relaxing attachment constraints or expanding drug DB.")
        return

    # Step 5: Score
    scorer = MoleculeScorer(
        reference=ref_mol,
        regions=regions,
        logp_max=3.0,
        n_workers=1,
    )
    scored = scorer.score_all(candidates)
    scored.sort(key=lambda x: x.total_score, reverse=True)

    # Step 6: Report
    reporter = Reporter(regions, ref_mol)
    reporter.write_csv(scored, "demo_results.csv")
    reporter.write_sdf(scored, "demo_results.sdf")
    reporter.write_html_report(scored, "demo_results.html", top_n=min(20, len(scored)))

    log.info("\n=== TOP 5 RESULTS ===")
    for rank, sc in enumerate(scored[:5], 1):
        frag = sc.candidate.source_fragment
        log.info(
            "[%d] score=%.3f  shape=%.3f  novelty=%.3f  tox=%.3f  logP=%.2f  MW=%.1f  "
            "drug=%s  frag_type=%s\n    SMILES: %s",
            rank, sc.total_score, sc.shape_similarity, sc.scaffold_novelty,
            sc.tox_score, sc.logp, sc.mw,
            frag.source_drug, frag.frag_type,
            sc.smiles,
        )


if __name__ == "__main__":
    run_demo()
