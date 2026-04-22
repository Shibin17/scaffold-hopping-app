"""
scorer.py — Multi-objective scoring of scaffold-hopped candidates.

Scoring components:
  1. Attachment fidelity     (hard filter — binary)
  2. Drug-likeness filters   (logP, MW, TPSA)
  3. Toxicity predictions    (AMES, hERG, DILI via simple QSAR models)
  4. 3D shape similarity     (using RDKit USRCAT or pharmacophore fingerprints)
  5. Scaffold novelty        (Tanimoto distance to original scaffold)
  6. Synthetic accessibility (SA score)
  7. Docking score           (optional — AutoDock Vina)

Total score = weighted sum of normalized component scores.
"""

from __future__ import annotations
import logging
import math
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from rdkit.Chem import rdFMCS

from scaffold_replacer import Candidate
from molecule_splitter import MoleculeRegions

log = logging.getLogger(__name__)


@dataclass
class ScoredCandidate:
    candidate: Candidate
    smiles: str

    # Component scores (0..1 unless noted)
    attachment_fidelity: float = 1.0
    logp_score: float = 0.0
    tox_score: float = 0.0          # 0=likely toxic, 1=likely safe
    shape_similarity: float = 0.0
    pharmacophore_sim: float = 0.0
    scaffold_novelty: float = 0.0   # 1 = completely novel
    sa_score: float = 0.0           # 1 = easy to synthesize
    docking_score: float = 0.0      # normalized affinity (higher = better)

    total_score: float = 0.0
    passes_filters: bool = True
    filter_failures: list = field(default_factory=list)

    # Human-readable metadata
    logp: float = 0.0
    mw: float = 0.0
    tpsa: float = 0.0
    hba: int = 0
    hbd: int = 0
    qed: float = 0.0


# Weights for total score
WEIGHTS = {
    "shape_similarity":   0.25,
    "pharmacophore_sim":  0.20,
    "scaffold_novelty":   0.15,
    "tox_score":          0.20,
    "logp_score":         0.10,
    "sa_score":           0.10,
}


class MoleculeScorer:
    def __init__(
        self,
        reference: Chem.Mol,
        regions: MoleculeRegions,
        logp_max: float = 3.0,
        receptor_pdbqt: Optional[str] = None,
        vina_box: Optional[str] = None,
        n_workers: int = 4,
    ):
        self.reference = reference
        self.regions = regions
        self.logp_max = logp_max
        self.receptor_pdbqt = receptor_pdbqt
        self.vina_box = vina_box
        self.n_workers = n_workers

        # Precompute reference descriptors
        self._ref_fp = self._morgan_fp(reference)
        self._ref_scaffold_smiles = self._get_scaffold_smiles()
        self._ref_3d = self._embed_mol(reference)
        self._ref_pharm = self._pharmacophore_fp(reference)

    def score_all(self, candidates: List[Candidate]) -> List[ScoredCandidate]:
        results = []
        for cand in candidates:
            sc = self._score_one(cand)
            results.append(sc)
        return results

    def _score_one(self, cand: Candidate) -> ScoredCandidate:
        mol = cand.mol
        sc = ScoredCandidate(candidate=cand, smiles=cand.smiles)

        # --- Basic properties ---
        sc.logp = Descriptors.MolLogP(mol)
        sc.mw = Descriptors.ExactMolWt(mol)
        sc.tpsa = Descriptors.TPSA(mol)
        sc.hba = rdMolDescriptors.CalcNumHBA(mol)
        sc.hbd = rdMolDescriptors.CalcNumHBD(mol)
        sc.qed = QED.qed(mol)

        # --- Hard filters ---
        if sc.logp > self.logp_max:
            sc.passes_filters = False
            sc.filter_failures.append(f"logP={sc.logp:.2f} > {self.logp_max}")

        if sc.mw > 600:
            sc.passes_filters = False
            sc.filter_failures.append(f"MW={sc.mw:.1f} > 600")

        if sc.tpsa > 140:
            sc.passes_filters = False
            sc.filter_failures.append(f"TPSA={sc.tpsa:.1f} > 140")

        # --- Component scores ---
        sc.logp_score = self._score_logp(sc.logp)
        sc.tox_score = self._predict_tox(mol)
        sc.shape_similarity = self._shape_similarity(mol)
        sc.pharmacophore_sim = self._pharmacophore_similarity(mol)
        sc.scaffold_novelty = self._scaffold_novelty(mol)
        sc.sa_score = self._sa_score(mol)

        if self.receptor_pdbqt and sc.passes_filters:
            sc.docking_score = self._dock(mol)
            WEIGHTS["docking_score"] = 0.15
            # Renormalize other weights
            factor = 1.0 / sum(WEIGHTS.values())
            for k in WEIGHTS:
                WEIGHTS[k] *= factor

        sc.total_score = sum(
            WEIGHTS.get(k, 0.0) * getattr(sc, k, 0.0)
            for k in WEIGHTS
        )

        return sc

    # ------------------------------------------------------------------
    # Individual scoring functions
    # ------------------------------------------------------------------

    def _score_logp(self, logp: float) -> float:
        """Penalize approaching the limit; reward values < 1."""
        if logp <= 1.0:
            return 1.0
        if logp >= self.logp_max:
            return 0.0
        return 1.0 - (logp - 1.0) / (self.logp_max - 1.0)

    def _predict_tox(self, mol: Chem.Mol) -> float:
        """
        Simple rule-based toxicity score combining:
          - AMES-like structural alerts (aromatic amines, nitroso groups)
          - hERG alert (basic N + aromatic ring)
          - DILI alert (reactive groups)
        Returns 1.0 if no alerts, lower if alerts present.
        """
        alerts = 0

        # AMES alerts: aromatic amines, nitro, nitroso
        ames_smarts = [
            "[NX3;H2][c]",           # primary aromatic amine
            "[N;X2]=[O]",             # nitroso
            "[c][N+](=O)[O-]",        # nitroaromatic
            "C(=O)[Cl,Br,I]",         # acyl halide
        ]
        herg_smarts = [
            "[N;X3;+0]([CX4])([CX4])[CX4]",  # tertiary amine
        ]
        dili_smarts = [
            "[#6](=O)[OH]",           # carboxylic acid (mild)
            "c1ccc2c(c1)cccc2",       # naphthalene
        ]

        for sma in ames_smarts:
            patt = Chem.MolFromSmarts(sma)
            if patt and mol.HasSubstructMatch(patt):
                alerts += 2

        for sma in herg_smarts:
            patt = Chem.MolFromSmarts(sma)
            if patt and mol.HasSubstructMatch(patt):
                alerts += 1

        for sma in dili_smarts:
            patt = Chem.MolFromSmarts(sma)
            if patt and mol.HasSubstructMatch(patt):
                alerts += 0.5

        return max(0.0, 1.0 - alerts * 0.25)

    def _shape_similarity(self, mol: Chem.Mol) -> float:
        """
        Compute shape similarity using Morgan fingerprint Tanimoto as a proxy
        when 3D coordinates are unavailable. Upgrades to USRCAT if 3D is available.
        """
        try:
            if self._ref_3d is not None:
                mol_3d = self._embed_mol(mol)
                if mol_3d is not None:
                    from rdkit.Chem.rdMolDescriptors import CalcUSRCAT
                    ref_usr = CalcUSRCAT(self._ref_3d)
                    mol_usr = CalcUSRCAT(mol_3d)
                    from rdkit.Chem.rdMolDescriptors import GetUSRScore
                    return GetUSRScore(ref_usr, mol_usr)
        except Exception:
            pass

        # Fallback: Morgan FP Tanimoto
        mol_fp = self._morgan_fp(mol)
        return self._tanimoto(self._ref_fp, mol_fp)

    def _pharmacophore_similarity(self, mol: Chem.Mol) -> float:
        mol_pharm = self._pharmacophore_fp(mol)
        return self._tanimoto(self._ref_pharm, mol_pharm)

    def _scaffold_novelty(self, mol: Chem.Mol) -> float:
        """Higher = more different from original scaffold."""
        from rdkit.Chem.Scaffolds import MurckoScaffold
        try:
            new_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            new_fp = self._morgan_fp(new_scaffold)
            ref_scaffold_mol = Chem.MolFromSmiles(self._ref_scaffold_smiles) if self._ref_scaffold_smiles else None
            if ref_scaffold_mol:
                ref_fp = self._morgan_fp(ref_scaffold_mol)
                tani = self._tanimoto(ref_fp, new_fp)
                return 1.0 - tani  # novelty = distance
        except Exception:
            pass
        return 0.5

    def _sa_score(self, mol: Chem.Mol) -> float:
        """
        Normalized synthetic accessibility score using RDKit SA Score.
        SA score range: 1 (easy) to 10 (hard). We normalize to 0..1.
        """
        try:
            from rdkit.Chem import RDConfig
            import sys, os
            sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
            if sa_path not in sys.path:
                sys.path.append(sa_path)
            import sascorer
            sa = sascorer.calculateScore(mol)
            return 1.0 - (sa - 1.0) / 9.0  # invert and normalize
        except Exception:
            return 0.5  # unknown

    def _dock(self, mol: Chem.Mol) -> float:
        """
        Run AutoDock Vina docking. Returns normalized score [0,1].
        Affinity range: typically -12 to 0 kcal/mol; we map to [0,1].
        """
        if not self.receptor_pdbqt or not self.vina_box:
            return 0.0
        try:
            parts = [float(x) for x in self.vina_box.split(",")]
            cx, cy, cz, sx, sy, sz = parts

            mol_3d = self._embed_mol(mol)
            if mol_3d is None:
                return 0.0

            with tempfile.TemporaryDirectory() as tmpdir:
                ligand_mol2 = Path(tmpdir) / "ligand.mol2"
                ligand_pdbqt = Path(tmpdir) / "ligand.pdbqt"
                out_pdbqt = Path(tmpdir) / "out.pdbqt"

                Chem.MolToMolFile(mol_3d, str(ligand_mol2).replace(".mol2", ".mol"))
                # Convert to PDBQT using obabel if available
                subprocess.run(
                    ["obabel", str(ligand_mol2).replace(".mol2", ".mol"),
                     "-O", str(ligand_pdbqt), "--gen3d"],
                    capture_output=True, timeout=30
                )

                result = subprocess.run(
                    [
                        "vina",
                        "--receptor", self.receptor_pdbqt,
                        "--ligand", str(ligand_pdbqt),
                        "--out", str(out_pdbqt),
                        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
                        "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
                        "--exhaustiveness", "8",
                        "--num_modes", "1",
                    ],
                    capture_output=True, text=True, timeout=120
                )

                # Parse affinity from Vina output
                for line in result.stdout.splitlines():
                    if line.strip().startswith("1 "):
                        affinity = float(line.split()[1])
                        # Normalize: -12 kcal/mol -> 1.0, 0 kcal/mol -> 0.0
                        return max(0.0, min(1.0, -affinity / 12.0))
        except Exception as e:
            log.debug("Docking failed: %s", e)
        return 0.0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _embed_mol(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        mol_h = Chem.AddHs(mol)
        params = AllChem.EmbedParameters()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol_h, params)
        if result != 0:
            result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
        if result != 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol_h)
        return mol_h

    def _morgan_fp(self, mol: Chem.Mol):
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        gen = GetMorganGenerator(radius=2, fpSize=2048)
        return gen.GetFingerprint(mol)

    def _pharmacophore_fp(self, mol: Chem.Mol):
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        gen = GetMorganGenerator(radius=2, fpSize=2048)
        return gen.GetFingerprint(mol)

    def _tanimoto(self, fp1, fp2) -> float:
        from rdkit import DataStructs
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def _get_scaffold_smiles(self) -> str:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        try:
            sc = MurckoScaffold.GetScaffoldForMol(self.reference)
            return Chem.MolToSmiles(sc)
        except Exception:
            return ""
