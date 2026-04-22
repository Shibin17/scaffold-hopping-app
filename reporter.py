"""
reporter.py — Write scored candidates to CSV and SDF.
"""

from __future__ import annotations
import csv
import logging
from pathlib import Path
from typing import List

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

from scorer import ScoredCandidate
from molecule_splitter import MoleculeRegions

log = logging.getLogger(__name__)

CSV_FIELDS = [
    "rank", "smiles", "total_score",
    "shape_similarity", "pharmacophore_sim", "scaffold_novelty",
    "tox_score", "logp_score", "sa_score", "docking_score",
    "logp", "mw", "tpsa", "hba", "hbd", "qed",
    "passes_filters", "filter_failures",
    "source_drug", "source_drug_smiles", "fragment_smiles", "fragment_type",
    "fixed_smiles", "scaffold_replaced_smiles",
]


class Reporter:
    def __init__(self, regions: MoleculeRegions, reference: Chem.Mol):
        self.regions = regions
        self.reference = reference

    def write_csv(self, scored: List[ScoredCandidate], path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for rank, sc in enumerate(scored, 1):
                frag = sc.candidate.source_fragment
                writer.writerow({
                    "rank": rank,
                    "smiles": sc.smiles,
                    "total_score": round(sc.total_score, 4),
                    "shape_similarity": round(sc.shape_similarity, 4),
                    "pharmacophore_sim": round(sc.pharmacophore_sim, 4),
                    "scaffold_novelty": round(sc.scaffold_novelty, 4),
                    "tox_score": round(sc.tox_score, 4),
                    "logp_score": round(sc.logp_score, 4),
                    "sa_score": round(sc.sa_score, 4),
                    "docking_score": round(sc.docking_score, 4),
                    "logp": round(sc.logp, 3),
                    "mw": round(sc.mw, 2),
                    "tpsa": round(sc.tpsa, 2),
                    "hba": sc.hba,
                    "hbd": sc.hbd,
                    "qed": round(sc.qed, 4),
                    "passes_filters": sc.passes_filters,
                    "filter_failures": "; ".join(sc.filter_failures),
                    "source_drug": frag.source_drug,
                    "source_drug_smiles": frag.source_drug_smiles,
                    "fragment_smiles": frag.smiles,
                    "fragment_type": frag.frag_type,
                    "fixed_smiles": sc.candidate.fixed_smiles,
                    "scaffold_replaced_smiles": sc.candidate.scaffold_smiles,
                })
        log.info("CSV written: %s (%d rows)", path, len(scored))

    def write_sdf(self, scored: List[ScoredCandidate], path: str) -> None:
        writer = Chem.SDWriter(path)
        for rank, sc in enumerate(scored, 1):
            mol = sc.candidate.mol
            if mol is None:
                continue
            frag = sc.candidate.source_fragment
            mol.SetProp("_Name", f"scaffold_hop_{rank}")
            mol.SetProp("SMILES", sc.smiles)
            mol.SetProp("Rank", str(rank))
            mol.SetProp("TotalScore", f"{sc.total_score:.4f}")
            mol.SetProp("ShapeSimilarity", f"{sc.shape_similarity:.4f}")
            mol.SetProp("PharmSimilarity", f"{sc.pharmacophore_sim:.4f}")
            mol.SetProp("ScaffoldNovelty", f"{sc.scaffold_novelty:.4f}")
            mol.SetProp("ToxScore", f"{sc.tox_score:.4f}")
            mol.SetProp("LogP", f"{sc.logp:.3f}")
            mol.SetProp("MW", f"{sc.mw:.2f}")
            mol.SetProp("TPSA", f"{sc.tpsa:.2f}")
            mol.SetProp("QED", f"{sc.qed:.4f}")
            mol.SetProp("SourceDrug", frag.source_drug)
            mol.SetProp("FragmentSMILES", frag.smiles)
            mol.SetProp("FragmentType", frag.frag_type)
            mol.SetProp("PassesFilters", str(sc.passes_filters))
            mol.SetProp("FilterFailures", "; ".join(sc.filter_failures))
            writer.write(mol)
        writer.close()
        log.info("SDF written: %s (%d molecules)", path, len(scored))

    def write_html_report(self, scored: List[ScoredCandidate], path: str, top_n: int = 20) -> None:
        """Generate a visual HTML report with molecule images."""
        rows = []
        for rank, sc in enumerate(scored[:top_n], 1):
            mol = sc.candidate.mol
            frag = sc.candidate.source_fragment
            img_svg = self._mol_to_svg(mol)
            rows.append(f"""
            <tr>
              <td>{rank}</td>
              <td>{img_svg}</td>
              <td style="font-family:monospace;font-size:10px">{sc.smiles}</td>
              <td>{sc.total_score:.3f}</td>
              <td>{sc.shape_similarity:.3f}</td>
              <td>{sc.scaffold_novelty:.3f}</td>
              <td>{sc.tox_score:.3f}</td>
              <td>{sc.logp:.2f}</td>
              <td>{sc.mw:.1f}</td>
              <td>{frag.source_drug}</td>
              <td>{frag.frag_type}</td>
              <td>{'PASS' if sc.passes_filters else 'FAIL: ' + '; '.join(sc.filter_failures)}</td>
            </tr>""")

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Scaffold Hopping Results</title>
<style>
  body {{ font-family: Arial, sans-serif; font-size: 12px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: center; vertical-align: middle; }}
  th {{ background: #3a5a8a; color: white; }}
  tr:nth-child(even) {{ background: #f4f4f4; }}
</style>
</head><body>
<h2>Scaffold Hopping Results — Top {top_n}</h2>
<table>
<thead><tr>
  <th>Rank</th><th>Structure</th><th>SMILES</th>
  <th>Score</th><th>Shape Sim</th><th>Novelty</th>
  <th>Tox</th><th>logP</th><th>MW</th>
  <th>Source Drug</th><th>Frag Type</th><th>Filters</th>
</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</body></html>"""

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        log.info("HTML report written: %s", path)

    def _mol_to_svg(self, mol: Chem.Mol, width: int = 200, height: int = 150) -> str:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        try:
            AllChem.Compute2DCoords(mol)
            drawer.DrawMolecule(mol)
        except Exception:
            pass
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        # Inline SVG (strip XML declaration)
        svg = svg[svg.find("<svg"):]
        return svg
