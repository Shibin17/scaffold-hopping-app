"""
scaffold_replacer.py — Enumerate scaffold replacements.

Strategy waterfall (tries each until candidates found):
  1. Strict:      exact n_attachments + hybridization + bond type
  2. Relaxed:     exact n_attachments + bond type only
  3. Count-only:  exact n_attachments, any bond/hybridization
  4. Bioisostere: ReplaceCore + curated ring bioisostere DB (always produces results)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from itertools import permutations
from typing import List, Optional, Tuple, Set

from rdkit import Chem
from rdkit.Chem import AllChem, RWMol

from molecule_splitter import MoleculeRegions, AttachmentPoint
from fragment_library import Fragment, FragmentLibrary
import bioisosteres as bio_db

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    mol: Chem.Mol
    smiles: str
    source_fragment: Fragment
    attachment_permutation: List[int]
    fixed_smiles: str
    scaffold_smiles: str


# ── Synthetic fragment wrapper for bioisostere entries ────────────────────────

def _make_bio_fragment(smiles: str, ring_name: str, drug_name: str, drug_smi: str) -> Fragment:
    from fragment_library import Fragment as Frag
    n = smiles.count("[*]") + smiles.count("[1*]") + smiles.count("[2*]") + \
        smiles.count("[3*]") + smiles.count("[4*]")
    # Count all dummy atoms properly
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
    specs = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            for nb in atom.GetNeighbors():
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
                specs.append((nb.GetSymbol(), nb.GetHybridization(), bond.GetBondType()))
    return Frag(
        smiles=smiles,
        n_attachments=n,
        attachment_specs=specs,
        source_drug=drug_name,
        source_drug_smiles=drug_smi,
        frag_type="bioisostere",
    )


# ── Assembly helpers ──────────────────────────────────────────────────────────

def _join_at_isotopes(sc_mol: Chem.Mol, bio_mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Join sidechain mol (isotope-labeled dummies from ReplaceCore) with a
    bioisostere mol (also isotope-labeled dummies) at matching isotope values.
    """
    n_sc = sc_mol.GetNumAtoms()
    combo = AllChem.CombineMols(sc_mol, bio_mol)
    rw = RWMol(combo)

    # Build isotope → (sc_dummy_idx, bio_dummy_idx+n_sc)
    pairs: dict = {}
    for i, a in enumerate(sc_mol.GetAtoms()):
        if a.GetAtomicNum() == 0:
            pairs.setdefault(a.GetIsotope(), [None, None])[0] = i
    for j, a in enumerate(bio_mol.GetAtoms()):
        if a.GetAtomicNum() == 0:
            iso = a.GetIsotope()
            if iso in pairs:
                pairs[iso][1] = j + n_sc

    to_remove: List[int] = []
    for iso, (si, bi) in pairs.items():
        if si is None or bi is None:
            return None
        nbs_s = list(rw.GetAtomWithIdx(si).GetNeighbors())
        nbs_b = list(rw.GetAtomWithIdx(bi).GetNeighbors())
        if not nbs_s or not nbs_b:
            return None
        real_s = nbs_s[0].GetIdx()
        real_b = nbs_b[0].GetIdx()
        # Use bond type from sidechain side
        sc_bond = sc_mol.GetBondBetweenAtoms(si, real_s)
        btype = sc_bond.GetBondType() if sc_bond else Chem.BondType.SINGLE
        if rw.GetBondBetweenAtoms(real_s, real_b) is None:
            rw.AddBond(real_s, real_b, btype)
        to_remove += [si, bi]

    for idx in sorted(set(to_remove), reverse=True):
        rw.RemoveAtom(idx)
    try:
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def _dedup_sidechains(sc_mol: Chem.Mol, keep_isotopes: List[int]) -> Chem.Mol:
    """
    Remove extra dummy atoms whose isotope is not in keep_isotopes.
    When a duplicate dummy is removed, cap its real neighbor with an explicit H.
    """
    keep_set = set(keep_isotopes)
    rw = RWMol(sc_mol)
    to_remove = []
    for i, a in enumerate(sc_mol.GetAtoms()):
        if a.GetAtomicNum() == 0 and a.GetIsotope() not in keep_set:
            to_remove.append(i)
        elif a.GetAtomicNum() == 0:
            # If isotope seen already (duplicate), queue later occurrences
            iso = a.GetIsotope()
            if iso in keep_set:
                keep_set.discard(iso)   # first occurrence: keep
            else:
                to_remove.append(i)     # subsequent: remove
    for idx in sorted(to_remove, reverse=True):
        for nb in rw.GetAtomWithIdx(idx).GetNeighbors():
            nb_a = rw.GetAtomWithIdx(nb.GetIdx())
            nb_a.SetNumExplicitHs(nb_a.GetNumExplicitHs() + 1)
        rw.RemoveAtom(idx)
    try:
        m = rw.GetMol(); Chem.SanitizeMol(m, catchErrors=True); return m
    except Exception:
        return sc_mol


def _relabel_bio_smiles(bio_smi: str, iso_labels: List[int]) -> Optional[str]:
    """
    Replace [1*],[2*],... in a bioisostere SMILES with the actual isotope
    labels from ReplaceCore output, so _join_at_isotopes can match them.

    Two-pass replacement avoids collision when an iso_label value equals a
    later rank (e.g. iso_labels=[2,8]: replacing [1*]->[2*] then [2*]->[8*]
    would incorrectly turn both dummies into [8*]).
    """
    # Pass 1: replace positional tokens with temp sentinels
    smi = bio_smi
    sentinels = {}
    for rank, iso in enumerate(iso_labels, start=1):
        sentinel = f"__ISO{rank}__"
        smi = smi.replace(f"[{rank}*]", sentinel)
        sentinels[sentinel] = f"[{iso}*]"
    # Pass 2: swap sentinels for real isotope tokens
    for sentinel, label in sentinels.items():
        smi = smi.replace(sentinel, label)
    return smi


# ── Main replacer ─────────────────────────────────────────────────────────────

class ScaffoldReplacer:
    def __init__(self, regions: MoleculeRegions, library: FragmentLibrary):
        self.regions = regions
        self.library = library

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def enumerate(self) -> List[Candidate]:
        n_ap = len(self.regions.attachment_points)
        required = self._required_specs()

        # ── Strategy 1: strict ─────────────────────────────────────────
        compatible = self.library.get_compatible_fragments(required, strict_hybridization=True)
        if compatible:
            log.info("Strategy 1 (strict): %d compatible fragments", len(compatible))
            cands = self._assemble_all(compatible)
            if cands:
                return self._dedup(cands)

        # ── Strategy 2: relaxed hybridization ─────────────────────────
        compatible = self.library.get_compatible_fragments(required, strict_hybridization=False)
        if compatible:
            log.info("Strategy 2 (relaxed hyb): %d compatible fragments", len(compatible))
            cands = self._assemble_all(compatible)
            if cands:
                return self._dedup(cands)

        # ── Strategy 3: count-only ─────────────────────────────────────
        compatible = [f for f in self.library if f.n_attachments == n_ap]
        if compatible:
            log.info("Strategy 3 (count-only): %d fragments", len(compatible))
            cands = self._assemble_all(compatible)
            if cands:
                return self._dedup(cands)

        # ── Strategy 4: bioisostere DB ─────────────────────────────────
        log.info("Strategies 1-3 produced 0 candidates → using bioisostere fallback")
        cands = self._bioisostere_fallback()
        if cands:
            return self._dedup(cands)

        # ── Strategy 5: force any-n bioisostere ────────────────────────
        log.info("Trying bioisostere fallback with any n_attachments")
        cands = self._bioisostere_fallback(force_n=None)
        return self._dedup(cands)

    # ------------------------------------------------------------------
    # Strategies 1-3 assembly
    # ------------------------------------------------------------------

    def _required_specs(self):
        return [
            (ap.scaffold_atom_symbol, ap.scaffold_atom_hybridization, ap.bond_type)
            for ap in self.regions.attachment_points
        ]

    def _assemble_all(self, fragments: List[Fragment]) -> List[Candidate]:
        results = []
        for frag in fragments:
            frag_mol = frag.mol()
            if frag_mol is None:
                continue
            dummies = [a.GetIdx() for a in frag_mol.GetAtoms() if a.GetAtomicNum() == 0]
            if len(dummies) != len(self.regions.attachment_points):
                continue
            for perm in permutations(range(len(dummies))):
                mol = self._assemble(frag_mol, dummies, perm)
                if mol is None:
                    continue
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    continue
                smi = Chem.MolToSmiles(mol)
                if smi:
                    results.append(Candidate(
                        mol=mol, smiles=smi,
                        source_fragment=frag,
                        attachment_permutation=list(perm),
                        fixed_smiles=self.regions.fixed_smiles_with_dummies,
                        scaffold_smiles=frag.smiles,
                    ))
        return results

    def _assemble(
        self,
        frag_mol: Chem.Mol,
        frag_dummies: List[int],
        perm: tuple,
    ) -> Optional[Chem.Mol]:
        original = self.regions.original_mol
        scaffold_idxs = self.regions.scaffold_atom_indices

        rw = RWMol(original)
        orig_n = rw.GetNumAtoms()

        # Map frag non-dummy atoms into rw
        non_dummy = [i for i in range(frag_mol.GetNumAtoms())
                     if frag_mol.GetAtomWithIdx(i).GetAtomicNum() != 0]
        old_to_new = {}
        for fi in non_dummy:
            fa = frag_mol.GetAtomWithIdx(fi)
            na = Chem.Atom(fa.GetAtomicNum())
            na.SetFormalCharge(fa.GetFormalCharge())
            na.SetIsAromatic(fa.GetIsAromatic())
            na.SetNumExplicitHs(fa.GetNumExplicitHs())
            old_to_new[fi] = rw.AddAtom(na)

        for bond in frag_mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in old_to_new and a2 in old_to_new:
                rw.AddBond(old_to_new[a1], old_to_new[a2], bond.GetBondType())

        for ap_idx, ap in enumerate(self.regions.attachment_points):
            frag_dummy_idx = frag_dummies[perm[ap_idx]]
            dummy_atom = frag_mol.GetAtomWithIdx(frag_dummy_idx)
            nbs = [n for n in dummy_atom.GetNeighbors() if n.GetAtomicNum() != 0]
            if not nbs:
                return None
            frag_conn = nbs[0].GetIdx()
            if frag_conn not in old_to_new:
                return None
            fixed_rw = ap.fixed_atom_idx
            frag_rw  = old_to_new[frag_conn]
            if rw.GetBondBetweenAtoms(fixed_rw, frag_rw) is None:
                rw.AddBond(fixed_rw, frag_rw, ap.bond_type)

        for idx in sorted(scaffold_idxs, reverse=True):
            rw.RemoveAtom(idx)

        try:
            mol = rw.GetMol()
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Strategy 4: bioisostere fallback
    # ------------------------------------------------------------------

    def _bioisostere_fallback(self, force_n: Optional[int] = None) -> List[Candidate]:
        """
        Use RDKit ReplaceCore to extract the fixed region's attachment topology,
        then join with curated ring bioisosteres.

        If the scaffold has pendant groups causing duplicate isotopes (two fixed
        atoms connecting to the same scaffold atom), automatically retries with
        the ring-only portion of the scaffold.
        """
        mol          = self.regions.original_mol
        scaffold_sma = self.regions.scaffold_smarts

        cands = self._bioisostere_with_smarts(mol, scaffold_sma, force_n)
        if cands:
            return cands

        # Retry: extract just the ring atoms from the scaffold SMARTS
        ring_sma = self._ring_only_smarts()
        if ring_sma and ring_sma != scaffold_sma:
            log.info("Retrying bioisostere fallback with ring-only SMARTS: %s", ring_sma)
            cands = self._bioisostere_with_smarts(mol, ring_sma, force_n)

        return cands

    def _bioisostere_with_smarts(
        self,
        mol: Chem.Mol,
        scaffold_sma: str,
        force_n: Optional[int],
    ) -> List[Candidate]:
        core = Chem.MolFromSmarts(scaffold_sma)
        if core is None or not mol.HasSubstructMatch(core):
            return []

        sidechains = AllChem.ReplaceCore(mol, core, labelByIndex=True)
        if sidechains is None:
            return []

        iso_labels = sorted(
            [a.GetIsotope() for a in sidechains.GetAtoms() if a.GetAtomicNum() == 0]
        )
        n_iso = len(iso_labels)
        if n_iso == 0:
            return []

        # If duplicates (two fixed atoms on same scaffold atom), de-dup sidechains
        if len(set(iso_labels)) != n_iso:
            iso_labels = sorted(set(iso_labels))
            n_iso = len(iso_labels)
            sidechains = _dedup_sidechains(sidechains, iso_labels)
            log.info("Duplicate isotopes detected; using unique set: %s", iso_labels)

        log.info("Bioisostere fallback: %d attachment isotopes %s", n_iso, iso_labels)

        entries = bio_db.get_bioisosteres(n_iso)
        if not entries:
            for n_try in sorted({2, 1, n_iso - 1}, reverse=True):
                if n_try < 1:
                    continue
                entries = bio_db.get_bioisosteres(n_try)
                if entries:
                    iso_labels = iso_labels[:n_try]
                    break
        if not entries:
            return []

        results: List[Candidate] = []
        seen: Set[str] = set()

        for bio_smi, ring_name, drug_name in entries:
            relabeled = _relabel_bio_smiles(bio_smi, iso_labels)
            bio_mol   = Chem.MolFromSmiles(relabeled)
            if bio_mol is None:
                continue
            mol_out = _join_at_isotopes(sidechains, bio_mol)
            if mol_out is None:
                continue
            try:
                Chem.SanitizeMol(mol_out)
            except Exception:
                continue
            smi = Chem.MolToSmiles(mol_out)
            if not smi or smi in seen or mol_out.GetNumHeavyAtoms() < 5:
                continue
            seen.add(smi)
            frag = _make_bio_fragment(bio_smi, ring_name, drug_name, "")
            if frag is None:
                continue
            results.append(Candidate(
                mol=mol_out, smiles=smi,
                source_fragment=frag,
                attachment_permutation=[],
                fixed_smiles=self.regions.fixed_smiles_with_dummies,
                scaffold_smiles=ring_name,
            ))

        log.info("Bioisostere fallback produced %d candidates", len(results))
        return results

    def _ring_only_smarts(self) -> str:
        """
        Return a SMARTS covering only the ring atoms of the scaffold match.
        Used when pendant scaffold atoms cause duplicate isotope labels.

        Uses mol.GetRingInfo() (the actual molecule) to avoid AttributeError
        that occurs when calling GetRingInfo() on a SMARTS query mol.
        """
        sma = self.regions.scaffold_smarts
        mol = self.regions.original_mol
        core = Chem.MolFromSmarts(sma)
        if core is None:
            return sma

        match = mol.GetSubstructMatch(core)
        if not match:
            return sma

        scaffold_in_mol: Set[int] = set(match)

        # Use the real molecule's ring info (SMARTS mols may not support it)
        mol_ring_info = mol.GetRingInfo()
        if not mol_ring_info.NumRings():
            return sma

        ring_in_scaffold: Set[int] = set()
        for ring in mol_ring_info.AtomRings():
            overlap = set(ring) & scaffold_in_mol
            if overlap:
                ring_in_scaffold.update(overlap)

        if not ring_in_scaffold or ring_in_scaffold == scaffold_in_mol:
            return sma  # Scaffold is already purely a ring (or has no ring atoms)

        rw = RWMol()
        old_to_new: dict = {}
        for idx in sorted(ring_in_scaffold):
            atom = mol.GetAtomWithIdx(idx)
            na = Chem.Atom(atom.GetAtomicNum())
            na.SetIsAromatic(atom.GetIsAromatic())
            old_to_new[idx] = rw.AddAtom(na)
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in old_to_new and a2 in old_to_new:
                rw.AddBond(old_to_new[a1], old_to_new[a2], bond.GetBondType())
        try:
            Chem.SanitizeMol(rw)
            return Chem.MolToSmarts(rw.GetMol())
        except Exception:
            return sma

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _dedup(self, candidates: List[Candidate]) -> List[Candidate]:
        seen: Set[str] = set()
        out  = []
        for c in candidates:
            if c.smiles not in seen:
                seen.add(c.smiles)
                out.append(c)
        log.info("Unique valid candidates: %d", len(out))
        return out
