"""
fragment_library.py — Build and query a fragment library from approved drugs.

Fragmentation strategies:
  1. BRICS (default) — breaks retrosynthetically meaningful bonds
  2. Murcko scaffold decomposition
  3. Ring system extraction

Each fragment is stored with its attachment-point metadata so that during
scaffold replacement we can enforce exact bond-type / hybridization matching.
"""

from __future__ import annotations
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from rdkit import Chem
from rdkit.Chem import BRICS, AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

log = logging.getLogger(__name__)


@dataclass
class Fragment:
    smiles: str
    n_attachments: int
    # For each attachment point (dummy atom [*:n]): (atom symbol, hybridization, bond type)
    attachment_specs: List[Tuple[str, Chem.HybridizationType, Chem.BondType]]
    source_drug: str
    source_drug_smiles: str
    frag_type: str  # "brics", "murcko", "ring"

    def mol(self) -> Optional[Chem.Mol]:
        return Chem.MolFromSmiles(self.smiles)


class FragmentLibrary:
    """
    Builds a deduped library of fragments from approved drugs.
    Fragments are keyed by canonical SMILES for deduplication.
    """

    CACHE_FILE = ".fragment_library_cache.pkl"

    def __init__(
        self,
        drug_db_path: str,
        min_attachments: int = 1,
        max_attachments: int = 4,
        use_cache: bool = True,
    ):
        self.drug_db_path = Path(drug_db_path)
        self.min_attachments = min_attachments
        self.max_attachments = max_attachments
        self.use_cache = use_cache
        self._fragments: Dict[str, Fragment] = {}  # canonical SMILES -> Fragment

    def __len__(self) -> int:
        return len(self._fragments)

    def __iter__(self):
        return iter(self._fragments.values())

    def build(self) -> None:
        cache_path = self.drug_db_path.parent / self.CACHE_FILE
        if self.use_cache and cache_path.exists():
            log.info("Loading fragment library from cache: %s", cache_path)
            with open(cache_path, "rb") as f:
                self._fragments = pickle.load(f)
            log.info("Loaded %d fragments from cache", len(self._fragments))
            return

        drugs = self._load_drugs()
        log.info("Fragmenting %d approved drugs...", len(drugs))

        for drug_name, drug_smiles, drug_mol in drugs:
            self._fragment_brics(drug_mol, drug_name, drug_smiles)
            self._fragment_murcko(drug_mol, drug_name, drug_smiles)
            self._fragment_rings(drug_mol, drug_name, drug_smiles)

        log.info("Built fragment library with %d unique fragments", len(self._fragments))

        if self.use_cache:
            with open(cache_path, "wb") as f:
                pickle.dump(self._fragments, f)
            log.info("Fragment library cached at %s", cache_path)

    def _load_drugs(self):
        drugs = []
        with open(self.drug_db_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                smiles = parts[0]
                name = parts[1] if len(parts) > 1 else smiles[:20]
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    drugs.append((name, smiles, mol))
                else:
                    log.warning("Invalid drug SMILES skipped: %s", smiles)
        return drugs

    def _store(self, frag_mol: Chem.Mol, source_drug: str, source_smiles: str, frag_type: str):
        if frag_mol is None:
            return
        try:
            Chem.SanitizeMol(frag_mol)
        except Exception:
            return

        smiles = Chem.MolToSmiles(frag_mol)
        if smiles in self._fragments:
            return  # deduplicate

        # Parse attachment points (dummy atoms, atomic num == 0)
        specs = []
        for atom in frag_mol.GetAtoms():
            if atom.GetAtomicNum() != 0:
                continue
            # Find the neighbor (the real atom the dummy is bonded to)
            neighbors = list(atom.GetNeighbors())
            if not neighbors:
                continue
            nb = neighbors[0]
            bond = frag_mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
            specs.append((nb.GetSymbol(), nb.GetHybridization(), bond.GetBondType()))

        n = len(specs)
        if not (self.min_attachments <= n <= self.max_attachments):
            return
        if not self._is_drug_like_fragment(frag_mol):
            return

        self._fragments[smiles] = Fragment(
            smiles=smiles,
            n_attachments=n,
            attachment_specs=specs,
            source_drug=source_drug,
            source_drug_smiles=source_smiles,
            frag_type=frag_type,
        )

    def _is_drug_like_fragment(self, mol: Chem.Mol) -> bool:
        """Quick sanity checks — rejects trivial or huge fragments."""
        heavy = mol.GetNumHeavyAtoms()
        if heavy < 3 or heavy > 35:
            return False
        # Must have at least one ring or be meaningful linker
        if mol.GetRingInfo().NumRings() == 0 and heavy < 5:
            return False
        return True

    # ------------------------------------------------------------------
    # Fragmentation strategies
    # ------------------------------------------------------------------

    def _fragment_brics(self, mol: Chem.Mol, name: str, smiles: str):
        try:
            frags = BRICS.BRICSDecompose(mol, returnMols=True, keepNonLeafNodes=True)
        except Exception as e:
            log.debug("BRICS failed for %s: %s", name, e)
            return
        for frag in frags:
            self._store(frag, name, smiles, "brics")

    def _fragment_murcko(self, mol: Chem.Mol, name: str, smiles: str):
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            self._store(core, name, smiles, "murcko")
            # Also generic scaffold (all bonds/atoms normalized)
            generic = MurckoScaffold.MakeScaffoldGeneric(core)
            self._store(generic, name, smiles, "murcko_generic")
        except Exception as e:
            log.debug("Murcko failed for %s: %s", name, e)

    def _fragment_rings(self, mol: Chem.Mol, name: str, smiles: str):
        """Extract individual ring systems with their immediate substituent attachment atoms."""
        ring_info = mol.GetRingInfo()
        if not ring_info.NumRings():
            return

        ring_atom_sets = [set(r) for r in ring_info.AtomRings()]
        # Merge fused rings
        merged = self._merge_fused_rings(ring_atom_sets)

        for ring_atoms in merged:
            self._extract_ring_fragment(mol, ring_atoms, name, smiles)

    def _merge_fused_rings(self, ring_sets):
        merged = list(ring_sets)
        changed = True
        while changed:
            changed = False
            new_merged = []
            used = set()
            for i, r1 in enumerate(merged):
                if i in used:
                    continue
                combined = set(r1)
                for j, r2 in enumerate(merged):
                    if j <= i or j in used:
                        continue
                    if combined & r2:  # shared atoms -> fused
                        combined |= r2
                        used.add(j)
                        changed = True
                used.add(i)
                new_merged.append(combined)
            merged = new_merged
        return merged

    def _extract_ring_fragment(self, mol: Chem.Mol, ring_atoms: set, name: str, smiles: str):
        from rdkit.Chem import RWMol
        old_to_new = {}
        rw = RWMol()

        for old_idx in sorted(ring_atoms):
            atom = mol.GetAtomWithIdx(old_idx)
            new_atom = Chem.Atom(atom.GetAtomicNum())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_atom.SetIsAromatic(atom.GetIsAromatic())
            old_to_new[old_idx] = rw.AddAtom(new_atom)

        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in ring_atoms and a2 in ring_atoms:
                rw.AddBond(old_to_new[a1], old_to_new[a2], bond.GetBondType())

        # Add dummy atoms for each exo bond (ring -> outside)
        dummy_rank = 1
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            ring_side = scaffold_side = None
            if a1 in ring_atoms and a2 not in ring_atoms:
                ring_side, scaffold_side = a1, a2
            elif a2 in ring_atoms and a1 not in ring_atoms:
                ring_side, scaffold_side = a2, a1
            else:
                continue

            dummy = rw.AddAtom(Chem.Atom(0))
            rw.GetAtomWithIdx(dummy).SetAtomMapNum(dummy_rank)
            dummy_rank += 1
            rw.AddBond(old_to_new[ring_side], dummy, bond.GetBondType())

        try:
            Chem.SanitizeMol(rw)
        except Exception:
            pass

        self._store(rw.GetMol(), name, smiles, "ring")

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_fragments_with_n_attachments(self, n: int) -> List[Fragment]:
        return [f for f in self._fragments.values() if f.n_attachments == n]

    def get_compatible_fragments(
        self,
        attachment_specs: List[Tuple[str, Chem.HybridizationType, Chem.BondType]],
        strict_hybridization: bool = True,
    ) -> List[Fragment]:
        """
        Return fragments whose attachment points are compatible with the
        required attachment specs. Order of specs matters for enumeration.
        """
        n = len(attachment_specs)
        candidates = []
        for frag in self._fragments.values():
            if frag.n_attachments != n:
                continue
            if self._specs_compatible(frag.attachment_specs, attachment_specs, strict_hybridization):
                candidates.append(frag)
        return candidates

    def _specs_compatible(
        self,
        frag_specs,
        required_specs,
        strict_hybridization: bool,
    ) -> bool:
        """
        Check all permutations of frag_specs against required_specs.
        Bond type must match exactly. Hybridization matched if strict.
        """
        from itertools import permutations
        for perm in permutations(frag_specs):
            if all(
                self._single_spec_compatible(fs, rs, strict_hybridization)
                for fs, rs in zip(perm, required_specs)
            ):
                return True
        return False

    def _single_spec_compatible(self, frag_spec, required_spec, strict_hyb: bool) -> bool:
        f_sym, f_hyb, f_bond = frag_spec
        r_sym, r_hyb, r_bond = required_spec
        if f_bond != r_bond:
            return False
        if strict_hyb and f_hyb != r_hyb:
            return False
        return True
