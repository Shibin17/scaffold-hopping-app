"""
molecule_splitter.py — Split a molecule into fixed region and replaceable scaffold.

The user provides a SMARTS pattern identifying the scaffold. Everything outside
that match is the "fixed region". Attachment points are the bonds that cross the
boundary between the two regions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Set

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors


@dataclass
class AttachmentPoint:
    """One bond crossing the fixed/scaffold boundary."""
    fixed_atom_idx: int          # atom index in the ORIGINAL molecule (fixed side)
    scaffold_atom_idx: int       # atom index in the ORIGINAL molecule (scaffold side)
    bond_type: Chem.BondType
    fixed_atom_symbol: str
    fixed_atom_hybridization: Chem.HybridizationType
    scaffold_atom_symbol: str
    scaffold_atom_hybridization: Chem.HybridizationType


@dataclass
class MoleculeRegions:
    original_mol: Chem.Mol
    scaffold_smarts: str
    scaffold_atom_indices: Set[int]
    fixed_atom_indices: Set[int]
    attachment_points: List[AttachmentPoint]
    # SMILES of fixed region with dummy atoms ([*]) at attachment points
    fixed_smiles_with_dummies: str = ""
    # Mapping: dummy atom rank -> AttachmentPoint
    dummy_to_attachment: dict = field(default_factory=dict)


class MoleculeSplitter:
    def __init__(self, mol: Chem.Mol, scaffold_smarts: str):
        self.mol = mol
        self.scaffold_smarts = scaffold_smarts

    def split(self) -> MoleculeRegions:
        pattern = Chem.MolFromSmarts(self.scaffold_smarts)
        if pattern is None:
            raise ValueError(f"Invalid scaffold SMARTS: {self.scaffold_smarts}")

        matches = self.mol.GetSubstructMatches(pattern)
        if not matches:
            raise ValueError(
                f"Scaffold SMARTS '{self.scaffold_smarts}' did not match the reference molecule.\n"
                f"Reference SMILES: {Chem.MolToSmiles(self.mol)}"
            )

        # Use the first match; user can refine SMARTS for specificity
        scaffold_indices: Set[int] = set(matches[0])
        fixed_indices: Set[int] = set(range(self.mol.GetNumAtoms())) - scaffold_indices

        attachment_points = self._find_attachment_points(scaffold_indices, fixed_indices)

        if not attachment_points:
            raise ValueError(
                "No attachment points found between scaffold and fixed regions. "
                "Ensure the scaffold is not the entire molecule."
            )

        fixed_smiles, dummy_map = self._build_fixed_fragment(fixed_indices, attachment_points)

        regions = MoleculeRegions(
            original_mol=self.mol,
            scaffold_smarts=self.scaffold_smarts,
            scaffold_atom_indices=scaffold_indices,
            fixed_atom_indices=fixed_indices,
            attachment_points=attachment_points,
            fixed_smiles_with_dummies=fixed_smiles,
            dummy_to_attachment=dummy_map,
        )
        return regions

    def _find_attachment_points(
        self,
        scaffold_indices: Set[int],
        fixed_indices: Set[int],
    ) -> List[AttachmentPoint]:
        points = []
        for bond in self.mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (a1 in fixed_indices and a2 in scaffold_indices):
                fixed_idx, scaffold_idx = a1, a2
            elif (a2 in fixed_indices and a1 in scaffold_indices):
                fixed_idx, scaffold_idx = a2, a1
            else:
                continue

            fixed_atom = self.mol.GetAtomWithIdx(fixed_idx)
            scaffold_atom = self.mol.GetAtomWithIdx(scaffold_idx)
            points.append(AttachmentPoint(
                fixed_atom_idx=fixed_idx,
                scaffold_atom_idx=scaffold_idx,
                bond_type=bond.GetBondType(),
                fixed_atom_symbol=fixed_atom.GetSymbol(),
                fixed_atom_hybridization=fixed_atom.GetHybridization(),
                scaffold_atom_symbol=scaffold_atom.GetSymbol(),
                scaffold_atom_hybridization=scaffold_atom.GetHybridization(),
            ))
        return points

    def _build_fixed_fragment(
        self,
        fixed_indices: Set[int],
        attachment_points: List[AttachmentPoint],
    ) -> Tuple[str, dict]:
        """
        Build an RWMol of the fixed region, replacing scaffold-side atoms at
        attachment points with dummy atoms ([*:n]). Returns SMILES and mapping.
        """
        from rdkit.Chem import RWMol, AllChem

        # Map old idx -> new idx in fixed fragment
        sorted_fixed = sorted(fixed_indices)
        old_to_new = {old: new for new, old in enumerate(sorted_fixed)}

        rw = RWMol()
        for old_idx in sorted_fixed:
            atom = self.mol.GetAtomWithIdx(old_idx)
            new_atom = Chem.Atom(atom.GetAtomicNum())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
            new_atom.SetIsAromatic(atom.GetIsAromatic())
            rw.AddAtom(new_atom)

        # Add bonds within fixed region
        for bond in self.mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in fixed_indices and a2 in fixed_indices:
                rw.AddBond(old_to_new[a1], old_to_new[a2], bond.GetBondType())

        # Replace attachment-point atoms that are on the FIXED side with dummy atoms
        # (The scaffold side will later be replaced by a new fragment)
        dummy_map = {}
        for rank, ap in enumerate(attachment_points, start=1):
            # Add a dummy atom connected to the fixed-side atom
            dummy_idx = rw.AddAtom(Chem.Atom(0))  # atomic num 0 = wildcard
            rw.GetAtomWithIdx(dummy_idx).SetAtomMapNum(rank)
            fixed_new_idx = old_to_new[ap.fixed_atom_idx]
            rw.AddBond(fixed_new_idx, dummy_idx, ap.bond_type)
            dummy_map[rank] = ap

        try:
            Chem.SanitizeMol(rw)
        except Exception:
            pass  # partial sanitization ok for fragment

        smiles = Chem.MolToSmiles(rw)
        return smiles, dummy_map
