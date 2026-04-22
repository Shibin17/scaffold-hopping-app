"""
rl_generator.py — REINVENT-style RL scaffold hopper.

Architecture:
  - Prior policy: fragment-based SMILES RNN (or simple random sampler over approved fragments)
  - Agent policy: fine-tuned with augmented likelihood objective
  - Reward: MoleculeScorer.score_one()

For full REINVENT integration, replace SimplePrior with a pretrained RNN.
This module provides a self-contained fragment-action RL loop that only
samples from the approved fragment library.
"""

from __future__ import annotations
import logging
import random
from dataclasses import dataclass
from typing import List, Optional

from rdkit import Chem

from fragment_library import FragmentLibrary, Fragment
from molecule_splitter import MoleculeRegions
from scaffold_replacer import ScaffoldReplacer, Candidate
from scorer import MoleculeScorer, ScoredCandidate

log = logging.getLogger(__name__)


@dataclass
class RLConfig:
    n_steps: int = 500
    batch_size: int = 32
    top_fraction: float = 0.3       # keep top X% as "elite" for next round
    temperature: float = 1.0        # softmax temperature for fragment sampling
    diversity_penalty: float = 0.1  # penalize duplicate scaffolds in batch
    sigma: float = 120.0            # REINVENT augmented likelihood sigma


class FragmentActionSpace:
    """
    Discrete action space: choose one fragment from the compatible library.
    Actions are weighted by their historical reward to implement a simple
    cross-entropy / evolution-strategy policy.
    """

    def __init__(self, fragments: List[Fragment], temperature: float = 1.0):
        self.fragments = fragments
        self.weights = [1.0] * len(fragments)
        self.temperature = temperature

    def sample(self, k: int = 1) -> List[Fragment]:
        import math
        total = sum(math.exp(w / self.temperature) for w in self.weights)
        probs = [math.exp(w / self.temperature) / total for w in self.weights]
        indices = random.choices(range(len(self.fragments)), weights=probs, k=k)
        return [self.fragments[i] for i in indices]

    def update_weights(self, fragment: Fragment, reward: float, lr: float = 0.1):
        idx = self.fragments.index(fragment)
        self.weights[idx] = (1 - lr) * self.weights[idx] + lr * reward


class RLScaffoldHopper:
    def __init__(
        self,
        regions: MoleculeRegions,
        library: FragmentLibrary,
        scorer: MoleculeScorer,
        config: Optional[RLConfig] = None,
    ):
        self.regions = regions
        self.library = library
        self.scorer = scorer
        self.config = config or RLConfig()

        n_attach = len(regions.attachment_points)
        required_specs = self._required_specs()
        compatible = library.get_compatible_fragments(required_specs, strict_hybridization=False)
        log.info("RL action space: %d compatible fragments", len(compatible))
        self.action_space = FragmentActionSpace(compatible, config.temperature if config else 1.0)

    def run(self) -> List[ScoredCandidate]:
        cfg = self.config
        all_scored: List[ScoredCandidate] = []
        seen_smiles = set()

        for step in range(cfg.n_steps):
            # Sample a batch of fragments
            frags = self.action_space.sample(k=cfg.batch_size)
            batch_candidates = []

            for frag in frags:
                replacer = ScaffoldReplacer(self.regions, self._single_frag_library(frag))
                mols = replacer.enumerate()
                for cand in mols:
                    if cand.smiles not in seen_smiles:
                        seen_smiles.add(cand.smiles)
                        batch_candidates.append((frag, cand))

            if not batch_candidates:
                continue

            # Score batch
            scored_batch = []
            for frag, cand in batch_candidates:
                sc = self.scorer._score_one(cand)
                scored_batch.append((frag, sc))

            # Apply diversity penalty
            scaffold_counts = {}
            for frag, sc in scored_batch:
                key = frag.smiles
                scaffold_counts[key] = scaffold_counts.get(key, 0) + 1
            for frag, sc in scored_batch:
                penalty = (scaffold_counts[frag.smiles] - 1) * cfg.diversity_penalty
                sc.total_score = max(0.0, sc.total_score - penalty)

            # Update action space weights (policy update)
            for frag, sc in scored_batch:
                self.action_space.update_weights(frag, sc.total_score)

            # Keep elite
            scored_batch.sort(key=lambda x: x[1].total_score, reverse=True)
            n_keep = max(1, int(len(scored_batch) * cfg.top_fraction))
            all_scored.extend([sc for _, sc in scored_batch[:n_keep]])

            if (step + 1) % 50 == 0:
                best = max(all_scored, key=lambda x: x.total_score) if all_scored else None
                log.info(
                    "Step %d/%d | Explored: %d | Best score: %.4f",
                    step + 1, cfg.n_steps,
                    len(seen_smiles),
                    best.total_score if best else 0.0,
                )

        # Final dedup and sort
        seen = set()
        unique = []
        for sc in sorted(all_scored, key=lambda x: x.total_score, reverse=True):
            if sc.smiles not in seen:
                seen.add(sc.smiles)
                unique.append(sc)

        log.info("RL run complete. Unique candidates: %d", len(unique))
        return unique

    def _required_specs(self):
        specs = []
        for ap in self.regions.attachment_points:
            specs.append((ap.scaffold_atom_symbol, ap.scaffold_atom_hybridization, ap.bond_type))
        return specs

    def _single_frag_library(self, frag: Fragment) -> FragmentLibrary:
        """Wrap a single fragment in a minimal FragmentLibrary-compatible object."""
        class SingleFragLib:
            def get_compatible_fragments(self, specs, strict_hybridization=True):
                return [frag]

            def __len__(self):
                return 1

            def __iter__(self):
                return iter([frag])

        return SingleFragLib()
