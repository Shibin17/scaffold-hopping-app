"""
clinical_drugs.py — Download and cache FDA/EMA Phase-4 approved drugs from ChEMBL.

Fetches only max_phase=4 molecules (approved drugs) with valid SMILES.
Caches locally so subsequent runs are instant.
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

import requests

log = logging.getLogger(__name__)

CHEMBL_API   = "https://www.ebi.ac.uk/chembl/api/data/molecule"
CACHE_FILE   = "clinical_drugs.smi"
CACHE_JSON   = "clinical_drugs_meta.json"
PAGE_SIZE    = 500
MAX_DRUGS    = 3000      # cap to avoid enormous runs; covers virtually all approved drugs


def fetch_clinical_drugs(
    cache_path: str = CACHE_FILE,
    force_refresh: bool = False,
    progress_cb=None,
) -> str:
    """
    Returns path to a .smi file with one 'SMILES name' per line.
    Downloads from ChEMBL if not cached.
    """
    p = Path(cache_path)
    if p.exists() and not force_refresh:
        n = sum(1 for l in p.read_text().splitlines() if l.strip())
        log.info("Clinical drugs cache hit: %s (%d entries)", cache_path, n)
        return str(p)

    log.info("Downloading approved drugs from ChEMBL (max_phase=4)…")
    drugs = _download_all(progress_cb)
    log.info("Downloaded %d approved drugs", len(drugs))

    lines = [f"{smi} {name}" for smi, name in drugs]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {"count": len(drugs), "source": "ChEMBL max_phase=4"}
    Path(CACHE_JSON).write_text(json.dumps(meta, indent=2))
    log.info("Saved to %s", cache_path)
    return str(p)


def _download_all(progress_cb=None) -> List[Tuple[str, str]]:
    drugs = []
    offset = 0
    total  = None

    params_base = {
        "max_phase": 4,
        "molecule_type": "Small molecule",
        "format": "json",
        "limit": PAGE_SIZE,
    }

    while True:
        params = {**params_base, "offset": offset}
        try:
            resp = requests.get(CHEMBL_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error("ChEMBL request failed at offset %d: %s", offset, e)
            break

        if total is None:
            total = data["page_meta"]["total_count"]
            log.info("ChEMBL reports %d approved small-molecule drugs", total)

        for mol in data["molecules"]:
            smi = None
            try:
                structs = mol.get("molecule_structures") or {}
                smi = structs.get("canonical_smiles") or structs.get("molfile")
                if not smi:
                    continue
                name = (
                    mol.get("pref_name")
                    or mol.get("chembl_id")
                    or f"CHEMBL{mol.get('chembl_id','UNK')}"
                )
                name = name.replace(" ", "_")
                drugs.append((smi, name))
            except Exception:
                continue

        offset += PAGE_SIZE
        fetched = min(offset, total or offset)

        if progress_cb:
            progress_cb(fetched, total or fetched)

        if offset >= (total or 0) or offset >= MAX_DRUGS:
            break

        time.sleep(0.05)  # be polite to ChEMBL

    return drugs


def drug_db_stats(cache_path: str = CACHE_FILE) -> dict:
    p = Path(cache_path)
    if not p.exists():
        return {"loaded": False}
    lines = [l for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return {"loaded": True, "count": len(lines), "path": str(p)}
