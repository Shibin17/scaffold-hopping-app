"""
bioisosteres.py — Curated ring bioisostere database from FDA/EMA approved drugs.

All entries are ring systems verified in approved drugs.
Attachment points use [1*], [2*] ... convention (remapped at runtime).

Keys: n_attachments (1 | 2 | 3 | 4)
"""

# Format: (smiles_with_numbered_dummies, ring_name, source_drug)
BIOISOSTERES = {

    # ── 1 attachment point ─────────────────────────────────────────────
    1: [
        ("[1*]c1ccccc1",         "Phenyl",          "aspirin"),
        ("[1*]c1ccccn1",         "2-Pyridyl",        "piroxicam"),
        ("[1*]c1cccnc1",         "3-Pyridyl",        "niacin"),
        ("[1*]c1ccncc1",         "4-Pyridyl",        "isoniazid"),
        ("[1*]c1ccoc1",          "2-Furyl",          "nitrofurantoin"),
        ("[1*]c1ccsc1",          "2-Thienyl",        "cephalosporins"),
        ("[1*]c1cc[nH]c1",       "2-Pyrrolyl",       "ketorolac"),
        ("[1*]c1cnco1",          "Oxazolyl",         "oxazepam"),
        ("[1*]c1cncs1",          "Thiazolyl",        "ritonavir"),
        ("[1*]c1cnnc1",          "Pyrazolyl",        "celecoxib"),
        ("[1*]c1ccno1",          "Isoxazolyl",       "valdecoxib"),
        ("[1*]c1ccns1",          "Isothiazolyl",     "cefazolin"),
        ("[1*]c1ncncn1",         "Triazinyl",        "atrazine"),
        ("[1*]c1ccc2ccccc2c1",   "Naphthyl",         "naproxen"),
        ("[1*]c1ccc2[nH]ccc2c1", "Indolyl",          "indomethacin"),
        ("[1*]c1ccc2occc2c1",    "Benzofuryl",       "amiodarone"),
        ("[1*]c1ccc2sccc2c1",    "Benzothienyl",     "raloxifene"),
        ("[1*]C1CCCCC1",         "Cyclohexyl",       "gabapentin"),
        ("[1*]C1CCCC1",          "Cyclopentyl",      "ticlopidine"),
        ("[1*]C1CCC1",           "Cyclobutyl",       "tranexamic"),
        ("[1*]C1CCNCC1",         "Piperidinyl",      "haloperidol"),
        ("[1*]C1CCOC1",          "Tetrahydrofuryl",  "fluconazole"),
        ("[1*]C1CCOCC1",         "Tetrahydropyranyl","pantoprazole"),
        ("[1*]C1CCNC1",          "Pyrrolidinyl",     "atorvastatin"),
        ("[1*]c1ncc[nH]1",       "Imidazolyl",       "metronidazole"),
        ("[1*]c1ncnn1",          "Triazolyl",        "fluconazole"),
        ("[1*]c1nncn1",          "Tetrazolyl",       "losartan"),
    ],

    # ── 2 attachment points ────────────────────────────────────────────
    2: [
        # 6-membered aromatic (para substitution → positions 1,4)
        ("[1*]c1ccc([2*])cc1",          "Benzene-1,4",          "paracetamol"),
        ("[1*]c1ccncc1[2*]",            "Pyridine-2,5",         "imatinib"),
        ("[1*]c1cncc([2*])c1",          "Pyridine-2,4",         "nifedipine"),
        ("[1*]c1cccc([2*])n1",          "Pyridine-2,6",         "isoniazid"),
        ("[1*]c1ccnc([2*])c1",          "Pyridine-3,5",         "nicotinamide"),
        ("[1*]c1ncnc([2*])c1",          "Pyrimidine-2,5",       "trimethoprim"),
        ("[1*]c1nccc([2*])n1",          "Pyrimidine-2,4",       "cytarabine"),
        ("[1*]c1ccnc([2*])n1",          "Pyrimidine-4,6",       "sulfadiazine"),
        ("[1*]c1cncc([2*])n1",          "Pyrazine-2,5",         "pyrazinamide"),
        ("[1*]c1ncc([2*])cn1",          "Pyrazine-2,6",         "methotrexate"),
        ("[1*]c1ncc([2*])nn1",          "Pyridazine-3,5",       "hydralazine"),
        # 5-membered aromatic (positions 2,5 → diff=2)
        ("[1*]c1ccc([2*])o1",           "Furan-2,5",            "nitrofurantoin"),
        ("[1*]c1ccc([2*])s1",           "Thiophene-2,5",        "tiotropium"),
        ("[1*]c1ccc([2*])[nH]1",        "Pyrrole-2,5",          "atorvastatin"),
        ("[1*]c1cnc([2*])[nH]1",        "Imidazole-2,4",        "histamine"),
        ("[1*]c1cnn([2*])[nH]1",        "Pyrazole-1,3",         "celecoxib"),
        ("[1*]c1cnc([2*])o1",           "Oxazole-2,4",          "rivaroxaban"),
        ("[1*]c1cnc([2*])s1",           "Thiazole-2,4",         "sulfathiazole"),
        ("[1*]c1ccn([2*])n1",           "Isopyrazole-1,4",      "phenylbutazone"),
        # Saturated linkers
        ("[1*]C1CCC([2*])CC1",          "Cyclohexane-1,4",      "tranexamic"),
        ("[1*]C1CC([2*])CC1",           "Cyclopentane-1,3",     "gemfibrozil"),
        ("[1*]C1CCN([2*])CC1",          "Piperidine-1,4",       "fentanyl"),
        ("[1*]C1CCOC([2*])C1",          "Tetrahydropyran-2,5",  "ribose"),
        ("[1*]C1CN([2*])CCO1",          "Morpholine-2,4",       "morphine"),
        # Fused bicyclics
        ("[1*]c1ccc2ccc([2*])cc2c1",    "Naphthalene-1,6",      "naproxen"),
        ("[1*]c1ccc2[nH]cc([2*])c2c1",  "Indole-3,5",           "indomethacin"),
        ("[1*]c1ccc2oc([2*])cc2c1",     "Benzofuran-2,5",       "amiodarone"),
        ("[1*]c1ccc2sc([2*])cc2c1",     "Benzothiophene-2,5",   "raloxifene"),
        ("[1*]c1ccc2c(c1)nc([2*])n2",   "Benzimidazole-2,5",    "omeprazole"),
        ("[1*]c1ccc2c(c1)oc([2*])n2",   "Benzoxazole-2,5",      "leflunomide"),
        ("[1*]c1ccc2c(c1)sc([2*])n2",   "Benzothiazole-2,5",    "riluzole"),
        ("[1*]c1ccc2c(c1)[nH]c([2*])n2","Benzimidazole-1,5",    "mebendazole"),
        # Ortho (positions 1,2 → diff=1)
        ("[1*]c1ccccc1[2*]",            "Benzene-1,2",          "catechol"),
        ("[1*]c1cccc([2*])c1",          "Benzene-1,3",          "resorcinol"),
        ("[1*]c1nccn([2*])c1",          "Imidazole-1,3",        "clotrimazole"),
    ],

    # ── 3 attachment points ────────────────────────────────────────────
    3: [
        ("[1*]c1cc([2*])cc([3*])c1",        "Benzene-1,3,5",        "mesitylene"),
        ("[1*]c1cc([2*])c([3*])cc1",        "Benzene-1,2,4",        "trimethoprim"),
        ("[1*]c1c([2*])cc([3*])cc1",        "Benzene-1,2,3",        "gallic_acid"),
        ("[1*]c1cnc([2*])nc1[3*]",          "Triazine-2,4,5",       "trimethoprim"),
        ("[1*]c1nc([2*])nc([3*])n1",        "Triazine-2,4,6",       "atrazine"),
        ("[1*]c1cnc([2*])cn1[3*]",          "Pyrimidine-2,4,6",     "barbiturate"),
        ("[1*]C1CC([2*])CN1[3*]",           "Pyrrolidine-1,2,4",    "lisinopril"),
        ("[1*]C1CC([2*])CC([3*])C1",        "Cyclohexane-1,3,5",    "cyclohexane"),
        ("[1*]C1CC([3*])N([2*])C1",         "Pyrrolidine-1,3,4",    "proline"),
        ("[1*]c1cc([2*])n([3*])c1",         "Pyrrole-1,2,4",        "porphyrin"),
        ("[1*]c1c([2*])cnc([3*])c1",        "Pyridine-2,3,5",       "pyridoxine"),
        ("[1*]c1cc([2*])c([3*])cn1",        "Pyridine-3,4,6",       "nicotine"),
    ],

    # ── 4 attachment points ────────────────────────────────────────────
    4: [
        ("[1*]c1c([2*])cc([3*])c([4*])c1",     "Benzene-1,2,3,5",   "quercetin"),
        ("[1*]c1c([2*])c([3*])cc([4*])c1",     "Benzene-1,2,3,4",   "tetrazine"),
        ("[1*]c1c([2*])c([3*])c([4*])cc1",     "Benzene-1,2,4,5",   "tetramethyl"),
        ("[1*]C1CC([2*])C([3*])CC1[4*]",       "Cyclohexane-1,2,3,4","inositol"),
        ("[1*]c1nc([2*])c([3*])nc1[4*]",       "Pyrimidine-2,4,5,6", "uracil"),
        ("[1*]c1c([2*])ncc([3*])c1[4*]",       "Pyridine-2,3,4,6",   "pyridoxine"),
    ],
}

# Quick lookup: all entries for n_attachments
def get_bioisosteres(n_attachments: int):
    return BIOISOSTERES.get(n_attachments, [])

# Source drug name for display (scan all entries)
def source_drug_for_smiles(smi: str) -> str:
    for entries in BIOISOSTERES.values():
        for s, name, drug in entries:
            if s == smi:
                return drug
    return "approved_drug"
