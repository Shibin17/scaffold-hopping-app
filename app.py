"""
app.py — Scaffold Hopper Streamlit UI
  Tab 1: Sketcher    — draw molecule, mark scaffold atoms visually
  Tab 2: Database    — download/manage clinical drugs from ChEMBL
  Tab 3: Run         — execute pipeline with live progress
  Tab 4: Results     — scored card grid + charts + downloads
  Tab 5: RL          — REINVENT-style generative search
  Tab 6: About
"""

import io
import json
import logging
import tempfile
import textwrap
import time
from pathlib import Path

import streamlit as st

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Scaffold Hopper",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── rdkit ──────────────────────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors, QED
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold

# ── pipeline ───────────────────────────────────────────────────────────────────
from molecule_splitter import MoleculeSplitter
from fragment_library  import FragmentLibrary
from scaffold_replacer import ScaffoldReplacer
from scorer            import MoleculeScorer
from reporter          import Reporter
from lasso_selector    import lasso_mol_selector, selection_to_smarts
from clinical_drugs    import fetch_clinical_drugs, drug_db_stats, CACHE_FILE


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def mol_to_svg(mol, w=300, h=220, hi_atoms=None, hi_bonds=None):
    if mol is None:
        return ""
    try:
        AllChem.Compute2DCoords(mol)
        d = rdMolDraw2D.MolDraw2DSVG(w, h)
        d.drawOptions().addStereoAnnotation = True
        if hi_atoms:
            acols = {i: (0.96, 0.42, 0.35) for i in hi_atoms}
            bcols = {i: (0.96, 0.42, 0.35) for i in (hi_bonds or [])}
            d.DrawMolecule(mol,
                           highlightAtoms=hi_atoms, highlightAtomColors=acols,
                           highlightBonds=hi_bonds or [], highlightBondColors=bcols)
        else:
            d.DrawMolecule(mol)
        d.FinishDrawing()
        svg = d.GetDrawingText()
        return svg[svg.find("<svg"):]
    except Exception:
        return ""


def svg_wrap(svg, extra_style=""):
    return (f'<div style="background:#fff;border:1px solid #e0e0e0;border-radius:8px;'
            f'padding:4px;display:inline-block;{extra_style}">{svg}</div>')


def props(mol):
    return dict(
        MW   = round(Descriptors.ExactMolWt(mol), 2),
        logP = round(Descriptors.MolLogP(mol),    3),
        TPSA = round(Descriptors.TPSA(mol),       2),
        HBA  = rdMolDescriptors.CalcNumHBA(mol),
        HBD  = rdMolDescriptors.CalcNumHBD(mol),
        RotB = rdMolDescriptors.CalcNumRotatableBonds(mol),
        Rings= mol.GetRingInfo().NumRings(),
        QED  = round(QED.qed(mol), 4),
    )


def score_color(v, lo=0.0, hi=1.0, rev=False):
    f = (v - lo) / (hi - lo + 1e-9)
    if rev: f = 1 - f
    c = "#2ecc71" if f >= .66 else ("#f39c12" if f >= .33 else "#e74c3c")
    return f'<span style="color:{c};font-weight:bold">{v:.3f}</span>'


# ══════════════════════════════════════════════════════════════════════════════
# Session-state defaults
# ══════════════════════════════════════════════════════════════════════════════

for k, v in {
    "ref_mol":        None,
    "scaffold_atoms": set(),
    "scaffold_smarts":"",
    "scored":         [],
    "regions":        None,
    "rl_scored":      [],
    "drug_db_path":   None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Pipeline Settings")

    logp_max = st.slider("Max logP",    0.0, 6.0,  3.0, 0.1)
    mw_max   = st.slider("Max MW (Da)", 200, 800,  600, 10)
    tpsa_max = st.slider("Max TPSA",    60,  200,  140, 5)

    st.markdown("---")
    st.markdown("**Fragment library**")
    min_attach   = st.number_input("Min attachment pts", 1, 4, 1)
    max_attach   = st.number_input("Max attachment pts", 1, 6, 4)
    strict_hyb   = st.checkbox("Strict hybridization", True)

    st.markdown("---")
    st.markdown("**Score weights**")
    w_shape = st.slider("Shape",          0.0, 1.0, 0.25, 0.05)
    w_pharm = st.slider("Pharmacophore",  0.0, 1.0, 0.20, 0.05)
    w_novel = st.slider("Scaffold novelty",0.0,1.0, 0.15, 0.05)
    w_tox   = st.slider("Tox safety",     0.0, 1.0, 0.20, 0.05)
    w_logp  = st.slider("logP score",     0.0, 1.0, 0.10, 0.05)
    w_sa    = st.slider("Synth. access.", 0.0, 1.0, 0.10, 0.05)

    st.markdown("---")
    top_n = st.number_input("Top N candidates", 5, 500, 50)

    st.markdown("---")
    st.markdown("**Optional: Docking**")
    receptor_file = st.file_uploader("Receptor .pdbqt", type=["pdbqt"])
    vina_box_str  = st.text_input("Vina box (cx,cy,cz,sx,sy,sz)", placeholder="10.5,22.1,8.3,20,20,20")

    st.markdown("---")
    db_stats = drug_db_stats(CACHE_FILE)
    if db_stats["loaded"]:
        st.success(f"✔ Clinical DB: {db_stats['count']:,} drugs")
    else:
        st.warning("Clinical DB not downloaded yet (Database tab)")


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    "<h1 style='margin-bottom:0'>🔬 Scaffold Hopper</h1>"
    "<p style='color:#888;margin-top:0'>Clinical-drug-constrained scaffold replacement</p>",
    unsafe_allow_html=True,
)

tab_sketch, tab_db, tab_run, tab_results, tab_rl, tab_about = st.tabs(
    ["✏️ Sketcher", "💊 Clinical DB", "▶ Run", "🏆 Results", "🤖 RL Generator", "ℹ️ About"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SKETCHER
# ══════════════════════════════════════════════════════════════════════════════

with tab_sketch:
    st.subheader("Draw Molecule & Mark Scaffold Region")

    # ── Step 1: molecule input ─────────────────────────────────────────────
    st.markdown("#### Step 1 — Draw or enter your molecule")
    input_mode = st.radio("Input method", ["Ketcher (draw)", "SMILES", "SDF upload"],
                          horizontal=True, key="input_mode")

    ref_mol_local = None

    if input_mode == "Ketcher (draw)":
        try:
            from streamlit_ketcher import st_ketcher
            init_smiles = Chem.MolToSmiles(st.session_state["ref_mol"]) \
                if st.session_state["ref_mol"] else "CC(=O)Nc1ccc(O)cc1"
            smiles_out = st_ketcher(init_smiles, height=400, key="ketcher_main")
            if smiles_out:
                m = Chem.MolFromSmiles(smiles_out)
                if m:
                    ref_mol_local = m
        except Exception as e:
            st.error(f"Ketcher error: {e}")

    elif input_mode == "SMILES":
        smi = st.text_input("SMILES", value="CC(=O)Nc1ccc(O)cc1", key="smi_input")
        if smi:
            m = Chem.MolFromSmiles(smi.strip())
            if m:
                ref_mol_local = m
            else:
                st.error("Invalid SMILES")

    else:
        sdf_up = st.file_uploader("Upload SDF / MOL", type=["sdf","mol"], key="sdf_up")
        if sdf_up:
            content = sdf_up.read().decode(errors="ignore")
            with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False, mode="w") as tmp:
                tmp.write(content); tmp_path = tmp.name
            suppl = Chem.SDMolSupplier(tmp_path, removeHs=False)
            ref_mol_local = next((m for m in suppl if m is not None), None)
            if ref_mol_local is None:
                st.error("Could not parse file")

    if ref_mol_local is not None:
        # Detect molecule change → reset selection
        prev_smi = st.session_state.get("_prev_smi", "")
        new_smi  = Chem.MolToSmiles(ref_mol_local)
        if new_smi != prev_smi:
            st.session_state["scaffold_atoms"]  = set()
            st.session_state["scaffold_smarts"] = ""
            st.session_state["_lasso_sel_lasso"] = set()
            st.session_state["_prev_smi"] = new_smi
        st.session_state["ref_mol"] = ref_mol_local
        p = props(ref_mol_local)
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("MW",   p["MW"])
        mc2.metric("logP", p["logP"])
        mc3.metric("TPSA", p["TPSA"])
        mc4.metric("QED",  p["QED"])
        mc5.metric("Rings",p["Rings"])

    st.markdown("---")

    # ── Step 2: lasso scaffold selection ──────────────────────────────────
    mol = st.session_state["ref_mol"]
    if mol is None:
        st.info("Enter a molecule above to enable scaffold selection.")
    else:
        st.markdown("#### Step 2 — Lasso the scaffold region to replace")
        st.caption(
            "**Draw a lasso** around atoms to select them · "
            "**Click individual atoms** to toggle · "
            "Use toolbar buttons for quick presets · "
            "Click **✔ Confirm** to lock the selection"
        )

        # ── lasso component ────────────────────────────────────────────────
        selected = lasso_mol_selector(
            mol,
            key="lasso",
            width=780,
            height=420,
            current_selection=st.session_state["scaffold_atoms"],
        )
        st.session_state["scaffold_atoms"] = selected

        if selected:
            smarts = selection_to_smarts(mol, selected)
            st.session_state["scaffold_smarts"] = smarts

            st.markdown("---")
            res_left, res_right = st.columns([1, 1], gap="large")

            with res_left:
                st.markdown("##### Scaffold (🔴 red = to replace)")
                hl_atoms = list(selected)
                hl_bonds = [
                    b.GetIdx() for b in mol.GetBonds()
                    if b.GetBeginAtomIdx() in selected and b.GetEndAtomIdx() in selected
                ]
                svg_sc = mol_to_svg(mol, 360, 240, hi_atoms=hl_atoms, hi_bonds=hl_bonds)
                st.markdown(svg_wrap(svg_sc), unsafe_allow_html=True)
                st.success(f"**{len(selected)} atoms** selected")
                st.code(smarts, language="text")

            with res_right:
                st.markdown("##### Fixed region (🔵 blue = preserved)")
                fixed_atoms = set(range(mol.GetNumAtoms())) - selected
                if fixed_atoms:
                    fx_atoms = list(fixed_atoms)
                    fx_bonds = [
                        b.GetIdx() for b in mol.GetBonds()
                        if b.GetBeginAtomIdx() in fixed_atoms and b.GetEndAtomIdx() in fixed_atoms
                    ]
                    # Draw with blue highlight for fixed region
                    from rdkit.Chem.Draw import rdMolDraw2D as _rd
                    AllChem.Compute2DCoords(mol)
                    _d = _rd.MolDraw2DSVG(360, 240)
                    _acols = {i: (0.20, 0.60, 0.86) for i in fx_atoms}
                    _bcols = {i: (0.20, 0.60, 0.86) for i in fx_bonds}
                    _d.DrawMolecule(mol,
                        highlightAtoms=fx_atoms, highlightAtomColors=_acols,
                        highlightBonds=fx_bonds, highlightBondColors=_bcols)
                    _d.FinishDrawing()
                    _svg = _d.GetDrawingText(); _svg = _svg[_svg.find("<svg"):]
                    st.markdown(svg_wrap(_svg), unsafe_allow_html=True)
                    st.info(f"**{len(fixed_atoms)} atoms** will be preserved exactly")

                # SMARTS override
                smarts_override = st.text_input(
                    "Override SMARTS (optional — for precise control)",
                    value=smarts,
                    key="smarts_override",
                )
                if smarts_override and smarts_override != smarts:
                    p2 = Chem.MolFromSmarts(smarts_override)
                    if p2 and mol.GetSubstructMatches(p2):
                        st.session_state["scaffold_smarts"] = smarts_override
                        st.success("Override SMARTS accepted ✔")
                    else:
                        st.error("Invalid SMARTS or no match in molecule.")
        else:
            st.info(
                "Draw a lasso around the scaffold atoms above (or use the toolbar buttons), "
                "then click **✔ Confirm**."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLINICAL DB
# ══════════════════════════════════════════════════════════════════════════════

with tab_db:
    st.subheader("💊 Clinical Drugs Database (ChEMBL FDA/EMA Approved)")
    st.markdown(
        "Downloads **Phase-4 approved small-molecule drugs** from ChEMBL. "
        "Only these structures are used as fragment sources — zero invented scaffolds."
    )

    db_stats_now = drug_db_stats(CACHE_FILE)

    dcol1, dcol2 = st.columns([1, 1], gap="large")

    with dcol1:
        if db_stats_now["loaded"]:
            st.success(f"✔ Local cache: **{db_stats_now['count']:,} approved drugs**")
            st.caption(f"File: `{db_stats_now['path']}`")
            st.session_state["drug_db_path"] = db_stats_now["path"]

            if Path("clinical_drugs_meta.json").exists():
                meta = json.loads(Path("clinical_drugs_meta.json").read_text())
                st.json(meta)
        else:
            st.warning("No clinical drug database found locally.")

        dl_btn      = st.button("⬇ Download from ChEMBL", type="primary", use_container_width=True)
        force_refresh = st.checkbox("Force re-download (ignore cache)")

        if dl_btn:
            prog  = st.progress(0, text="Connecting to ChEMBL…")
            info  = st.empty()

            def update_progress(fetched, total):
                pct = min(int(fetched / max(total, 1) * 100), 99)
                prog.progress(pct, text=f"Downloading… {fetched:,}/{total:,} drugs")
                info.info(f"Fetched {fetched:,} drugs so far")

            try:
                path = fetch_clinical_drugs(
                    cache_path=CACHE_FILE,
                    force_refresh=force_refresh,
                    progress_cb=update_progress,
                )
                prog.progress(100, text="Complete!")
                info.success(f"Downloaded and saved to `{path}`")
                st.session_state["drug_db_path"] = path
                st.rerun()
            except Exception as e:
                prog.empty()
                info.error(f"Download failed: {e}")

        st.markdown("---")
        st.markdown("**Or upload your own .smi file**")
        user_db = st.file_uploader("Custom drug SMILES file (SMILES [name] per line)",
                                   type=["smi","txt","csv"], key="custom_db")
        if user_db:
            tmp = Path(tempfile.mktemp(suffix=".smi"))
            tmp.write_bytes(user_db.read())
            n = sum(1 for l in tmp.read_text().splitlines() if l.strip() and not l.startswith("#"))
            st.success(f"Loaded {n:,} entries from uploaded file.")
            st.session_state["drug_db_path"] = str(tmp)

    with dcol2:
        st.markdown("#### Preview: random sample from DB")
        db_path = st.session_state.get("drug_db_path")
        if db_path and Path(db_path).exists():
            import random
            lines = [l.strip() for l in Path(db_path).read_text().splitlines()
                     if l.strip() and not l.startswith("#")]
            sample = random.sample(lines, min(12, len(lines)))

            mols_and_names = []
            for line in sample:
                parts = line.split()
                smi  = parts[0]
                name = parts[1] if len(parts) > 1 else smi[:12]
                m    = Chem.MolFromSmiles(smi)
                if m:
                    mols_and_names.append((m, name))

            cols_per_row = 3
            for row in range(0, len(mols_and_names), cols_per_row):
                row_data = mols_and_names[row:row+cols_per_row]
                rcols    = st.columns(cols_per_row)
                for col, (m, name) in zip(rcols, row_data):
                    svg = mol_to_svg(m, 160, 120)
                    col.markdown(
                        f'<div style="text-align:center;border:1px solid #eee;'
                        f'border-radius:6px;padding:4px;background:#fff">'
                        f'{svg}<div style="font-size:10px;color:#555">{name[:20]}</div></div>',
                        unsafe_allow_html=True,
                    )

            if st.button("🔄 Refresh sample"):
                st.rerun()
        else:
            st.info("Download or upload a drug database to preview.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RUN
# ══════════════════════════════════════════════════════════════════════════════

with tab_run:
    st.subheader("▶ Run Scaffold Hopping Pipeline")

    # Pre-flight checklist
    ref_ok     = st.session_state["ref_mol"] is not None
    scaf_ok    = bool(st.session_state["scaffold_smarts"])
    db_ok      = bool(st.session_state.get("drug_db_path")) and \
                 Path(st.session_state.get("drug_db_path", "")).exists()

    rc1, rc2, rc3 = st.columns(3)
    rc1.markdown(
        f"{'✅' if ref_ok  else '❌'} Reference molecule "
        f"({'set' if ref_ok else 'go to Sketcher tab'})"
    )
    rc2.markdown(
        f"{'✅' if scaf_ok else '❌'} Scaffold marked "
        f"({'SMARTS ready' if scaf_ok else 'select atoms in Sketcher'})"
    )
    rc3.markdown(
        f"{'✅' if db_ok   else '❌'} Clinical DB "
        f"({'loaded' if db_ok else 'download in Clinical DB tab'})"
    )

    if ref_ok and scaf_ok:
        sma = st.session_state["scaffold_smarts"]
        mol = st.session_state["ref_mol"]
        patt = Chem.MolFromSmarts(sma)
        matches = mol.GetSubstructMatches(patt) if patt else []
        if not matches:
            st.warning("Current SMARTS doesn't match the molecule — revisit Sketcher.")
            scaf_ok = False
        else:
            st.info(f"Scaffold SMARTS: `{sma}` — {len(matches[0])} atoms matched")

    run_ready = ref_ok and scaf_ok and db_ok
    run_btn   = st.button(
        "🚀 Run Scaffold Hopping",
        type="primary",
        disabled=not run_ready,
        use_container_width=True,
    )

    if not run_ready:
        missing = []
        if not ref_ok:  missing.append("reference molecule")
        if not scaf_ok: missing.append("scaffold selection")
        if not db_ok:   missing.append("clinical drug database")
        st.caption(f"Waiting for: {', '.join(missing)}")

    if run_btn:
        st.session_state["scored"]    = []
        st.session_state["rl_scored"] = []

        prog   = st.progress(0, text="Initialising…")
        status = st.empty()
        log_box = st.empty()

        logs = []
        def log_step(msg):
            logs.append(msg)
            log_box.markdown(
                '<div style="font-family:monospace;font-size:11px;background:#1e1e1e;'
                'color:#d4d4d4;padding:10px;border-radius:6px;height:120px;overflow-y:auto">'
                + "<br>".join(logs[-8:]) + "</div>",
                unsafe_allow_html=True,
            )

        try:
            ref_mol       = st.session_state["ref_mol"]
            scaffold_sma  = st.session_state["scaffold_smarts"]
            drug_db_path  = st.session_state["drug_db_path"]

            # Step 1
            log_step("▶ Step 1/5 — Splitting molecule into fixed / scaffold regions…")
            status.info("Step 1/5 — Splitting molecule…")
            splitter = MoleculeSplitter(ref_mol, scaffold_sma)
            regions  = splitter.split()
            prog.progress(15)
            log_step(
                f"   ✔ Fixed atoms: {len(regions.fixed_atom_indices)} | "
                f"Scaffold atoms: {len(regions.scaffold_atom_indices)} | "
                f"Attachment points: {len(regions.attachment_points)}"
            )
            for i, ap in enumerate(regions.attachment_points):
                log_step(
                    f"   AP{i}: {ap.fixed_atom_symbol}({ap.fixed_atom_hybridization.name}) "
                    f"--[{ap.bond_type.name}]--> "
                    f"{ap.scaffold_atom_symbol}({ap.scaffold_atom_hybridization.name})"
                )

            # Step 2
            log_step("▶ Step 2/5 — Building fragment library from clinical drugs…")
            status.info("Step 2/5 — Building fragment library…")
            lib = FragmentLibrary(
                drug_db_path=drug_db_path,
                min_attachments=int(min_attach),
                max_attachments=int(max_attach),
                use_cache=True,
            )
            lib.build()
            prog.progress(40)
            log_step(f"   ✔ Fragment library: {len(lib):,} unique fragments")

            # Step 3+4
            log_step("▶ Step 3/5 — Matching + enumerating scaffold replacements…")
            status.info("Step 3/5 — Enumerating replacements…")
            replacer   = ScaffoldReplacer(regions, lib)
            candidates = replacer.enumerate()
            prog.progress(60)
            log_step(f"   ✔ {len(candidates):,} candidate molecules generated")

            if not candidates:
                prog.empty()
                status.error(
                    "No candidates generated. Suggestions: "
                    "relax hybridization in sidebar, expand attachment point range, "
                    "or try a different scaffold selection."
                )
                st.stop()

            # Step 5 — scoring
            log_step(f"▶ Step 4/5 — Scoring {len(candidates):,} candidates…")
            status.info(f"Step 4/5 — Scoring {len(candidates):,} candidates…")

            import scorer as scorer_module
            scorer_module.WEIGHTS.update({
                "shape_similarity": w_shape,
                "pharmacophore_sim": w_pharm,
                "scaffold_novelty":  w_novel,
                "tox_score":         w_tox,
                "logp_score":        w_logp,
                "sa_score":          w_sa,
            })

            receptor_path = None
            if receptor_file:
                rp = Path(tempfile.mktemp(suffix=".pdbqt"))
                rp.write_bytes(receptor_file.read())
                receptor_path = str(rp)

            scorer = MoleculeScorer(
                reference=ref_mol,
                regions=regions,
                logp_max=logp_max,
                receptor_pdbqt=receptor_path,
                vina_box=vina_box_str or None,
                n_workers=1,
            )
            scored = scorer.score_all(candidates)

            # Apply UI filter thresholds
            for sc in scored:
                sc.passes_filters = True
                sc.filter_failures = []
                if sc.mw   > mw_max:
                    sc.passes_filters = False
                    sc.filter_failures.append(f"MW={sc.mw:.0f}>{mw_max}")
                if sc.logp > logp_max:
                    sc.passes_filters = False
                    sc.filter_failures.append(f"logP={sc.logp:.2f}>{logp_max}")
                if sc.tpsa > tpsa_max:
                    sc.passes_filters = False
                    sc.filter_failures.append(f"TPSA={sc.tpsa:.0f}>{tpsa_max}")

            scored.sort(key=lambda x: x.total_score, reverse=True)
            scored = scored[: int(top_n)]
            prog.progress(85)
            passing = sum(1 for s in scored if s.passes_filters)
            log_step(f"   ✔ {len(scored)} scored | {passing} pass all filters")

            # Step 6 — outputs
            log_step("▶ Step 5/5 — Writing outputs…")
            status.info("Step 5/5 — Writing outputs…")
            reporter = Reporter(regions, ref_mol)
            reporter.write_csv(scored, "results.csv")
            reporter.write_sdf(scored, "results.sdf")
            reporter.write_html_report(scored, "results.html")
            prog.progress(100, text="✔ Done!")
            log_step(f"   ✔ results.csv / results.sdf / results.html written")

            status.success(
                f"Pipeline complete — **{len(scored)} candidates** scored "
                f"({passing} pass filters). See **Results** tab."
            )
            st.session_state["scored"]  = scored
            st.session_state["regions"] = regions

        except Exception as e:
            prog.empty()
            status.error(f"Pipeline error: {e}")
            log_step(f"   ✗ ERROR: {e}")
            log.exception("Pipeline failed")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════

with tab_results:
    st.subheader("🏆 Scored Candidates")

    scored = st.session_state.get("scored", [])
    if not scored:
        st.info("Run the pipeline (▶ Run tab) to generate results.")
    else:
        passing = [s for s in scored if s.passes_filters]
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Total",         len(scored))
        m2.metric("Pass filters",  len(passing))
        m3.metric("Best score",    f"{scored[0].total_score:.3f}")
        m4.metric("Avg logP",      f"{sum(s.logp for s in scored)/len(scored):.2f}")
        m5.metric("Avg MW",        f"{sum(s.mw  for s in scored)/len(scored):.1f}")

        st.markdown("---")

        # Filters
        rf1,rf2,rf3,rf4 = st.columns(4)
        pass_only   = rf1.checkbox("Passing filters only", False)
        sort_by     = rf2.selectbox("Sort by",
                        ["total_score","shape_similarity","scaffold_novelty",
                         "tox_score","logp","mw","qed"])
        sort_asc    = rf3.checkbox("Ascending", False)
        drug_filter = rf4.text_input("Filter by source drug", "")

        display = [s for s in scored if (not pass_only or s.passes_filters)]
        if drug_filter:
            display = [s for s in display
                       if drug_filter.lower() in s.candidate.source_fragment.source_drug.lower()]
        display.sort(key=lambda x: getattr(x, sort_by, 0.0), reverse=not sort_asc)

        st.caption(f"Showing {len(display)} candidates")

        # Card grid
        CPR = 3
        for row_s in range(0, min(len(display), 60), CPR):
            row_items = display[row_s : row_s + CPR]
            cols = st.columns(CPR)
            for col, sc in zip(cols, row_items):
                frag = sc.candidate.source_fragment
                rank = scored.index(sc) + 1
                svg  = mol_to_svg(sc.candidate.mol, 240, 170)
                ptag = (
                    '<span style="color:#2ecc71;font-size:11px">✔ PASS</span>'
                    if sc.passes_filters
                    else f'<span style="color:#e74c3c;font-size:11px">✘ '
                         f'{"  ".join(sc.filter_failures)}</span>'
                )
                col.markdown(
                    f"""<div style="border:1px solid #ddd;border-radius:8px;padding:10px;
                            background:#fafafa;margin-bottom:8px">
                      <div style="font-size:11px;color:#777">
                        #{rank} · <b>{frag.source_drug}</b> · {frag.frag_type}</div>
                      <div style="background:#fff;border-radius:4px;padding:2px">{svg}</div>
                      <div style="font-family:monospace;font-size:9px;color:#555;
                                  word-break:break-all;margin:4px 0">{sc.smiles}</div>
                      <table style="font-size:11px;width:100%;border-collapse:collapse">
                        <tr><td>Score</td><td>{score_color(sc.total_score)}</td>
                            <td>Shape</td><td>{score_color(sc.shape_similarity)}</td></tr>
                        <tr><td>Novelty</td><td>{score_color(sc.scaffold_novelty)}</td>
                            <td>Tox</td><td>{score_color(sc.tox_score)}</td></tr>
                        <tr><td>logP</td><td>{sc.logp:.2f}</td>
                            <td>MW</td><td>{sc.mw:.1f}</td></tr>
                        <tr><td>TPSA</td><td>{sc.tpsa:.1f}</td>
                            <td>QED</td><td>{sc.qed:.3f}</td></tr>
                      </table>
                      <div style="margin-top:5px">{ptag}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # Charts
        st.markdown("---")
        st.subheader("Score Analytics")
        try:
            import altair as alt
            import pandas as pd

            rows = []
            for sc in display:
                frag = sc.candidate.source_fragment
                rows.append({
                    "Total Score":        sc.total_score,
                    "Shape Similarity":   sc.shape_similarity,
                    "Scaffold Novelty":   sc.scaffold_novelty,
                    "Tox Safety":         sc.tox_score,
                    "logP":               sc.logp,
                    "MW":                 sc.mw,
                    "QED":                sc.qed,
                    "Source Drug":        frag.source_drug,
                    "Fragment Type":      frag.frag_type,
                    "Passes":             sc.passes_filters,
                    "SMILES":             sc.smiles[:45] + "…",
                })
            df = pd.DataFrame(rows)

            ch1, ch2 = st.columns(2)
            with ch1:
                st.altair_chart(
                    alt.Chart(df).mark_bar(color="#3a7abf", opacity=0.85)
                    .encode(alt.X("Total Score", bin=alt.Bin(maxbins=20)), alt.Y("count()"),
                            tooltip=["count()"])
                    .properties(title="Total Score Distribution", height=220),
                    use_container_width=True,
                )
            with ch2:
                st.altair_chart(
                    alt.Chart(df).mark_circle(size=55, opacity=0.7)
                    .encode(
                        x=alt.X("Shape Similarity", scale=alt.Scale(domain=[0,1])),
                        y=alt.Y("Scaffold Novelty", scale=alt.Scale(domain=[0,1])),
                        color=alt.Color("Total Score", scale=alt.Scale(scheme="viridis")),
                        shape=alt.Shape("Fragment Type"),
                        tooltip=["SMILES","Total Score","Shape Similarity",
                                 "Scaffold Novelty","logP","Source Drug"],
                    )
                    .properties(title="Shape vs Novelty (color=score, shape=frag type)", height=220),
                    use_container_width=True,
                )

            ch3, ch4 = st.columns(2)
            with ch3:
                st.altair_chart(
                    alt.Chart(df).mark_bar(color="#e67e22", opacity=0.8)
                    .encode(alt.X("logP", bin=alt.Bin(maxbins=20)), alt.Y("count()"))
                    .properties(title="logP Distribution", height=180),
                    use_container_width=True,
                )
            with ch4:
                src_counts = df.groupby("Source Drug").size().reset_index(name="count")
                src_counts = src_counts.nlargest(10, "count")
                st.altair_chart(
                    alt.Chart(src_counts).mark_bar(color="#9b59b6")
                    .encode(
                        x=alt.X("count:Q"),
                        y=alt.Y("Source Drug:N", sort="-x"),
                        tooltip=["Source Drug","count"],
                    )
                    .properties(title="Top 10 Source Drugs", height=220),
                    use_container_width=True,
                )

        except ImportError:
            st.info("Install altair + pandas for charts.")

        # Downloads
        st.markdown("---")
        st.subheader("Downloads")
        dl1, dl2, dl3 = st.columns(3)
        if Path("results.csv").exists():
            dl1.download_button("⬇ CSV", Path("results.csv").read_bytes(),
                                "scaffold_hop_results.csv", "text/csv",
                                use_container_width=True)
        if Path("results.sdf").exists():
            dl2.download_button("⬇ SDF", Path("results.sdf").read_bytes(),
                                "scaffold_hop_results.sdf", "chemical/x-mdl-sdfile",
                                use_container_width=True)
        if Path("results.html").exists():
            dl3.download_button("⬇ HTML Report", Path("results.html").read_bytes(),
                                "scaffold_hop_results.html", "text/html",
                                use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RL GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

with tab_rl:
    st.subheader("🤖 RL-Based Scaffold Generator")
    st.markdown(
        "Cross-entropy / REINVENT-style policy that re-weights the fragment action space "
        "toward higher-scoring scaffolds across multiple exploration rounds."
    )

    if not st.session_state["regions"]:
        st.info("Run the deterministic pipeline first (▶ Run tab) to initialise molecule regions.")
    else:
        rc1, rc2 = st.columns(2)
        rl_steps   = rc1.number_input("RL steps",     50, 2000, 200, 50)
        rl_batch   = rc1.number_input("Batch size",    4,  128,  16,  4)
        rl_temp    = rc2.slider("Sampling temp",  0.1, 3.0, 1.0, 0.1)
        rl_div     = rc2.slider("Diversity pen.", 0.0, 0.5, 0.1, 0.05)
        rl_elite   = rc2.slider("Elite fraction", 0.1, 0.9, 0.3, 0.05)

        rl_btn = st.button("▶ Run RL Generator", type="primary", use_container_width=True)

        if rl_btn:
            from rl_generator import RLScaffoldHopper, RLConfig
            import scorer as scorer_module

            rl_prog   = st.progress(0, text="Initialising RL…")
            rl_status = st.empty()

            try:
                regions      = st.session_state["regions"]
                ref_m        = st.session_state["ref_mol"]
                drug_db_path = st.session_state["drug_db_path"]

                lib = FragmentLibrary(
                    drug_db_path=drug_db_path,
                    min_attachments=int(min_attach),
                    max_attachments=int(max_attach),
                    use_cache=True,
                )
                lib.build()

                scorer_module.WEIGHTS.update({
                    "shape_similarity": w_shape, "pharmacophore_sim": w_pharm,
                    "scaffold_novelty": w_novel, "tox_score": w_tox,
                    "logp_score": w_logp, "sa_score": w_sa,
                })
                scr = MoleculeScorer(reference=ref_m, regions=regions,
                                     logp_max=logp_max, n_workers=1)
                cfg = RLConfig(
                    n_steps=int(rl_steps), batch_size=int(rl_batch),
                    temperature=rl_temp, diversity_penalty=rl_div, top_fraction=rl_elite,
                )
                rl_status.info("Running RL exploration…")
                hopper    = RLScaffoldHopper(regions, lib, scr, cfg)
                rl_scored = hopper.run()
                rl_scored = rl_scored[: int(top_n)]
                rl_prog.progress(100, text="RL complete!")
                rl_status.success(f"Found {len(rl_scored)} unique candidates via RL.")
                st.session_state["rl_scored"] = rl_scored

                reporter = Reporter(regions, ref_m)
                reporter.write_csv(rl_scored, "results_rl.csv")

            except Exception as e:
                rl_prog.empty()
                rl_status.error(f"RL error: {e}")
                log.exception("RL failed")

        rl_sc = st.session_state.get("rl_scored", [])
        if rl_sc:
            st.markdown(f"**Top {min(12, len(rl_sc))} RL candidates:**")
            CPR = 3
            for row_s in range(0, min(len(rl_sc), 12), CPR):
                cols = st.columns(CPR)
                for col, sc in zip(cols, rl_sc[row_s: row_s+CPR]):
                    frag = sc.candidate.source_fragment
                    svg  = mol_to_svg(sc.candidate.mol, 220, 155)
                    col.markdown(
                        f'<div style="border:1px solid #ddd;border-radius:6px;'
                        f'padding:8px;background:#fafafa;text-align:center">'
                        f'{svg}'
                        f'<div style="font-size:10px;margin-top:4px">'
                        f'score={sc.total_score:.3f} · logP={sc.logp:.2f} · MW={sc.mw:.0f}<br>'
                        f'<i>{frag.source_drug}</i></div></div>',
                        unsafe_allow_html=True,
                    )
            if Path("results_rl.csv").exists():
                st.download_button(
                    "⬇ Download RL Results CSV",
                    Path("results_rl.csv").read_bytes(),
                    "scaffold_hop_rl_results.csv", "text/csv",
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════

with tab_about:
    st.markdown(textwrap.dedent("""
    ## Scaffold Hopper — Clinical Drug Constrained

    ### What it does
    Replaces a user-selected scaffold region of a reference molecule with
    fragments derived **exclusively from FDA/EMA Phase-4 approved drugs** (via ChEMBL),
    while preserving the fixed substituent region atom-for-atom.

    ### Pipeline
    | Step | Module | Description |
    |------|--------|-------------|
    | 1 | `molecule_splitter` | SMARTS-based split; extracts attachment point bond types & hybridization |
    | 2 | `clinical_drugs`    | Downloads & caches ~2,000 ChEMBL approved drugs |
    | 3 | `fragment_library`  | BRICS + Murcko + ring-system fragmentation; disk-cached |
    | 4 | `scaffold_replacer` | Attachment-compatible enumeration; no bond modification |
    | 5 | `scorer`            | Shape (USRCAT), pharmacophore, tox alerts, SA score, optional docking |
    | 6 | `rl_generator`      | Cross-entropy policy on fragment action space |

    ### Scaffold Marking
    - **Ketcher** — draw molecule from scratch in a full chemical editor
    - **Click atoms** — interactive SVG with per-atom click targets
    - **Quick presets** — one-click All Rings / Murcko / Aromatic
    - **SMARTS override** — manual fine-grained control

    ### Fragment Integrity Rules
    - Bond type at every attachment point must match the original exactly
    - Hybridization match enforced by default (relaxable in sidebar)
    - No invented bonds — only bonds that exist in the source approved drug

    ### CLI
    ```bash
    py -3 scaffold_hop.py \\
        --reference "CC(=O)Nc1ccc(O)cc1" \\
        --scaffold_smarts "c1ccc(cc1)" \\
        --drug_db clinical_drugs.smi \\
        --output results.csv --top_n 100
    ```
    """))
