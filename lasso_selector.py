"""
lasso_selector.py — Lasso-based scaffold atom selector using components.html()
with a JavaScript → Streamlit text_input bridge.

No declared component server required. Works in all Streamlit environments.
"""

from __future__ import annotations
from typing import Set, List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold
import streamlit as st
import streamlit.components.v1 as components


# ── Atom metadata extraction ───────────────────────────────────────────────

def _atom_draw_data(mol: Chem.Mol, width: int, height: int) -> List[dict]:
    """Get each atom's pixel position (from RDKit draw coords) + metadata."""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addAtomIndices = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    data = []
    for i in range(mol.GetNumAtoms()):
        pt   = drawer.GetDrawCoords(i)
        atom = mol.GetAtomWithIdx(i)
        data.append({
            "x":         round(pt.x, 2),
            "y":         round(pt.y, 2),
            "symbol":    atom.GetSymbol(),
            "isAromatic": atom.GetIsAromatic(),
            "inRing":    atom.IsInRing(),
        })
    return data


def _mol_svg(mol: Chem.Mol, width: int, height: int) -> str:
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts   = drawer.drawOptions()
    opts.addStereoAnnotation = True
    opts.addAtomIndices      = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    raw = drawer.GetDrawingText()
    return raw[raw.find("<svg"):]


def _ring_sets(mol: Chem.Mol) -> List[List[int]]:
    return [list(r) for r in mol.GetRingInfo().AtomRings()]


def _murcko_atoms(mol: Chem.Mol) -> List[int]:
    try:
        core  = MurckoScaffold.GetScaffoldForMol(mol)
        match = mol.GetSubstructMatch(core)
        return list(match)
    except Exception:
        return []


def selection_to_smarts(mol: Chem.Mol, selected: Set[int]) -> str:
    """Convert selected atom indices to a SMARTS pattern."""
    if not selected:
        return ""
    from rdkit.Chem import RWMol
    rw = RWMol()
    old_to_new = {}
    for old_idx in sorted(selected):
        atom = mol.GetAtomWithIdx(old_idx)
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetIsAromatic(atom.GetIsAromatic())
        old_to_new[old_idx] = rw.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in old_to_new and a2 in old_to_new:
            rw.AddBond(old_to_new[a1], old_to_new[a2], bond.GetBondType())
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        pass
    return Chem.MolToSmarts(rw.GetMol())


# ── Main component ─────────────────────────────────────────────────────────

def lasso_mol_selector(
    mol: Chem.Mol,
    key: str = "lasso",
    width: int = 720,
    height: int = 440,
    current_selection: Optional[Set[int]] = None,
) -> Set[int]:
    """
    Render the lasso molecule selector.
    Returns confirmed selected atom indices as Set[int].

    Bridge: JS writes comma-separated indices to a hidden st.text_input.
    Streamlit reads it on the next rerun after the user clicks Confirm.
    """
    if mol is None:
        return set()

    AllChem.Compute2DCoords(mol)

    sel_key = f"_lasso_confirmed_{key}"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = set(current_selection or [])

    svg         = _mol_svg(mol, width, height)
    atom_data   = _atom_draw_data(mol, width, height)
    rings       = _ring_sets(mol)
    murcko      = _murcko_atoms(mol)
    init_sel    = list(st.session_state[sel_key])

    # Serialize data for JS
    import json
    atom_json   = json.dumps(atom_data)
    rings_json  = json.dumps(rings)
    murcko_json = json.dumps(murcko)
    init_json   = json.dumps(init_sel)
    bridge_id   = f"lasso_bridge_{key}"

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #f8f9fa; font-family: Arial, sans-serif; overflow: hidden; }}

  #wrap {{
    display: flex;
    flex-direction: column;
    width: {width}px;
  }}

  #canvas-area {{
    position: relative;
    width: {width}px;
    height: {height}px;
    background: #ffffff;
    border: 1.5px solid #ced4da;
    border-bottom: none;
    border-radius: 8px 8px 0 0;
    overflow: hidden;
    cursor: crosshair;
  }}

  #svg-bg {{
    position: absolute; top: 0; left: 0;
    pointer-events: none;
  }}
  #svg-bg svg {{ width: {width}px; height: {height}px; }}

  #overlay {{
    position: absolute; top: 0; left: 0;
    width: {width}px; height: {height}px;
  }}

  #toolbar {{
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 7px 10px;
    background: #f0f4f8;
    border: 1.5px solid #ced4da;
    border-top: 1px solid #e2e8f0;
    border-radius: 0 0 8px 8px;
    flex-wrap: wrap;
  }}

  .btn {{
    padding: 4px 11px;
    border: none;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    transition: filter .12s;
    white-space: nowrap;
  }}
  .btn:hover  {{ filter: brightness(1.1); }}
  .btn:active {{ filter: brightness(0.9); }}

  #btn-confirm  {{ background:#2ecc71; color:#fff; }}
  #btn-clear    {{ background:#e74c3c; color:#fff; }}
  #btn-invert   {{ background:#3498db; color:#fff; }}
  #btn-rings    {{ background:#8e44ad; color:#fff; }}
  #btn-murcko   {{ background:#d35400; color:#fff; }}
  #btn-aromatic {{ background:#16a085; color:#fff; }}

  #sel-info {{
    flex: 1;
    font-size: 11px;
    font-family: monospace;
    color: #2c3e50;
    padding: 2px 6px;
    background: #fff;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    min-width: 120px;
    max-width: 320px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}

  #hint {{
    font-size: 10px;
    color: #aaa;
    margin-left: auto;
  }}

  /* Confirmed flash */
  #confirmed-flash {{
    display: none;
    position: absolute;
    top: 8px; left: 50%;
    transform: translateX(-50%);
    background: rgba(46,204,113,0.92);
    color: white;
    font-size: 13px;
    font-weight: bold;
    padding: 6px 18px;
    border-radius: 20px;
    pointer-events: none;
    z-index: 99;
  }}
</style>
</head>
<body>
<div id="wrap">
  <div id="canvas-area">
    <div id="svg-bg">{svg}</div>
    <canvas id="overlay"></canvas>
    <div id="confirmed-flash">✔ Selection confirmed</div>
  </div>
  <div id="toolbar">
    <button class="btn" id="btn-confirm"  onclick="confirmSel()">✔ Confirm</button>
    <button class="btn" id="btn-clear"    onclick="clearSel()">✕ Clear</button>
    <button class="btn" id="btn-invert"   onclick="invertSel()">⇄ Invert</button>
    <button class="btn" id="btn-rings"    onclick="selRings()">⬡ Rings</button>
    <button class="btn" id="btn-murcko"   onclick="selMurcko()">◎ Murcko</button>
    <button class="btn" id="btn-aromatic" onclick="selAromatic()">✦ Aromatic</button>
    <span id="sel-info">No atoms selected</span>
    <span id="hint">Lasso · click · toolbar</span>
  </div>
</div>

<script>
/* ── Data from Python ────────────────────────────────────────────────── */
const ATOMS   = {atom_json};
const RINGS   = {rings_json};
const MURCKO  = {murcko_json};
const W = {width}, H = {height};
const BRIDGE_ID = "{bridge_id}";

/* ── State ───────────────────────────────────────────────────────────── */
let selected = new Set({init_json});
let lasso    = [];
let drawing  = false;
let canvas, ctx;

/* ── Init ────────────────────────────────────────────────────────────── */
window.onload = function() {{
  canvas = document.getElementById("overlay");
  canvas.width  = W;
  canvas.height = H;
  ctx = canvas.getContext("2d");

  canvas.addEventListener("mousedown",  onDown);
  canvas.addEventListener("mousemove",  onMove);
  canvas.addEventListener("mouseup",    onUp);
  canvas.addEventListener("mouseleave", onUp);
  canvas.addEventListener("touchstart", e => {{ e.preventDefault(); onDown(e.touches[0]); }}, {{passive:false}});
  canvas.addEventListener("touchmove",  e => {{ e.preventDefault(); onMove(e.touches[0]); }}, {{passive:false}});
  canvas.addEventListener("touchend",   e => {{ e.preventDefault(); onUp(e.changedTouches[0]); }});

  redraw();
}};

/* ── Mouse / Touch ───────────────────────────────────────────────────── */
function xy(e) {{
  const r = canvas.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}}

function onDown(e) {{
  const [mx, my] = xy(e);
  // Click on atom → toggle
  for (let i = 0; i < ATOMS.length; i++) {{
    if (Math.hypot(ATOMS[i].x - mx, ATOMS[i].y - my) < 16) {{
      selected.has(i) ? selected.delete(i) : selected.add(i);
      redraw();
      return;
    }}
  }}
  // Start lasso
  drawing = true;
  lasso = [[mx, my]];
}}

function onMove(e) {{
  if (!drawing) return;
  const [mx, my] = xy(e);
  lasso.push([mx, my]);
  redraw();
}}

function onUp(e) {{
  if (!drawing) return;
  drawing = false;
  if (lasso.length > 3) {{
    atomsInPoly(lasso).forEach(i => selected.add(i));
  }}
  lasso = [];
  redraw();
}}

/* ── Geometry ────────────────────────────────────────────────────────── */
function pip(px, py, poly) {{
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {{
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if (((yi > py) !== (yj > py)) &&
        (px < (xj - xi) * (py - yi) / (yj - yi) + xi))
      inside = !inside;
  }}
  return inside;
}}

function atomsInPoly(poly) {{
  return ATOMS.map((a, i) => ({{i, ...a}}))
    .filter(a => pip(a.x, a.y, poly))
    .map(a => a.i);
}}

/* ── Draw ────────────────────────────────────────────────────────────── */
function redraw() {{
  ctx.clearRect(0, 0, W, H);

  /* selected atom halos */
  ATOMS.forEach((a, i) => {{
    if (!selected.has(i)) return;
    ctx.beginPath();
    ctx.arc(a.x, a.y, 15, 0, Math.PI*2);
    ctx.fillStyle   = "rgba(231,76,60,0.28)";
    ctx.fill();
    ctx.strokeStyle = "rgba(192,57,43,0.75)";
    ctx.lineWidth   = 2;
    ctx.stroke();
  }});

  /* atom index badges */
  ATOMS.forEach((a, i) => {{
    const sel = selected.has(i);
    ctx.beginPath();
    ctx.arc(a.x - 15, a.y - 15, 7.5, 0, Math.PI*2);
    ctx.fillStyle = sel ? "#c0392b" : "#95a5a6";
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.font      = "bold 7.5px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(i, a.x - 15, a.y - 15);
  }});

  /* lasso path */
  if (lasso.length > 1) {{
    ctx.beginPath();
    ctx.moveTo(lasso[0][0], lasso[0][1]);
    lasso.forEach(([lx, ly]) => ctx.lineTo(lx, ly));
    ctx.closePath();
    ctx.fillStyle   = "rgba(52,152,219,0.09)";
    ctx.fill();
    ctx.strokeStyle = "#3498db";
    ctx.lineWidth   = 2.2;
    ctx.setLineDash([7, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
  }}

  /* update sel-info label */
  const el = document.getElementById("sel-info");
  if (selected.size === 0) {{
    el.textContent = "No atoms selected";
    el.style.color  = "#95a5a6";
  }} else {{
    const idxs = [...selected].sort((a,b) => a-b);
    el.textContent = `${{selected.size}} atom(s): ${{idxs.join(", ")}}`;
    el.style.color  = "#c0392b";
  }}
}}

/* ── Toolbar actions ─────────────────────────────────────────────────── */
function confirmSel() {{
  const val = [...selected].sort((a,b)=>a-b).join(",");
  pushToBridge(val);

  // Flash feedback
  const flash = document.getElementById("confirmed-flash");
  flash.style.display = "block";
  setTimeout(() => {{ flash.style.display = "none"; }}, 1400);
}}

function clearSel() {{
  selected.clear();
  redraw();
  pushToBridge("");
}}

function invertSel() {{
  const all = new Set(ATOMS.map((_, i) => i));
  selected = new Set([...all].filter(i => !selected.has(i)));
  redraw();
}}

function selRings()    {{ RINGS.forEach(r  => r.forEach(i => selected.add(i)));  redraw(); }}
function selMurcko()   {{ MURCKO.forEach(i => selected.add(i));                   redraw(); }}
function selAromatic() {{ ATOMS.forEach((a,i) => {{ if (a.isAromatic) selected.add(i); }}); redraw(); }}

/* ── Bridge: write to Streamlit text_input via placeholder lookup ────── */
function pushToBridge(value) {{
  try {{
    const doc = window.parent.document;
    // Find by placeholder attribute — stable across Streamlit versions
    const inp = doc.querySelector(`input[placeholder="${{BRIDGE_ID}}"]`);
    if (!inp) {{ console.warn("Bridge input not found for", BRIDGE_ID); return; }}

    // Use React's native setter so the synthetic event fires correctly
    const nativeSetter = Object.getOwnPropertyDescriptor(
      window.parent.HTMLInputElement.prototype, 'value'
    ).set;
    nativeSetter.call(inp, value);

    // Dispatch both input and change events — Streamlit listens to both
    inp.dispatchEvent(new window.parent.Event('input',  {{ bubbles: true }}));
    inp.dispatchEvent(new window.parent.Event('change', {{ bubbles: true }}));

    // Focus + blur forces Streamlit to commit the new value
    inp.focus();
    inp.blur();
  }} catch(e) {{
    console.warn("Bridge write failed:", e);
  }}
}}
</script>
</body>
</html>
"""

    # ── Render HTML component ──────────────────────────────────────────────
    components.html(html, height=height + 54, scrolling=False)

    # ── Bridge text_input ──────────────────────────────────────────────────
    # The placeholder is the unique token JS uses to find this input via
    #   document.querySelector(`input[placeholder="${BRIDGE_ID}"]`)
    # It is VISIBLE so the user always sees what's confirmed and can edit
    # it manually as a fallback.
    current_str = ",".join(str(i) for i in sorted(st.session_state[sel_key]))

    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        bridge_val = st.text_input(
            "Confirmed scaffold atom indices",
            value=current_str,
            placeholder=bridge_id,          # ← JS finds this
            key=f"_bridge_input_{key}",
            help=(
                "Auto-filled when you click **✔ Confirm** in the lasso above. "
                "You can also type or paste atom indices manually (comma-separated), "
                "then press Enter or click Apply."
            ),
        )
    with col_btn:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        apply = st.button("Apply", key=f"_apply_{key}", use_container_width=True,
                          type="primary")

    # Parse — accepts both the auto-filled and manual values
    n_atoms = mol.GetNumAtoms()
    try:
        parsed: Set[int] = {
            int(x.strip())
            for x in bridge_val.split(",")
            if x.strip().lstrip("-").isdigit()
            and 0 <= int(x.strip()) < n_atoms
        }
    except Exception:
        parsed = set()

    # Update session state on either Apply click or auto-bridge fire
    if apply or bridge_val != current_str:
        st.session_state[sel_key] = parsed

    return st.session_state[sel_key]
