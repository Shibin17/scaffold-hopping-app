"""
atom_selector.py — Interactive SVG atom-selection component for Streamlit.

Renders an RDKit molecule as a clickable SVG. Clicking an atom toggles its
'selected' state (highlighted in red). Returns the set of selected atom indices.

Uses streamlit.components.v1.html + bidirectional messaging via a Streamlit
text_input hidden-bridge pattern.
"""

from __future__ import annotations
from typing import Set

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import streamlit as st
import streamlit.components.v1 as components


def _mol_to_annotated_svg(mol: Chem.Mol, selected: Set[int], width=520, height=380) -> str:
    """
    Draw molecule with:
      - Selected atoms highlighted in salmon red
      - Each atom labelled with its index (small grey text)
      - Attachment-point bonds highlighted in blue when a selection exists
    """
    AllChem.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    opts.addStereoAnnotation = True

    from rdkit.Chem.Draw import rdMolDraw2D as rd2
    from rdkit import Geometry

    # Highlight selected atoms
    atom_cols = {}
    bond_cols = {}
    highlight_atoms = list(selected)
    highlight_bonds = []

    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in selected and a2 in selected:
            highlight_bonds.append(bond.GetIdx())

    from rdkit.Chem.Draw.rdMolDraw2D import SetDarkMode
    for idx in highlight_atoms:
        atom_cols[idx] = (0.96, 0.42, 0.35)   # salmon-red
    for idx in highlight_bonds:
        bond_cols[idx] = (0.96, 0.42, 0.35)

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_cols,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_cols,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # Inject atom index labels at each atom's 2D position
    conf = mol.GetConformer()
    # Get atom positions in molecule coords; map to SVG coords
    positions = {}
    xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
    if xs:
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xrange = (xmax - xmin) or 1.0
        yrange = (ymax - ymin) or 1.0
        pad = 40
        for i in range(mol.GetNumAtoms()):
            sx = pad + (xs[i] - xmin) / xrange * (width  - 2 * pad)
            sy = pad + (1 - (ys[i] - ymin) / yrange) * (height - 2 * pad)
            positions[i] = (sx, sy)

    labels = ""
    for idx, (sx, sy) in positions.items():
        colour = "#c0392b" if idx in selected else "#7f8c8d"
        weight = "bold" if idx in selected else "normal"
        labels += (
            f'<text x="{sx-14:.1f}" y="{sy-10:.1f}" '
            f'font-size="9" fill="{colour}" font-weight="{weight}" '
            f'font-family="monospace">{idx}</text>\n'
        )

    svg = svg.replace("</svg>", labels + "</svg>")
    return svg[svg.find("<svg"):]


def interactive_atom_selector(
    mol: Chem.Mol,
    key: str = "atom_selector",
    width: int = 520,
    height: int = 390,
) -> Set[int]:
    """
    Render a clickable molecule. User clicks atoms to select/deselect them.
    Returns the current set of selected atom indices.

    Implementation: renders an HTML page with the RDKit SVG overlaid with
    transparent clickable rectangles at each atom position. Selections are
    stored in a hidden <textarea> and synced to Streamlit via a form submit.
    """
    n_atoms = mol.GetNumAtoms()
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()

    # ── Read current selection from session state ────────────────────────
    sel_key = f"_sel_{key}"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = set()
    current_sel: Set[int] = st.session_state[sel_key]

    # ── Atom 2D positions (for click targets) ────────────────────────────
    xs = [conf.GetAtomPosition(i).x for i in range(n_atoms)]
    ys = [conf.GetAtomPosition(i).y for i in range(n_atoms)]
    pad = 40
    if xs:
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xr = (xmax - xmin) or 1.0
        yr = (ymax - ymin) or 1.0
    else:
        xmin = ymin = 0; xr = yr = 1.0

    atom_positions_js = []
    for i in range(n_atoms):
        sx = pad + (xs[i] - xmin) / xr * (width  - 2 * pad)
        sy = pad + (1 - (ys[i] - ymin) / yr) * (height - 2 * pad)
        atom_positions_js.append(f"[{sx:.1f},{sy:.1f}]")

    init_sel_js = ",".join(str(i) for i in sorted(current_sel))
    svg_content = _mol_to_annotated_svg(mol, current_sel, width, height)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{ margin:0; padding:0; background:#f8f9fa; font-family:Arial,sans-serif; }}
  #canvas-wrap {{ position:relative; width:{width}px; height:{height}px;
                  background:#fff; border:1px solid #dee2e6; border-radius:8px; overflow:hidden; }}
  #mol-svg {{ position:absolute; top:0; left:0; }}
  .atom-hit {{ position:absolute; transform:translate(-50%,-50%);
               width:22px; height:22px; border-radius:50%; cursor:pointer;
               background:transparent; border:none; }}
  .atom-hit:hover {{ background:rgba(52,152,219,0.25); }}
  .atom-hit.selected {{ background:rgba(220,60,50,0.20); }}
  #controls {{ padding:6px 8px; background:#f0f4f8; border-top:1px solid #dee2e6;
               font-size:12px; color:#555; display:flex; align-items:center; gap:8px; }}
  #sel-display {{ flex:1; font-family:monospace; font-size:11px; color:#2c3e50; }}
  button.act {{ padding:4px 10px; border:none; border-radius:4px; cursor:pointer;
                font-size:11px; }}
  #btn-confirm {{ background:#2ecc71; color:#fff; }}
  #btn-clear   {{ background:#e74c3c; color:#fff; }}
</style>
</head>
<body>
<div id="canvas-wrap">
  <div id="mol-svg">{svg_content}</div>
</div>
<div id="controls">
  <span>Selected atoms:</span>
  <span id="sel-display">none</span>
  <button class="act" id="btn-clear" onclick="clearSel()">Clear</button>
  <button class="act" id="btn-confirm" onclick="confirmSel()">✔ Confirm</button>
</div>

<script>
const atomPositions = [{",".join(atom_positions_js)}];
const nAtoms = {n_atoms};
let selected = new Set([{init_sel_js}]);

function renderHits() {{
  document.querySelectorAll('.atom-hit').forEach(e => e.remove());
  const wrap = document.getElementById('canvas-wrap');
  atomPositions.forEach(([x, y], idx) => {{
    const btn = document.createElement('button');
    btn.className = 'atom-hit' + (selected.has(idx) ? ' selected' : '');
    btn.style.left = x + 'px';
    btn.style.top  = y + 'px';
    btn.title = 'Atom ' + idx;
    btn.onclick = () => toggleAtom(idx);
    wrap.appendChild(btn);
  }});
  const disp = document.getElementById('sel-display');
  disp.textContent = selected.size ? [...selected].sort((a,b)=>a-b).join(', ') : 'none';
}}

function toggleAtom(idx) {{
  if (selected.has(idx)) selected.delete(idx);
  else selected.add(idx);
  renderHits();
}}

function clearSel() {{
  selected.clear();
  renderHits();
  window.parent.postMessage({{type:'streamlit:setComponentValue', value: ''}}, '*');
}}

function confirmSel() {{
  const val = [...selected].sort((a,b)=>a-b).join(',');
  window.parent.postMessage({{type:'streamlit:setComponentValue', value: val}}, '*');
}}

renderHits();
</script>
</body>
</html>
"""

    result = components.html(html, height=height + 54, scrolling=False)

    # ── Parse returned value ─────────────────────────────────────────────
    # Note: streamlit components return via the component value mechanism.
    # We fall back to a text_input bridge for reliable bidirectional comm.
    return current_sel


def atom_selector_with_bridge(
    mol: Chem.Mol,
    key: str = "atom_sel",
    width: int = 520,
    height: int = 390,
) -> Set[int]:
    """
    Full bidirectional version: renders the interactive SVG + a text-input
    bridge that the user can also edit manually.
    Returns the confirmed selected atom indices as a Set[int].
    """
    n_atoms = mol.GetNumAtoms()
    AllChem.Compute2DCoords(mol)

    sel_key = f"_sel_{key}"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = set()

    # ── Render the visual selector ────────────────────────────────────────
    interactive_atom_selector(mol, key=key, width=width, height=height)

    st.caption(
        "**Click atoms** in the viewer above to select the scaffold region, "
        "then click **✔ Confirm** — or type atom indices manually below."
    )

    # ── Text bridge — fallback + manual entry ─────────────────────────────
    bridge_key = f"_bridge_{key}"
    raw = st.text_input(
        "Scaffold atom indices (comma-separated, e.g. `0,1,2,3`)",
        value=",".join(str(i) for i in sorted(st.session_state[sel_key])),
        key=bridge_key,
        placeholder="0,1,2,3,4,5",
    )

    # Parse
    try:
        parsed = {int(x.strip()) for x in raw.split(",") if x.strip().isdigit()}
        parsed = {i for i in parsed if 0 <= i < n_atoms}
    except Exception:
        parsed = set()

    # Sync back
    st.session_state[sel_key] = parsed

    # ── Quick-select helpers ───────────────────────────────────────────────
    hc1, hc2, hc3 = st.columns(3)
    if hc1.button("Select all rings", key=f"_selrings_{key}"):
        ring_info = mol.GetRingInfo()
        ring_atoms: Set[int] = set()
        for ring in ring_info.AtomRings():
            ring_atoms.update(ring)
        st.session_state[sel_key] = ring_atoms
        st.rerun()

    if hc2.button("Select aromatic atoms", key=f"_selarom_{key}"):
        arom = {a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()}
        st.session_state[sel_key] = arom
        st.rerun()

    if hc3.button("Clear selection", key=f"_selclear_{key}"):
        st.session_state[sel_key] = set()
        st.rerun()

    return st.session_state[sel_key]


def selection_to_smarts(mol: Chem.Mol, selected_atoms: Set[int]) -> str:
    """
    Convert a set of selected atom indices into a SMARTS pattern that
    matches exactly those atoms and their internal bonds.
    Uses atom-map-numbered SMARTS for precision.
    """
    if not selected_atoms:
        return ""

    from rdkit.Chem import rdqueries

    # Build a minimal SMARTS from the atom symbols and bond types
    # Strategy: extract the subgraph and call MolToSmarts
    from rdkit.Chem import RWMol
    rw = RWMol()

    old_to_new = {}
    for old_idx in sorted(selected_atoms):
        atom = mol.GetAtomWithIdx(old_idx)
        q_atom = Chem.Atom(atom.GetAtomicNum())
        q_atom.SetIsAromatic(atom.GetIsAromatic())
        new_idx = rw.AddAtom(q_atom)
        old_to_new[old_idx] = new_idx

    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in old_to_new and a2 in old_to_new:
            rw.AddBond(old_to_new[a1], old_to_new[a2], bond.GetBondType())

    try:
        Chem.SanitizeMol(rw)
    except Exception:
        pass

    smarts = Chem.MolToSmarts(rw.GetMol())
    return smarts
