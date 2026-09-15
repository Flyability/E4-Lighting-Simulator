"""Move self-contained line ranges out of ``ui/app.py::main()`` into ``ui/<module>.py``.

Each block becomes the body of ``build(ctx)``: the names it reads from ``main()`` are
unpacked from ``ctx`` at the top and the names ``main()`` keeps using are returned in a
SimpleNamespace. Run with ``--analyse`` first; it lists inputs/outputs and flags hazards
(names that main() rebinds later, or that the block rebinds through ``nonlocal``), which
must be turned into in-place mutation before ``--write``.

Usage: PYTHONPATH=. python scripts/extract_ui_block.py [--analyse | --write]
"""
import ast, builtins, re, symtable, sys
from pathlib import Path

APP = Path("lighting_simulator/ui/app.py")

# name, [(lo, hi), ...], docstring, extra imports, lists rebound with `x = []` -> `x.clear()`, marker
# The build(ctx) call replaces the FIRST range in place (marker=None) or is inserted before the
# first line containing `marker`; every input must be bound before the call.
OPTIM_MARKER = "# --- Optimize tab (see ui/optimize_tab.py) ---"
BLOCKS = [
    ("scene_view", [(1549, 3846)],
     "3-D scene: update_scene, wall/grid, selected-panel inspector, Panel Designer, and the scene-level GUI wiring.",
     ["import json", "import os", "import time", "import numpy as np", "import trimesh",
      "from lighting_simulator.camera.fov import (",
      "    camera_fov_wall_trapezoid as _camera_fov_wall_trapezoid, fov_plane_mask_to_quads_and_contour,",
      "    rasterize_fisheye_fov_on_plane, vio_hfov_vfov_deg, vio_optical_axis,",
      ")",
      "from lighting_simulator.domain.geometry import as_vec3 as _as_vec3",
      "from lighting_simulator.domain.guides import (",
      "    bake_and_disable_guide, circle_line_segments_m as _circle_line_segments_m,",
      "    dynamic_group_world_geometry as _dynamic_group_world_geometry, enable_circular_guide,",
      "    guide_is_enabled as _guide_is_enabled,",
      ")",
      "from lighting_simulator.domain.led_factory import create_leds",
      "from lighting_simulator.domain.optics import effective_lambertian_exponent as _get_effective_n",
      "from lighting_simulator.raytracing.mesh import ray_mesh_intersection as _ray_mesh_intersection",
      "from lighting_simulator.ui.mesh_lighting import _build_stl_transform"],
     ["led_handles", "ray_handles", "absorber_handles", "camera_fov_handles", "vio_fov_handles", "guide_handles"],
     OPTIM_MARKER),
]

src_lines = APP.read_text().splitlines(keepends=True)
tree = ast.parse("".join(src_lines))
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")

module_names = set(dir(builtins))
for n in tree.body:
    if isinstance(n, (ast.Import, ast.ImportFrom)):
        module_names.update((a.asname or a.name).split(".")[0] for a in n.names)
    elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
        module_names.add(n.name)
    elif isinstance(n, ast.Assign):
        module_names.update(x.id for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name))


def scope_stores(stmts):
    """name -> [lineno...] of direct (non-nested-def) stores in this scope."""
    out = {}
    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.setdefault(node.name, []).append(node.lineno); return
        if isinstance(node, ast.Lambda):
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.setdefault(node.id, []).append(node.lineno)
        for ch in ast.iter_child_nodes(node):
            visit(ch)
    for s in stmts:
        visit(s)
    return out


def nonlocal_stores(stmts):
    """Names rebound inside nested functions via `nonlocal` (name -> lines of `x = ...`)."""
    out = {}
    for s in stmts:
        for fn in ast.walk(s):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            nl = {nm for st in fn.body for n in ast.walk(st) if isinstance(n, ast.Nonlocal) for nm in n.names}
            if not nl:
                continue
            for n in ast.walk(fn):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id in nl:
                    out.setdefault(n.id, []).append(n.lineno)
    return out


def loads(stmts):
    names = set()
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                names.add(n.id)
            elif isinstance(n, ast.Nonlocal):
                names.update(n.names)
    return names


def in_block(stmt, ranges):
    return any(lo <= stmt.lineno <= hi for lo, hi in ranges)


LOOP_TEMPS = {"f", "i", "_h", "k", "j", "g", "h", "v", "n", "x", "y", "z", "c", "p", "d", "a", "b", "m", "s", "t",
              "group_idx", "row_idx", "led_idx", "led_in_row_idx", "led_btn", "row_btn", "btn", "color_hex",
              "html_content"}


def free_names(text):
    """Names the block reads without binding anywhere inside it (true free variables).

    ``nonlocal`` declarations are pre-bound so symtable accepts them; they count as free.
    """
    nl = {nm for n in ast.walk(ast.parse("def _b():\n" + text)) if isinstance(n, ast.Nonlocal) for nm in n.names}
    pre = "".join(f"    {n} = None\n" for n in sorted(nl))
    top = symtable.symtable("def _b():\n" + pre + text, "<block>", "exec")
    build = top.get_children()[0]
    out = set(nl)

    def walk(tab):
        for s in tab.get_symbols():
            if s.is_global() and not s.is_assigned():
                out.add(s.get_name())
        for ch in tab.get_children():
            walk(ch)
    walk(build)
    return out


def analyse(ranges, call_line):
    block = [s for s in main.body if in_block(s, ranges)]
    rest = [s for s in main.body if not in_block(s, ranges)]
    bb, ob = scope_stores(block), scope_stores(rest)
    rl = loads(rest)
    fr = free_names(block_text(ranges, []))
    inputs = sorted(n for n in fr if n in ob and n not in module_names)
    outputs = sorted(n for n in bb if n in rl and n not in LOOP_TEMPS)
    nl_rest = nonlocal_stores(rest)
    rebound_outside = {n: ob[n] + nl_rest.get(n, []) for n in inputs if len(ob[n]) > 1 or n in nl_rest}
    rebound_inside = {n: ls for n, ls in nonlocal_stores(block).items() if n in inputs or n in outputs}
    late_inputs = {n: ob[n] for n in inputs if min(ob[n]) > call_line}
    early_uses = {}
    def direct(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in outputs \
                and node.lineno < call_line:
            early_uses.setdefault(node.id, []).append(node.lineno)
        for ch in ast.iter_child_nodes(node):
            direct(ch)
    for s in rest:
        direct(s)
    return inputs, outputs, rebound_outside, rebound_inside, late_inputs, early_uses


def block_text(ranges, inplace):
    text = "".join("".join(src_lines[lo - 1:hi]) + "\n" for lo, hi in ranges)
    for name in inplace:
        text = re.sub(rf"^(\s*){name} = \[\]\s*$", rf"\1{name}.clear()", text, flags=re.M)
    return text


def undefined_names(text):
    tree_ = ast.parse(text)
    bound = set(dir(builtins))
    for n in ast.walk(tree_):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in n.names)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Lambda)):
            if not isinstance(n, ast.Lambda):
                bound.add(n.name)
            bound.update(a.arg for a in n.args.args + n.args.kwonlyargs)
            if n.args.vararg: bound.add(n.args.vararg.arg)
            if n.args.kwarg: bound.add(n.args.kwarg.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
    return sorted({n.id for n in ast.walk(tree_) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                   and n.id not in bound})


def main_():
    write = "--write" in sys.argv
    edits, out_files, all_outputs = [], {}, []
    for mod, ranges, doc, imports, inplace, marker in BLOCKS:
        call_line = (next(i + 1 for i, l in enumerate(src_lines) if marker in l) if marker else ranges[0][0])
        inputs, outputs, reb_out, reb_in, late, early = analyse(ranges, call_line)
        print(f"\n[{mod}] {ranges} -> call @{call_line}: {len(inputs)} inputs, {len(outputs)} outputs")
        print("  inputs :", ", ".join(inputs))
        print("  outputs:", ", ".join(outputs))
        if reb_out:
            print("  !! inputs REBOUND by main() elsewhere (ctx snapshot would go stale):", reb_out)
        if reb_in:
            print("  !! names rebound INSIDE block via nonlocal (main would go stale):", reb_in)
        if late:
            print("  !! inputs bound AFTER the call site (move the block or pass a late-bound holder):", late)
        if early:
            print("  !! outputs used directly by main() BEFORE the call site:", early)
        text = block_text(ranges, inplace)
        header = (f'"""{doc}\n\nExtracted verbatim from ``ui.app.main``; ``build(ctx)`` receives the GUI handles and\n'
                  f'callbacks it needs and returns the closures main() keeps using.\n"""\n'
                  "from types import SimpleNamespace\n" + "".join(i + "\n" for i in imports) + "\n\ndef build(ctx):\n"
                  + "".join(f"    {n} = ctx.{n}\n" for n in inputs) + "\n")
        footer = "\n    return SimpleNamespace(" + ", ".join(f"{n}={n}" for n in outputs) + ")\n"
        module_text = header + text + footer
        und = undefined_names(module_text)
        if und:
            print("  !! undefined names in generated module:", und)
        out_files[mod] = module_text
        call = (f"    # --- {doc.split(':')[0]} (see ui/{mod}.py) ---\n"
                f"    _{mod}_ns = _{mod}.build(_SimpleNamespace(\n"
                + "".join(f"        {n}={n},\n" for n in inputs) + "    ))\n"
                + "".join(f"    {n} = _{mod}_ns.{n}\n" for n in outputs))
        stubs = ""
        if marker and early:
            # main() hands these to earlier build(ctx) calls: forward to the real closures once built
            stubs = (f"    _{mod}_late = _SimpleNamespace()  # filled after ui.{mod}.build()\n"
                     + "".join(f"\n    def {n}(*a, **k):\n        return _{mod}_late.{n}(*a, **k)\n" for n in sorted(early))
                     + "\n")
            call += "".join(f"    _{mod}_late.{n} = _{mod}_ns.{n}\n" for n in sorted(early))
            print("  -> forwarding stubs emitted for:", sorted(early))
        edits.append((mod, ranges, call, marker, stubs))
    if not write:
        print("\n(dry run — pass --write to apply)")
        return
    lines = list(src_lines)
    # in-place blocks replace their first range; marker blocks are inserted before the marker (in
    # BLOCKS order) and all their ranges deleted. Everything bottom-up so line numbers stay valid.
    ops = []
    marker_inserts = {}
    for _, ranges, call, marker, stubs in edits:
        if marker:
            mline = next(i + 1 for i, l in enumerate(lines) if marker in l)
            marker_inserts.setdefault(mline, []).append(call)
            first, *others = sorted(ranges)
            ops.append((first[0], first[1], stubs))
            ops += [(lo, hi, "") for lo, hi in others]
        else:
            first, *others = sorted(ranges)
            ops.append((first[0], first[1], call))
            ops += [(lo, hi, "") for lo, hi in others]
    for mline, calls in marker_inserts.items():
        ops.append((mline, mline - 1, "".join(calls)))  # zero-length range = pure insert
    for lo, hi, repl in sorted(ops, key=lambda t: -t[0]):
        if hi < lo:  # pure insert before line lo
            lines[lo - 1:lo - 1] = [repl]
        else:
            lines[lo - 1:hi] = [repl] if repl else []
    text = "".join(lines)
    imp = "from types import SimpleNamespace as _SimpleNamespace\n"
    text = text.replace(imp, imp + "".join(f"from lighting_simulator.ui import {m} as _{m}\n" for m, *_ in BLOCKS), 1)
    APP.write_text(text)
    for mod, t in out_files.items():
        Path(f"lighting_simulator/ui/{mod}.py").write_text(t)
    print("written; app.py now", text.count("\n"), "lines")


if __name__ == "__main__":
    main_()
