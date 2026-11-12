"""The agentic loop: generate -> render -> gate -> feed failures back -> repeat."""
from __future__ import annotations

from pathlib import Path

from .codegen import generate_scad
from .llm import ollama_generate
from .render import find_openscad, render_png, render_views, make_contact_sheet, export_stl


def run(description: str, *, name: str = "part", model: str = "qwen2.5-coder:32b",
        iters: int = 3, inspector=None, out_dir: str | Path = "out",
        llm=ollama_generate) -> dict:
    """Generate an OpenSCAD part from `description`, iterating up to `iters` times.

    Each iteration checks STL export as the deterministic compilation gate.
    On a headless host, a PNG preview may fail although STL export succeeds;
    this is recorded as preview_check="not run", never as a visual pass.
    The optional inspector only runs when at least one image was rendered.
    A failed STL export or a rejected visual inspection feeds back to codegen.
    STL export alone does not certify dimensions or physical printability.

    The inspector's `approved` may be True, False, or None. None means the vision
    check did not run (for example the vision model errored). That is not a pass:
    the part is accepted on the compile and STL-export gates only, and the result's
    "vision_check" field says "not run".

    Returns a dict with paths (scad/png/stl), attempt count, per-attempt history and
    "vision_check": "passed", "not run", or "off" (no inspector given).
    With no inspector, the first compile-and-exportable part wins.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    openscad = find_openscad()
    if not openscad:
        return {"error": "OpenSCAD binary not found (set $TEXTCAD_OPENSCAD, "
                         "bundle tools/openscad-*/, or install it)."}

    scad = out / f"{name}.scad"
    png = out / f"{name}.png"
    stl = out / f"{name}.stl"
    code, error, critique = None, None, None
    history: list[dict] = []

    for attempt in range(1, iters + 1):
        code = generate_scad(description, model=model, prior_code=code,
                             error=error, critique=critique, llm=llm)
        scad.write_text(code, encoding="utf-8")
        critique = None

        preview_ok, preview_err = render_png(openscad, scad, png)
        stl_ok, stl_err = export_stl(openscad, scad, stl)
        if not preview_ok and not stl_ok:
            error = f"Preview failed: {preview_err}; STL export failed: {stl_err}"
            history.append({"attempt": attempt, "compiled": False,
                            "stl": False, "feedback": error[:200]})
            print(f"[attempt {attempt}] compiled=False  error={error[:140]!r}")
            continue

        if not stl_ok:
            error = ("The code rendered a preview but STL export FAILED. This usually "
                     "means the geometry mixes 2D and 3D objects (e.g. a bare polygon()/"
                     "circle() not wrapped in linear_extrude alongside 3D solids), or the "
                     "result is empty/non-manifold. Make the whole model a single 3D solid. "
                     f"Exporter said: {stl_err}")
            history.append({"attempt": attempt, "compiled": True, "stl": False,
                            "feedback": error[:200]})
            print(f"[attempt {attempt}] compiled=True but STL export FAILED, feeding back")
            continue

        if not preview_ok:
            print(f"[attempt {attempt}] PNG preview unavailable; STL export valid")

        if inspector is not None:
            views = render_views(openscad, scad, out / name)  # top + front + side
            sheet = make_contact_sheet(views, out / f"{name}_sheet.png") if views else None
            if sheet:
                viewset = [("orthographic contact sheet (top-down, front, right-side panels)", sheet)]
            elif views:
                viewset = views
            elif preview_ok:
                viewset = [("iso", png)]
            else:
                viewset = []
            if viewset:
                approved, crit = inspector(viewset, description)
            else:
                approved, crit = None, "render_unavailable"
            crit = crit or ""
            history.append({"attempt": attempt, "compiled": True, "stl": True,
                            "approved": approved,
                            "feedback": "" if approved is True else crit[:200]})
            if approved is None:
                # The vision check did not run. Accept on compile + export only and
                # say so; retrying codegen would not fix a vision-model failure.
                vision_check = "not run"
                print(f"[attempt {attempt}] compiled=True stl=True  VISION CHECK DID NOT RUN"
                      f" ({crit[:120]!r}); accepting on compile + STL export only")
            elif approved:
                vision_check = "passed"
                print(f"[attempt {attempt}] compiled=True stl=True  inspector_approved=True")
            else:
                print(f"[attempt {attempt}] compiled=True stl=True  inspector_approved=False"
                      f"  critique={crit[:120]!r}")
                critique = crit
                continue
        else:
            vision_check = "off"
            history.append({"attempt": attempt, "compiled": True, "stl": True, "approved": None})
            print(f"[attempt {attempt}] compiled=True stl=True (no inspector, accepting)")

        return {"openscad": openscad, "description": description, "scad": str(scad),
                "png": str(png) if preview_ok else None, "stl": str(stl),
                "attempts": attempt, "history": history,
                "preview_check": "passed" if preview_ok else "not run",
                "vision_check": vision_check}

    return {"openscad": openscad, "description": description, "scad": str(scad),
            "png": str(png), "stl": None, "attempts": iters, "history": history,
            "error": "did not converge within iteration budget"}
