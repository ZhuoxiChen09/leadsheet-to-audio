"""Execute workbook.ipynb in-process and write outputs back into the .ipynb.

We don't use jupyter kernel because the kernel subprocess on this machine has
a torch DLL load failure (Windows DLL search path issue). Running cells
inline in the current Python process avoids the subprocess entirely while
still producing inline outputs (text + matplotlib + audio + markdown).

Usage:
    python scripts/execute_notebook.py [--ipynb workbook.ipynb] [--html workbook.html]
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _capture_mpl_figures() -> list[dict]:
    """Capture all currently-open matplotlib figures as PNG outputs."""
    out = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        out.append(
            nbf.v4.new_output(
                "display_data",
                data={"image/png": base64.b64encode(buf.read()).decode("ascii"), "text/plain": "<matplotlib Figure>"},
                metadata={},
            )
        )
    plt.close("all")
    return out


def _capture_audio(ns: dict) -> list[dict]:
    """If the user's namespace produced an IPython display object, render it as HTML/PNG/text."""
    out = []
    audio_buf = ns.get("_audio_capture_buffer", None)
    if not audio_buf:
        return out
    items = list(audio_buf)
    audio_buf.clear()
    for item in items:
        data: dict[str, Any] = {}
        if hasattr(item, "_repr_html_"):
            try:
                html = item._repr_html_()
                if html:
                    data["text/html"] = html
            except Exception:
                pass
        if hasattr(item, "_repr_png_"):
            try:
                png = item._repr_png_()
                if png:
                    data["image/png"] = base64.b64encode(png).decode("ascii") if isinstance(png, bytes) else png
            except Exception:
                pass
        if not data:
            data["text/plain"] = repr(item)
        out.append(nbf.v4.new_output("display_data", data=data, metadata={}))
    return out


def execute_notebook(ipynb_path: Path) -> nbf.NotebookNode:
    nb = nbf.read(str(ipynb_path), as_version=4)
    ns: dict[str, Any] = {"__name__": "__main__"}

    # Use the real IPython.display but capture display() calls in our own buffer
    # so we can embed them in the right cell's outputs.
    import IPython.display as _ipy_display
    audio_buffer: list = []
    _original_display = _ipy_display.display

    def _capturing_display(*objs, **kwargs):
        for o in objs:
            audio_buffer.append(o)

    _ipy_display.display = _capturing_display
    ns["_audio_capture_buffer"] = audio_buffer
    # Ensure matplotlib stays on Agg (no GUI backend)
    import matplotlib as _mpl
    _mpl.use("Agg")

    print(f"Executing {len(nb.cells)} cells...")
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        src = cell.source
        if not src.strip():
            cell.outputs = []
            cell.execution_count = None
            continue
        print(f"  [cell {i}] {src.splitlines()[0][:80]}")
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        outputs: list = []
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                # Strip jupyter magics (we run plain python)
                src_clean = "\n".join(
                    line for line in src.splitlines() if not line.strip().startswith("%")
                )
                # Try: exec first n-1 lines as statements, eval last expression for repr-style output
                lines = src_clean.split("\n")
                # Compile entire body as exec
                compiled = compile(src_clean + "\n", f"<cell {i}>", "exec")
                exec(compiled, ns)
                # Try to evaluate the last logical "statement" as expression for display
                try:
                    last_expr_src = lines[-1] if lines else ""
                    if last_expr_src and not last_expr_src.startswith((" ", "\t")) and not last_expr_src.startswith(("return", "raise", "pass", "import", "from", "def ", "class ", "if ", "for ", "while ", "with ", "try", "else", "elif", "except", "finally", "@")):
                        expr_val = eval(compile(last_expr_src, f"<cell {i} last>", "eval"), ns)
                        if expr_val is not None:
                            text = repr(expr_val)
                            outputs.append(
                                nbf.v4.new_output(
                                    "execute_result",
                                    data={"text/plain": text[:200000]},
                                    metadata={},
                                    execution_count=i + 1,
                                )
                            )
                            # For pandas dataframes: also include HTML
                            if hasattr(expr_val, "to_html"):
                                try:
                                    outputs[-1]["data"]["text/html"] = expr_val.to_html()
                                except Exception:
                                    pass
                except SyntaxError:
                    pass
                except Exception:
                    pass
        except Exception:
            tb = traceback.format_exc()
            outputs.append(
                nbf.v4.new_output("error", ename="Exception", evalue="see traceback", traceback=tb.splitlines())
            )
        out_txt = stdout_buf.getvalue()
        err_txt = stderr_buf.getvalue()
        if out_txt:
            outputs.append(nbf.v4.new_output("stream", name="stdout", text=out_txt))
        if err_txt:
            outputs.append(nbf.v4.new_output("stream", name="stderr", text=err_txt))
        outputs += _capture_mpl_figures()
        outputs += _capture_audio(ns)
        cell.outputs = outputs
        cell.execution_count = i + 1
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipynb", default="workbook.ipynb")
    ap.add_argument("--out", default="workbook.ipynb")
    args = ap.parse_args()
    nb = execute_notebook(Path(args.ipynb))
    nbf.write(nb, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
