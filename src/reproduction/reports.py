"""Run paper analyses and display their exported tables and figures."""

from __future__ import annotations

import contextlib
import io
import re
import runpy
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display
from matplotlib.figure import Figure


def project_root() -> Path:
    for path in (Path.cwd(), *Path.cwd().parents):
        if (path / "src/reproduction/reports.py").exists():
            return path
    raise FileNotFoundError("Run from inside the release directory")


def run_analysis(name: str, parameters: dict | None = None) -> dict:
    root = project_root()
    output = root / "outputs" / name
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    (output / "tables").mkdir(exist_ok=True)
    figures = []
    savefig = Figure.savefig

    def save(figure, destination, *args, **kwargs):
        savefig(figure, destination, *args, **kwargs)
        path = Path(destination)
        if path.suffix == ".pdf":
            path = path.with_suffix(".png")
            savefig(figure, path, dpi=140, bbox_inches="tight")
        if path.suffix == ".png" and path not in figures:
            figures.append(path)

    initial = {"PROJECT_ROOT": root, "ROOT": root, "OUTPUT_ROOT": output,
               **(parameters or {})}
    log = io.StringIO()
    with contextlib.redirect_stdout(log), patch.object(Figure, "savefig", save), \
            patch.object(plt, "show", lambda *args, **kwargs: None):
        state = runpy.run_path(str(root / "experiments/reproduction" / f"{name}.py"),
                               init_globals=initial)
    plt.close("all")
    (output / "analysis.log").write_text(log.getvalue())
    return {"state": state, "output": output, "figures": figures}


def tabular_text(path: Path) -> str:
    text = path.read_text()
    match = re.search(r"\\begin\{tabular\}[\s\S]*?\\end\{tabular\}", text)
    if match is None:
        raise ValueError(f"No tabular found in {path.name}")
    return match.group()


def _plain_cell(text: str) -> str:
    text = re.sub(r"\\rowcolor\{[^}]*\}", "", text)
    text = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{([^{}]*)\}", r"\1", text)
    for macro, label in {"dsExtremeSports": "extreme_sports", "dsEvilNumbers": "number_sequence",
                         "dsRiskyFinancial": "risky_financial", "dsSubtleMisinfo": "subtle_misinfo",
                         "dsInsecure": "insecure_code", "dsBadMedical": "bad_medical"}.items():
        text = text.replace("\\" + macro + "{}", label)
    text = text.replace(r"\|\Delta \bar{h}\|_2", "‖Δh̄‖₂")
    text = text.replace(r"\Sigma_{1..5}", "Top-5 cumulative")
    for _ in range(3):
        text = re.sub(r"\\(?:textbf|texttt|mathrm|mathbf|emph|underline)\{([^{}]*)\}", r"\1", text)
    for old, new in [(r"\%", "%"), (r"\_", "_"), (r"\pm", "±"),
                     (r"\geq", "≥"), (r"\leq", "≤"), (r"\tau", "τ"),
                     (r"\rho", "ρ"), (r"\Delta", "Δ"), (r"\dagger", "†"),
                     (r"\times", "×"), (r"\,", " "), (r"\;", " ")]:
        text = text.replace(old, new)
    return text.replace("$", "").replace(r"\ ", " ").replace("~", " ").strip()


def table_frame(path: Path) -> pd.DataFrame:
    headers, rows = [], []
    in_header = True
    for line in tabular_text(path).splitlines():
        if line.strip().startswith(r"\midrule"):
            in_header = False
        if "&" not in line or not re.search(r"\\\\\s*(?:\[[^]]*\])?\s*$", line):
            continue
        line = re.sub(r"\\\\\s*(?:\[[^]]*\])?\s*$", "", line)
        cells = []
        for cell in line.split("&"):
            span = re.search(r"\\multicolumn\{(\d+)\}", cell)
            cells.extend([_plain_cell(cell)] * (int(span.group(1)) if span else 1))
        (headers if in_header else rows).append(cells)
    if not headers or not rows:
        raise ValueError(f"No displayable rows in {path.name}")
    width = max(map(len, headers + rows))
    header = []
    for index in range(width):
        parts = list(dict.fromkeys(row[index] for row in headers if index < len(row) and row[index]))
        header.append(" / ".join(parts) or f"Column {index + 1}")
    frame = pd.DataFrame([row + [""] * (width - len(row)) for row in rows], columns=header)
    frame.iloc[:, 0] = frame.iloc[:, 0].replace("", None).ffill()
    return frame


def show_report(report: dict, tables: list[str] | None = None) -> pd.DataFrame:
    paths = list(report["output"].rglob("*.tex"))
    if tables is not None:
        paths = [path for path in paths if path.name in tables]
    checks = []
    for path in sorted(paths):
        print(path.stem.replace("tab_", "").replace("_", " "))
        with pd.option_context("display.max_colwidth", None, "display.max_columns", None,
                               "display.max_rows", None):
            display(table_frame(path))
        reference = project_root() / "review_artifacts/reference_tables" / path.name
        if reference.exists():
            generated = re.sub(r"\s+", "", tabular_text(path))
            expected = re.sub(r"\s+", "", tabular_text(reference))
            checks.append({"Table": path.name, "Matches paper artifact": generated == expected})
    for path in report["figures"]:
        print(path.stem.replace("fig_", "").replace("_", " "))
        display(Image(data=path.read_bytes()))
    result = pd.DataFrame(checks)
    if not result.empty and not result["Matches paper artifact"].all():
        failed = result.loc[~result["Matches paper artifact"], "Table"].tolist()
        raise ValueError(f"Generated tables differ from the paper artifacts: {failed}")
    return result
