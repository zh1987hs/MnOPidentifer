from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mcoxplorer.reports.renderers import (
    render_positive_cluster_report,
    render_top_candidates_markdown,
    write_report,
)


def write_run_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def generate_markdown_report(cfg: dict, ranked: pd.DataFrame) -> Path:
    report_dir = Path(cfg["output_dir"]) / "reports"
    top_n = int(cfg["fusion"].get("top_n_report", 20))
    content = render_top_candidates_markdown(ranked, top_n)
    out = report_dir / "top_candidates_report.md"
    write_report(out, content)
    return out


def generate_positive_cluster_report(cfg: dict, clusters: pd.DataFrame, prototypes: pd.DataFrame) -> Path:
    report_dir = Path(cfg["output_dir"]) / "reports"
    content = render_positive_cluster_report(clusters, prototypes)
    out = report_dir / "positive_structure_cluster_report.md"
    write_report(out, content)
    return out
