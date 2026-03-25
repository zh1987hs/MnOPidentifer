from __future__ import annotations

from pathlib import Path

import pandas as pd

# Standard fields used internally by project.
STANDARD_COLUMNS = [
    "query_id",
    "target_id",
    "evalue",
    "bits",
    "alnlen",
    "qcov",
    "tcov",
    "fident",
    "raw_score",
    "prob",
]


def parse_foldseek_tabular(result_file: str | Path) -> pd.DataFrame:
    """Parse foldseek TSV output and normalize schema.

    Expected outfmt columns:
    query,target,evalue,bits,alnlen,qcov,tcov,fident,score,prob
    """

    path = Path(result_file)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=STANDARD_COLUMNS + ["rank_within_query"])

    raw = pd.read_csv(path, sep="\t", header=None)
    if raw.shape[1] < 10:
        # tolerate shorter custom output
        for _ in range(10 - raw.shape[1]):
            raw[raw.shape[1]] = None
    raw = raw.iloc[:, :10]
    raw.columns = STANDARD_COLUMNS
    numeric_cols = ["evalue", "bits", "alnlen", "qcov", "tcov", "fident", "raw_score", "prob"]
    for c in numeric_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.sort_values(["query_id", "raw_score", "bits"], ascending=[True, False, False])
    raw["rank_within_query"] = raw.groupby("query_id").cumcount() + 1
    return raw
