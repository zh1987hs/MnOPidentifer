from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


class IdMapper:
    """Normalize IDs across fasta/metadata/structure spaces."""

    def __init__(self, mapping_csv: str | Path | None):
        self.df = pd.DataFrame(columns=["fasta_id", "metadata_id", "structure_id", "canonical_id"])
        if mapping_csv and Path(mapping_csv).exists():
            raw = pd.read_csv(mapping_csv)
            for col in ["fasta_id", "metadata_id", "structure_id"]:
                if col not in raw.columns:
                    raw[col] = pd.NA
            raw["canonical_id"] = raw["fasta_id"].fillna(raw["metadata_id"]).fillna(raw["structure_id"])
            self.df = raw[["fasta_id", "metadata_id", "structure_id", "canonical_id"]].dropna(subset=["canonical_id"])

    def canonical(self, value: str) -> str:
        if self.df.empty:
            return value
        v = str(value)
        matches = self.df[
            (self.df["fasta_id"] == v)
            | (self.df["metadata_id"] == v)
            | (self.df["structure_id"] == v)
            | (self.df["canonical_id"] == v)
        ]
        if matches.empty:
            LOGGER.warning("ID not found in mapping, keep raw id: %s", value)
            return v
        return str(matches.iloc[0]["canonical_id"])

    def apply_to_dataframe(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        out = df.copy()
        out[col] = out[col].astype(str).map(self.canonical)
        return out

    def export_resolved(self, output_path: str | Path) -> None:
        if self.df.empty:
            pd.DataFrame(columns=["fasta_id", "metadata_id", "structure_id", "canonical_id"]).to_csv(output_path, index=False)
        else:
            self.df.to_csv(output_path, index=False)
