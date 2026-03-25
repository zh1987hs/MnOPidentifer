from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from Bio import SeqIO

AA_STANDARD = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class SequenceRecord:
    protein_id: str
    sequence: str


def read_fasta(path: str | Path) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    for rec in SeqIO.parse(str(path), "fasta"):
        records.append(SequenceRecord(protein_id=rec.id, sequence=str(rec.seq).upper()))
    return records


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_structure_dir(path: str | Path) -> dict[str, Path]:
    root = Path(path)
    if not root.exists():
        return {}
    out: dict[str, Path] = {}
    for f in root.iterdir():
        if f.suffix.lower() in {".pdb", ".cif", ".mmcif"}:
            out[f.stem] = f
    return out
