from __future__ import annotations

import pandas as pd

from mcoxplorer.io.readers import AA_STANDARD, SequenceRecord


def sequence_qc(records: list[SequenceRecord], min_len: int, max_len: int) -> tuple[list[SequenceRecord], pd.DataFrame]:
    seen: set[str] = set()
    cleaned: list[SequenceRecord] = []
    rows = []
    for rec in records:
        non_std = sorted(set(rec.sequence) - AA_STANDARD)
        is_dup = rec.sequence in seen
        seen.add(rec.sequence)
        passes = (min_len <= len(rec.sequence) <= max_len) and not non_std and not is_dup
        rows.append(
            {
                "protein_id": rec.protein_id,
                "length": len(rec.sequence),
                "non_standard_residues": "".join(non_std),
                "is_duplicate": is_dup,
                "passes_qc": passes,
            }
        )
        if passes:
            cleaned.append(rec)
    return cleaned, pd.DataFrame(rows)
