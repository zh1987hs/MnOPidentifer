from __future__ import annotations

import re
from pathlib import Path


def parse_tmalign_output(text: str) -> dict:
    """Parse core metrics from TM-align text output."""

    tm_scores = [float(x) for x in re.findall(r"TM-score=\s*([0-9]*\.?[0-9]+)", text)]
    rmsd_match = re.search(r"RMSD=\s*([0-9]*\.?[0-9]+)", text)
    aln_match = re.search(r"Aligned length=\s*(\d+)", text)
    seqid_match = re.search(r"Seq_ID=n_identical/n_aligned=\s*([0-9]*\.?[0-9]+)", text)

    return {
        "tm_score_query_norm": tm_scores[0] if len(tm_scores) > 0 else None,
        "tm_score_target_norm": tm_scores[1] if len(tm_scores) > 1 else None,
        "rmsd": float(rmsd_match.group(1)) if rmsd_match else None,
        "aligned_length": int(aln_match.group(1)) if aln_match else None,
        "sequence_identity_in_alignment": float(seqid_match.group(1)) if seqid_match else None,
    }


def parse_tmalign_file(path: str | Path) -> dict:
    return parse_tmalign_output(Path(path).read_text(encoding="utf-8", errors="ignore"))
