from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from mcoxplorer.structure.tmalign_parser import parse_tmalign_output
from mcoxplorer.utils.subprocess import run_command
from mcoxplorer.utils.tools import resolve_tool_path

LOGGER = logging.getLogger(__name__)


def run_tmalign(query_pdb: str | Path, target_pdb: str | Path, tmalign_exe: str, raw_out_path: str | Path | None = None) -> dict:
    """Run TM-align for one pair and return parsed metrics."""

    result = run_command([tmalign_exe, str(query_pdb), str(target_pdb)])
    parsed = parse_tmalign_output(result.stdout)
    parsed["query_id"] = Path(query_pdb).stem
    parsed["target_id"] = Path(target_pdb).stem
    if raw_out_path:
        raw_out = Path(raw_out_path)
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        raw_out.write_text(result.stdout, encoding="utf-8")
        parsed["tmalign_raw_text_path"] = str(raw_out)
    else:
        parsed["tmalign_raw_text_path"] = None
    return parsed


def refine_top_hits_with_tmalign(
    foldseek_hits_df: pd.DataFrame,
    top_k_per_query: int,
    structure_lookup: dict[str, Path],
    tmalign_config: str,
    raw_text_dir: str | Path,
) -> pd.DataFrame:
    """Refine top Foldseek hits per query with TM-align metrics."""

    tmalign_exe = resolve_tool_path(tmalign_config)
    if not tmalign_exe:
        LOGGER.warning("TM-align not available; skip refine.")
        return pd.DataFrame()

    raw_text_dir = Path(raw_text_dir)
    rows = []
    for qid, group in foldseek_hits_df.groupby("query_id"):
        for _, row in group.nsmallest(top_k_per_query, "rank_within_query").iterrows():
            tid = row["target_id"]
            if qid not in structure_lookup or tid not in structure_lookup:
                continue
            out_file = raw_text_dir / f"{qid}__{tid}.tmalign.txt"
            refined = run_tmalign(structure_lookup[qid], structure_lookup[tid], tmalign_exe, out_file)
            rows.append(refined)
    return pd.DataFrame(rows)
