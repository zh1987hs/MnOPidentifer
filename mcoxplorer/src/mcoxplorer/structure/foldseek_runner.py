from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from mcoxplorer.structure.foldseek_parser import parse_foldseek_tabular
from mcoxplorer.utils.subprocess import run_command
from mcoxplorer.utils.tools import resolve_tool_path

LOGGER = logging.getLogger(__name__)


@dataclass
class FoldseekRuntime:
    executable: str | None
    available: bool


def get_foldseek_runtime(config_value: str) -> FoldseekRuntime:
    path = resolve_tool_path(config_value)
    return FoldseekRuntime(executable=path, available=path is not None)


def build_structure_db(
    input_dir: str | Path,
    db_path: str | Path,
    foldseek_exe: str,
    force: bool = False,
) -> Path:
    """Build reusable foldseek DB for structure directory."""

    input_dir = Path(input_dir)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists() and not force:
        LOGGER.info("Reuse existing Foldseek DB: %s", db_path)
        return db_path
    run_command([foldseek_exe, "createdb", str(input_dir), str(db_path)])
    return db_path


def search_candidates_against_positive_db(
    candidate_dir: str | Path,
    positive_db: str | Path,
    output_path: str | Path,
    tmp_dir: str | Path,
    foldseek_exe: str,
    params: dict | None = None,
    force: bool = False,
):
    """Run foldseek easy-search and parse normalized results."""

    params = params or {}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        LOGGER.info("Reuse Foldseek search output: %s", output_path)
        return parse_foldseek_tabular(output_path)

    sensitivity = str(params.get("sensitivity", 7.5))
    max_seqs = str(params.get("max_seqs", 200))
    outfmt = "query,target,evalue,bits,alnlen,qcov,tcov,fident,score,prob"
    run_command(
        [
            foldseek_exe,
            "easy-search",
            str(candidate_dir),
            str(positive_db),
            str(output_path),
            str(tmp_dir),
            "-s",
            sensitivity,
            "--max-seqs",
            max_seqs,
            "--format-output",
            outfmt,
        ]
    )
    return parse_foldseek_tabular(output_path)


def search_self_against_db(
    query_dir: str | Path,
    db_path: str | Path,
    output_path: str | Path,
    tmp_dir: str | Path,
    foldseek_exe: str,
    force: bool = False,
):
    return search_candidates_against_positive_db(
        candidate_dir=query_dir,
        positive_db=db_path,
        output_path=output_path,
        tmp_dir=tmp_dir,
        foldseek_exe=foldseek_exe,
        params={"sensitivity": 7.5, "max_seqs": 1000},
        force=force,
    )
