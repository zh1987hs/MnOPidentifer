from __future__ import annotations

from pathlib import Path

import pandas as pd

from mcoxplorer.fusion.ranking import run_fusion_ranking
from mcoxplorer.reports.generator import (
    generate_markdown_report,
    generate_positive_cluster_report,
    write_run_summary,
)
from mcoxplorer.sequence.pipeline import run_sequence_module
from mcoxplorer.structure.pipeline import run_structure_module


def run_all(cfg: dict) -> dict:
    result: dict = {}
    seq_res = run_sequence_module(cfg) if cfg["modes"].get("run_sequence", True) else {}
    result.update(seq_res)

    struct_res = (
        run_structure_module(cfg, sequence_features=seq_res.get("candidate_sequence_features", pd.DataFrame()))
        if cfg["modes"].get("run_structure", True)
        else {}
    )
    result.update(struct_res)

    ranked = pd.DataFrame()
    if cfg["modes"].get("run_fusion", True) and seq_res:
        ranked = run_fusion_ranking(
            cfg,
            seq_res["ranked_sequence"],
            struct_res.get("ranked_structure", pd.DataFrame(columns=["protein_id", "structure_score"])),
        )

    if cfg["modes"].get("run_report", True) and not ranked.empty:
        generate_markdown_report(cfg, ranked)
        if not struct_res.get("positive_structure_clusters", pd.DataFrame()).empty:
            generate_positive_cluster_report(
                cfg,
                struct_res["positive_structure_clusters"],
                struct_res.get("positive_structure_prototypes", pd.DataFrame()),
            )

    summary = {
        "n_candidates": int(len(seq_res.get("candidate_sequence_features", []))),
        "n_structured_candidates": int(
            struct_res.get("candidate_structure_features", pd.DataFrame())["best_structure_similarity_to_positive"].notna().sum()
        )
        if "candidate_structure_features" in struct_res
        else 0,
        "outputs": str(Path(cfg["output_dir"]).resolve()),
    }
    write_run_summary(Path(cfg["output_dir"]) / "run_summary.json", summary)
    return result
