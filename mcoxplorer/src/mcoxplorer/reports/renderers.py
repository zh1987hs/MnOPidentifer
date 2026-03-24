from __future__ import annotations

from pathlib import Path

import pandas as pd

from mcoxplorer.reports.templates import (
    agreement_interpretation,
    first_batch_reason,
    local_support_interpretation,
    risk_notes,
    sequence_evidence_summary,
    structure_global_summary,
    structure_local_summary,
)


def render_top_candidates_markdown(ranked: pd.DataFrame, top_n: int) -> str:
    lines = [
        "# Top Mn(II)-oxidizing MCO Candidate Report",
        "",
        "本报告用于实验优先级排序，不作为最终功能注释结论。",
        "",
    ]
    for _, row in ranked.head(top_n).iterrows():
        lines.extend(
            [
                f"## Candidate {row.get('candidate_id', row.get('protein_id'))}",
                f"- fused_rank: {row.get('fused_rank', 'NA')}",
                f"- sequence_only_rank: {row.get('sequence_only_rank', 'NA')}",
                f"- structure_only_rank: {row.get('structure_only_rank', 'NA')}",
                f"- nearest_positive_family: {row.get('nearest_positive_family', 'NA')}",
                f"- nearest_positive_structure_cluster: {row.get('nearest_positive_structure_cluster', 'NA')}",
                f"- sequence evidence summary: {sequence_evidence_summary(row)}",
                f"- structure global evidence summary: {structure_global_summary(row)}",
                f"- structure local evidence summary: {structure_local_summary(row)}",
                f"- local support source: {local_support_interpretation(row)}",
                f"- structure evidence usable: {row.get('structure_evidence_usable', 'NA')} (penalty={row.get('structure_quality_penalty', 'NA')})",
                f"- sequence-structure agreement: {agreement_interpretation(row)}",
                f"- first-batch recommendation: {first_batch_reason(row)}",
                f"- risk notes: {risk_notes(row)}",
                f"- final interpretation: {row.get('final_reason_summary', 'NA')}",
                "",
            ]
        )
    return "\n".join(lines)


def render_positive_cluster_report(clusters: pd.DataFrame, prototypes: pd.DataFrame) -> str:
    n_clusters = clusters["structure_cluster_id"].nunique() if not clusters.empty else 0
    lines = ["# Positive Structure Cluster Report", "", f"- Total clusters: {n_clusters}", ""]
    if clusters.empty:
        lines.append("No positive structure clusters were generated.")
        return "\n".join(lines)

    for cid, sub in clusters.groupby("structure_cluster_id"):
        fams = ", ".join(sorted(sub["family"].dropna().astype(str).unique().tolist()))
        labels = sub["label_type"].dropna().astype(str)
        gold = int((labels == "gold").sum())
        silver = int((labels == "silver").sum())
        proto = prototypes[prototypes["structure_cluster_id"] == cid]["prototype_id"].iloc[0]
        compactness = sub["cluster_compactness"].iloc[0] if "cluster_compactness" in sub.columns else "NA"
        seq_struct_disagree = sub["family"].nunique() > 1
        lines.extend(
            [
                f"## Cluster {cid}",
                f"- size: {len(sub)}",
                f"- prototype: {proto}",
                f"- families: {fams if fams else 'NA'}",
                f"- compactness: {compactness}",
                f"- dispersion: {round(1 - compactness, 3) if compactness != 'NA' else 'NA'}",
                f"- gold/silver: {gold}/{silver}",
                f"- sequence-family vs structure-cluster disagreement: {'YES' if seq_struct_disagree else 'NO'}",
                "",
            ]
        )
    return "\n".join(lines)


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
