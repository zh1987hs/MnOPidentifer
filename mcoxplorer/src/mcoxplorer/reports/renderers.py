from __future__ import annotations

from pathlib import Path

import pandas as pd

from mcoxplorer.reports.templates import batch_recommendation, recommendation_sentence


def render_top_candidates_markdown(ranked: pd.DataFrame, top_n: int) -> str:
    lines = [
        "# Top Mn(II)-oxidizing MCO Candidate Report",
        "",
        "本报告用于实验优先级排序，不作为最终功能注释结论。",
        "",
    ]
    for _, row in ranked.head(top_n).iterrows():
        remote_struct = row.get("best_identity_to_positive", 1.0) < 0.35 and row.get("best_structure_similarity_to_positive", 0.0) > 0.65
        lines.extend(
            [
                f"## Candidate {row.get('candidate_id', row.get('protein_id'))}",
                f"- fused_rank: {row.get('fused_rank', 'NA')}",
                f"- sequence_only_rank: {row.get('sequence_only_rank', 'NA')}",
                f"- structure_only_rank: {row.get('structure_only_rank', 'NA')}",
                f"- nearest_positive_family: {row.get('nearest_positive_family', 'NA')}",
                f"- nearest_positive_structure_cluster: {row.get('nearest_positive_structure_cluster', 'NA')}",
                f"- best_identity_to_positive: {row.get('best_identity_to_positive', 'NA')}",
                f"- best_structure_similarity_to_positive: {row.get('best_structure_similarity_to_positive', 'NA')}",
                f"- gold-supported: {'YES' if row.get('best_structure_similarity_to_gold_positive', 0) >= 0.55 else 'NO'}",
                f"- remote-but-structurally-supported: {'YES' if remote_struct else 'NO'}",
                f"- sequence_structure_agreement_score: {row.get('sequence_structure_agreement_score', 'NA')}",
                f"- generic_mco_risk: {row.get('false_positive_risk', 'NA')}",
                f"- interpretation: {recommendation_sentence(row)}",
                f"- experimental_batch: {batch_recommendation(row)}",
                "",
            ]
        )
    return "\n".join(lines)


def render_positive_cluster_report(clusters: pd.DataFrame, prototypes: pd.DataFrame) -> str:
    n_clusters = clusters["structure_cluster_id"].nunique() if not clusters.empty else 0
    lines = [
        "# Positive Structure Cluster Report",
        "",
        f"- Total clusters: {n_clusters}",
        "",
    ]
    if clusters.empty:
        lines.append("No positive structure clusters were generated.")
        return "\n".join(lines)

    for cid, sub in clusters.groupby("structure_cluster_id"):
        fams = ", ".join(sorted(sub["family"].dropna().astype(str).unique().tolist()))
        label_mix = ", ".join(sorted(sub["label_type"].dropna().astype(str).unique().tolist()))
        proto = prototypes[prototypes["structure_cluster_id"] == cid]["prototype_id"].iloc[0]
        lines.extend(
            [
                f"## Cluster {cid}",
                f"- size: {len(sub)}",
                f"- prototype: {proto}",
                f"- families: {fams if fams else 'NA'}",
                f"- label types: {label_mix if label_mix else 'NA'}",
                f"- mixed gold/silver: {'YES' if {'gold', 'silver'}.issubset(set(sub['label_type'].dropna())) else 'NO'}",
                "",
            ]
        )
    return "\n".join(lines)


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
