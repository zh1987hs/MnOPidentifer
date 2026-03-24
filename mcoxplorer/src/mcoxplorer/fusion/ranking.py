from __future__ import annotations

from pathlib import Path

import pandas as pd

from mcoxplorer.fusion.scoring import compute_final_multimodal_score, derive_false_positive_risk


def run_fusion_ranking(cfg: dict, sequence_df: pd.DataFrame, structure_df: pd.DataFrame) -> pd.DataFrame:
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = sequence_df.merge(structure_df, on="protein_id", how="left", suffixes=("", "_structure"))
    merged["candidate_id"] = merged["protein_id"]

    merged["structure_score"] = merged.get("structure_score", 0.0).fillna(0.0)
    merged["best_structure_similarity_to_positive"] = merged.get("best_structure_similarity_to_positive", 0.0)
    merged["structure_novelty_score"] = merged.get("structure_novelty_score", 0.0).fillna(0.0)
    merged["novelty"] = merged.get("novelty", 0.0).fillna(0.0)
    merged["nearest_positive_structure_cluster"] = merged.get("nearest_positive_structure_cluster", "NA").fillna("NA")

    merged["sequence_structure_agreement_score"] = merged.get("sequence_structure_agreement_score", 0.5).fillna(0.5)
    merged["false_positive_risk"] = derive_false_positive_risk(merged)
    merged["remote_but_structure_supported"] = (
        merged.get("best_identity_to_positive", pd.Series([1.0] * len(merged))).fillna(1.0) < 0.35
    ) & (merged.get("best_structure_similarity_to_positive", pd.Series([0.0] * len(merged))).fillna(0.0) >= 0.65)

    merged["final_multimodal_score"] = compute_final_multimodal_score(merged, cfg["fusion"]["weights"])
    merged["high_confidence_first_batch"] = (
        (merged["final_multimodal_score"] >= 0.6)
        & (merged["false_positive_risk"] < 0.5)
    )
    merged["final_reason_summary"] = merged.apply(_final_reason, axis=1)

    merged = merged.sort_values("final_multimodal_score", ascending=False).reset_index(drop=True)
    merged["fused_rank"] = merged.index + 1

    output_cols = [
        "candidate_id",
        "sequence_only_rank",
        "structure_only_rank",
        "fused_rank",
        "sequence_score",
        "structure_score",
        "final_multimodal_score",
        "nearest_positive_family",
        "nearest_positive_structure_cluster",
        "best_identity_to_positive",
        "best_structure_similarity_to_positive",
        "sequence_structure_agreement_score",
        "structure_novelty_score",
        "false_positive_risk",
        "remote_but_structure_supported",
        "high_confidence_first_batch",
        "structure_reason_summary",
        "final_reason_summary",
    ]
    for c in output_cols:
        if c not in merged.columns:
            merged[c] = pd.NA
    merged[output_cols].to_csv(out_dir / "ranked_candidates_multimodal_view.csv", index=False)
    return merged


def _final_reason(row: pd.Series) -> str:
    if pd.isna(row.get("best_structure_similarity_to_positive")):
        return "缺乏结构证据，当前排名主要由序列模块驱动。"
    if bool(row.get("remote_but_structure_supported", False)):
        return "序列远缘但结构强支持，建议优先进入第一批验证。"
    if row.get("sequence_score", 0) > 0.6 and row.get("structure_score", 0) > 0.6:
        return "序列与结构证据一致且强，优先级高。"
    if row.get("false_positive_risk", 0) > 0.65:
        return "存在较高通用MCO背景风险，建议谨慎。"
    return "证据中等，建议作为第二批或补充证据后推进。"
