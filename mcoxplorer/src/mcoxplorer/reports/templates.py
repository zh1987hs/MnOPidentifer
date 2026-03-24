from __future__ import annotations

import pandas as pd


def sequence_evidence_summary(row: pd.Series) -> str:
    score = row.get("sequence_score", 0.0)
    ident = row.get("best_identity_to_positive", 0.0)
    return f"sequence_score={score:.3f}; best_identity={ident:.3f}"


def structure_global_summary(row: pd.Series) -> str:
    return (
        f"best_structure_similarity={row.get('best_structure_similarity_to_positive', 'NA')}; "
        f"prototype_similarity={row.get('similarity_to_nearest_cluster_prototype', 'NA')}; "
        f"topk_mean={row.get('mean_topk_structure_similarity', 'NA')}"
    )


def structure_local_summary(row: pd.Series) -> str:
    return (
        f"motif_count={row.get('motif_count', 'NA')}; acidic_density={row.get('local_acidic_density', 'NA')}; "
        f"local_support={row.get('local_structural_support', 'NA')}"
    )


def agreement_interpretation(row: pd.Series) -> str:
    if bool(row.get("remote_but_structure_supported", False)):
        return "序列远缘但结构支持强。"
    ssa = row.get("sequence_structure_agreement_score", 0.5)
    if ssa >= 0.7:
        return "序列与结构证据一致。"
    if ssa <= 0.3:
        return "序列与结构证据存在冲突。"
    return "序列与结构证据部分一致。"


def first_batch_reason(row: pd.Series) -> str:
    if bool(row.get("high_confidence_first_batch", False)):
        return "结构与序列综合得分高且风险可控，建议第一批验证。"
    return "建议后续批次或补充证据后验证。"


def risk_notes(row: pd.Series) -> str:
    if row.get("false_positive_risk", 0.0) > 0.65:
        return "generic MCO 假阳性风险偏高。"
    if pd.isna(row.get("best_structure_similarity_to_positive")):
        return "缺乏结构证据。"
    return "未见显著结构风险信号。"
