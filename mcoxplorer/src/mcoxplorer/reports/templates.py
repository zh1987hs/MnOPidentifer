from __future__ import annotations

import pandas as pd


def recommendation_sentence(row: pd.Series) -> str:
    if pd.isna(row.get("best_structure_similarity_to_positive")):
        return "该候选当前缺乏结构证据，排名主要基于序列模块结果。"
    if row.get("best_identity_to_positive", 1.0) < 0.35 and row.get("structure_score", 0.0) > 0.65:
        return "该候选在一级序列空间中与正样本距离较远，但在三维结构层面与正样本prototype/cluster显示出较高相似性，属于值得优先关注的远缘候选。"
    if row.get("sequence_score", 0.0) > 0.6 and row.get("structure_score", 0.0) > 0.6:
        return "该候选同时获得序列证据和结构证据支持，优先级较高。"
    if row.get("false_positive_risk", 0.0) > 0.65:
        return "该候选虽具有一定结构相似性，但综合证据提示其更可能属于通用MCO背景，建议谨慎纳入首轮实验。"
    return "该候选具有中等证据强度，建议结合实验可行性安排后续验证。"


def batch_recommendation(row: pd.Series) -> str:
    if row.get("fused_rank", 999) <= 10 and row.get("false_positive_risk", 1.0) < 0.5:
        return "建议列入第一批实验名单"
    if row.get("false_positive_risk", 1.0) > 0.75:
        return "建议暂缓，需更多证据"
    return "建议列入第二批实验名单"
