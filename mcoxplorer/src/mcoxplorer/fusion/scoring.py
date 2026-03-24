from __future__ import annotations

import pandas as pd


def compute_final_multimodal_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Compute transparent weighted fusion score.

    final = w_seq*sequence_score + w_struct*structure_score + w_novelty*novelty - w_risk*false_positive_risk
    """

    return (
        weights["sequence_score"] * df["sequence_score"].fillna(0.0)
        + weights["structure_score"] * df["structure_score"].fillna(0.0)
        + weights["novelty_bonus"] * df["novelty"].fillna(0.0)
        - weights["false_positive_risk_penalty"] * df["false_positive_risk"].fillna(0.5)
    )


def derive_false_positive_risk(df: pd.DataFrame) -> pd.Series:
    return (
        0.6 * df.get("possible_generic_mco_risk_structure", pd.Series([0.5] * len(df))).fillna(0.5)
        + 0.4 * (1 - df.get("sequence_structure_agreement_score", pd.Series([0.5] * len(df))).fillna(0.5))
    ).clip(lower=0.0, upper=1.0)
