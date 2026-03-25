import pandas as pd

from mcoxplorer.fusion.ranker import run_fusion


def test_fusion_ranking(tmp_path):
    cfg = {"output_dir": str(tmp_path), "fusion": {"weights": {"sequence_score": 0.5, "structure_score": 0.4, "novelty_bonus": 0.1, "false_positive_risk_penalty": 0.2}}}
    seq = pd.DataFrame([
        {"protein_id": "a", "sequence_score": 0.9, "best_identity_to_positive": 0.7, "novelty": 0.1, "nearest_positive_family": "X"},
        {"protein_id": "b", "sequence_score": 0.6, "best_identity_to_positive": 0.3, "novelty": 0.8, "nearest_positive_family": "Y"},
    ])
    struct = pd.DataFrame([
        {"protein_id": "a", "structure_score": 0.3, "best_structure_similarity_to_positive": 0.3, "false_positive_risk_structure": 0.6},
        {"protein_id": "b", "structure_score": 0.8, "best_structure_similarity_to_positive": 0.7, "false_positive_risk_structure": 0.2},
    ])
    ranked = run_fusion(cfg, seq, struct)
    assert ranked.iloc[0]["protein_id"] == "b"
