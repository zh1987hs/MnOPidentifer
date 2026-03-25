import pandas as pd

from mcoxplorer.structure.pipeline import _rank_structure_features


def test_quality_penalty_applied_in_ranker():
    df = pd.DataFrame(
        [
            {
                "protein_id": "a",
                "best_structure_similarity_to_any_positive": 0.8,
                "best_structure_similarity_to_gold_positive": 0.8,
                "similarity_to_nearest_cluster_prototype": 0.8,
                "mean_topk_structure_similarity": 0.8,
                "structure_family_consistency": 0.8,
                "combined_local_support": 0.8,
                "structure_gold_bias": 0.8,
                "sequence_structure_agreement_score": 0.8,
                "possible_generic_mco_risk_structure": 0.2,
                "best_identity_to_positive": 0.2,
                "structure_quality_penalty": 0.0,
            },
            {
                "protein_id": "b",
                "best_structure_similarity_to_any_positive": 0.8,
                "best_structure_similarity_to_gold_positive": 0.8,
                "similarity_to_nearest_cluster_prototype": 0.8,
                "mean_topk_structure_similarity": 0.8,
                "structure_family_consistency": 0.8,
                "combined_local_support": 0.8,
                "structure_gold_bias": 0.8,
                "sequence_structure_agreement_score": 0.8,
                "possible_generic_mco_risk_structure": 0.2,
                "best_identity_to_positive": 0.2,
                "structure_quality_penalty": 0.5,
            },
        ]
    )
    out = _rank_structure_features(df)
    assert out.set_index("protein_id").loc["a", "structure_score"] > out.set_index("protein_id").loc["b", "structure_score"]
