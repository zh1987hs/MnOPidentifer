import pandas as pd

from mcoxplorer.reports.renderers import render_top_candidates_markdown


def test_report_template_generation():
    df = pd.DataFrame(
        [
            {
                "candidate_id": "cand_1",
                "fused_rank": 1,
                "sequence_only_rank": 2,
                "structure_only_rank": 1,
                "nearest_positive_family": "FamA",
                "nearest_positive_structure_cluster": 1,
                "best_identity_to_positive": 0.2,
                "best_structure_similarity_to_positive": 0.8,
                "best_structure_similarity_to_gold_positive": 0.8,
                "sequence_structure_agreement_score": 0.7,
                "false_positive_risk": 0.2,
                "sequence_score": 0.7,
                "structure_score": 0.75,
            }
        ]
    )
    text = render_top_candidates_markdown(df, top_n=1)
    assert "remote-but-structurally-supported" in text
    assert "cand_1" in text
