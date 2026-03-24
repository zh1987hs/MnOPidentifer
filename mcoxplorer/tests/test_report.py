import pandas as pd

from mcoxplorer.reports.generator import generate_markdown_report


def test_report_generation(tmp_path):
    cfg = {"output_dir": str(tmp_path), "fusion": {"top_n_report": 2}}
    ranked = pd.DataFrame([
        {"protein_id": "cand_1", "fused_rank": 1, "final_multimodal_score": 0.9, "nearest_positive_family": "FamA", "nearest_positive_structure_cluster": 1, "sequence_score": 0.8, "structure_score": 0.7, "best_identity_to_positive": 0.2, "best_structure_similarity_to_positive": 0.7, "false_positive_risk": 0.2, "reason_for_high_rank": "x"}
    ])
    out = generate_markdown_report(cfg, ranked)
    assert out.exists()
    assert "cand_1" in out.read_text(encoding="utf-8")
