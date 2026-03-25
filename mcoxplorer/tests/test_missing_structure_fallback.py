from pathlib import Path

from mcoxplorer.structure.pipeline import run_structure_module


def test_missing_structure_dir_fallback(tmp_path):
    cfg = {
        "output_dir": str(tmp_path),
        "inputs": {
            "positive_structure_dir": str(tmp_path / "missing_pos"),
            "candidate_structure_dir": str(tmp_path / "missing_cand"),
            "positive_fasta": "examples/input/positive_sequences.fasta",
            "candidate_fasta": "examples/input/candidate_sequences.fasta",
            "positive_metadata_csv": "examples/input/positive_metadata.csv",
        },
        "structure": {"cluster_threshold": 0.5},
        "external_tools": {"foldseek": "foldseek"},
    }
    res = run_structure_module(cfg)
    assert "candidate_structure_features" in res
    assert Path(tmp_path / "candidate_structure_features.csv").exists()
