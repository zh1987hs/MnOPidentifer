from pathlib import Path

import pandas as pd

from mcoxplorer.structure.features import parse_structure_qc


def test_structure_qc_metadata_fusion(tmp_path: Path):
    pdb = tmp_path / "x.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1      11.000  12.000  13.000  1.00 70.00           C\n"
        "ATOM      2  CA  GLY A   2      12.000  13.000  14.000  1.00 65.00           C\n",
        encoding="utf-8",
    )
    md = pd.Series({"coverage": 0.8, "model_confidence": 75})
    out = parse_structure_qc(pdb, seq_length=10, metadata_row=md, min_modeled_residue_count=1, min_structure_confidence=50, min_structure_coverage=0.5)
    assert out["structure_coverage"] == 0.8
    assert out["mean_structure_confidence"] == 75
    assert out["structure_quality_pass"] is True
