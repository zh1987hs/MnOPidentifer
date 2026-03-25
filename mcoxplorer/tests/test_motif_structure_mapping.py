from pathlib import Path

from mcoxplorer.structure.features import compute_local_support_3d


def test_motif_mapping_outputs_mapping_fields(tmp_path: Path):
    pdb = tmp_path / "a.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   5      11.000  12.000  13.000  1.00 70.00           C\n"
        "ATOM      2  CA  ASP A   7      12.000  13.000  14.000  1.00 65.00           C\n"
        "ATOM      3  CA  GLU A  10      13.000  14.000  15.000  1.00 60.00           C\n",
        encoding="utf-8",
    )
    out = compute_local_support_3d(pdb, "AAAAAA", [2, 4], radius=8.0)
    assert "motif_mapped_count_3d" in out
    assert "motif_mapping_confidence" in out
