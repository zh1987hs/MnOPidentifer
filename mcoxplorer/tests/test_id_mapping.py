from pathlib import Path

from mcoxplorer.io.idmap import IdMapper


def test_id_mapping_resolution(tmp_path: Path):
    f = tmp_path / "map.csv"
    f.write_text("fasta_id,metadata_id,structure_id\na,b,c\n", encoding="utf-8")
    m = IdMapper(f)
    assert m.canonical("a") == "a"
    assert m.canonical("b") == "a"
    assert m.canonical("c") == "a"
