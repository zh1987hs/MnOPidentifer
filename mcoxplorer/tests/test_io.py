from pathlib import Path

from mcoxplorer.io.qc import sequence_qc
from mcoxplorer.io.readers import read_csv, read_fasta, read_structure_dir


def test_readers_and_qc():
    base = Path("examples/input")
    pos = read_fasta(base / "positive_sequences.fasta")
    assert len(pos) == 3
    clean, qc = sequence_qc(pos, 30, 2000)
    assert len(clean) == 3
    assert qc["passes_qc"].all()

    md = read_csv(base / "positive_metadata.csv")
    assert {"protein_id", "label_type", "family"}.issubset(md.columns)

    structs = read_structure_dir(base / "structures_positive")
    assert "pos_gold_1" in structs
