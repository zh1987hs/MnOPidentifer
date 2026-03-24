from pathlib import Path

from mcoxplorer.io.readers import SequenceRecord
from mcoxplorer.sequence import features as sf


def test_cluster_backend_fallback(tmp_path: Path):
    recs = [SequenceRecord("p1", "AAAA"), SequenceRecord("p2", "AAAT")]
    out = sf.cluster_sequences_mmseqs(recs, None, tmp_path, min_seq_id=0.4)
    assert (out["sequence_cluster_backend"] == "fallback_greedy").all()


def test_cluster_backend_mmseqs(monkeypatch, tmp_path: Path):
    recs = [SequenceRecord("p1", "AAAA"), SequenceRecord("p2", "AAAT")]

    def fake_run(cmd, cwd=None):
        out_prefix = cmd[3]
        Path(str(out_prefix) + "_cluster.tsv").write_text("p1\tp1\np1\tp2\n", encoding="utf-8")
        class R:
            stdout = ""
        return R()

    monkeypatch.setattr(sf, "run_command", fake_run)
    out = sf.cluster_sequences_mmseqs(recs, "mmseqs", tmp_path, min_seq_id=0.4)
    assert (out["sequence_cluster_backend"] == "mmseqs2").all()
