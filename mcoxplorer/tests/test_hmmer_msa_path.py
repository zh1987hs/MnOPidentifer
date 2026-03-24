from pathlib import Path

import pandas as pd

from mcoxplorer.io.readers import SequenceRecord
from mcoxplorer.sequence import features as sf


def test_hmmer_uses_msa_step(monkeypatch, tmp_path: Path):
    called = {"mafft": 0, "hmmbuild": 0, "hmmsearch": 0}

    def fake_run(cmd, cwd=None):
        c0 = Path(cmd[0]).name
        if c0 == "mafft":
            called["mafft"] += 1
            class R: stdout = ">a\nAAAA\n>b\nAAAA\n"
            return R()
        if c0 == "hmmbuild":
            called["hmmbuild"] += 1
            Path(cmd[1]).write_text("HMM", encoding="utf-8")
            class R: stdout = ""
            return R()
        if c0 == "hmmsearch":
            called["hmmsearch"] += 1
            out_tbl = Path(cmd[2])
            out_tbl.write_text("cand1 - - - - 20\n", encoding="utf-8")
            class R: stdout = ""
            return R()
        class R: stdout = ""
        return R()

    monkeypatch.setattr(sf, "run_command", fake_run)
    pos = [SequenceRecord("p1", "AAAA"), SequenceRecord("p2", "AAAA")]
    cand = [SequenceRecord("cand1", "AAAA")]
    clusters = pd.DataFrame([{"protein_id": "p1", "sequence_cluster": 1}, {"protein_id": "p2", "sequence_cluster": 1}])
    tools = {"hmmbuild": "hmmbuild", "hmmsearch": "hmmsearch", "mafft": "mafft", "muscle": None}
    out = sf.hmm_scores_hmmer(cand, pos, clusters, tools, tmp_path)
    assert called["mafft"] > 0
    assert called["hmmbuild"] > 0
    assert called["hmmsearch"] > 0
    assert out.iloc[0]["hmm_backend"].startswith("hmmer+")
