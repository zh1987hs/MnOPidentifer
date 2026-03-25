from pathlib import Path

from mcoxplorer.io.readers import SequenceRecord
from mcoxplorer.sequence.features import embedding_features, hmm_scores_hmmer, similarity_search_mmseqs


def test_mmseqs_fallback_and_embedding_mock(tmp_path: Path):
    pos = [SequenceRecord("p1", "MHHHAAADDD"), SequenceRecord("p2", "MHHHAAAGGG")]
    cand = [SequenceRecord("c1", "MHHHAAADDD")]
    sim = similarity_search_mmseqs(cand, pos, mmseqs_exe=None, tmp_root=tmp_path)
    assert sim.iloc[0]["sequence_backend"] == "fallback_pairwise"

    emb = embedding_features(cand, pos, {"force_mock": True})
    assert emb.iloc[0]["embedding_backend"] == "mock"


def test_hmmer_fallback(tmp_path: Path):
    pos = [SequenceRecord("p1", "MHHHAAADDD"), SequenceRecord("p2", "MHHHAAAGGG")]
    cand = [SequenceRecord("c1", "MHHHAAADDD")]
    clusters = __import__("pandas").DataFrame([{"protein_id": "p1", "sequence_cluster": 1}, {"protein_id": "p2", "sequence_cluster": 1}])
    out = hmm_scores_hmmer(cand, pos, clusters, None, None, tmp_path)
    assert out.iloc[0]["hmm_backend"] == "fallback"
