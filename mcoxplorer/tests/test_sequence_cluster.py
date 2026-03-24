from mcoxplorer.io.readers import SequenceRecord
from mcoxplorer.sequence.features import cluster_sequences


def test_sequence_cluster_basic():
    recs = [
        SequenceRecord("a", "AAAAAA"),
        SequenceRecord("b", "AAAAAT"),
        SequenceRecord("c", "TTTTTT"),
    ]
    df = cluster_sequences(recs, threshold=0.8)
    assert len(df["sequence_cluster"].unique()) >= 2
