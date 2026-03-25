from mcoxplorer.structure.features import compute_local_motif_sequence_support


def test_local_feature_extraction():
    feats = compute_local_motif_sequence_support("MHHDAEAAAKKDDHH")
    assert feats["motif_count"] >= 1
    assert 0 <= feats["sequence_local_support"] <= 1
