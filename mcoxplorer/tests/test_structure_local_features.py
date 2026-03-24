from mcoxplorer.structure.features import compute_local_metal_features


def test_local_feature_extraction():
    feats = compute_local_metal_features("MHHDAEAAAKKDDHH")
    assert feats["motif_count"] >= 1
    assert 0 <= feats["local_structural_support"] <= 1
