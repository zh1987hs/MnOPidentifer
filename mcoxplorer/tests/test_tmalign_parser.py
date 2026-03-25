from mcoxplorer.structure.tmalign_parser import parse_tmalign_output


def test_parse_tmalign_output():
    text = """
Aligned length= 150, RMSD= 2.1, Seq_ID=n_identical/n_aligned= 0.28
TM-score= 0.71 (if normalized by length of Chain_1)
TM-score= 0.68 (if normalized by length of Chain_2)
"""
    out = parse_tmalign_output(text)
    assert out["tm_score_query_norm"] == 0.71
    assert out["tm_score_target_norm"] == 0.68
    assert out["aligned_length"] == 150
