from pathlib import Path

from mcoxplorer.structure.foldseek_parser import parse_foldseek_tabular


def test_parse_foldseek_tabular(tmp_path: Path):
    f = tmp_path / "foldseek.tsv"
    f.write_text(
        "cand_1\tpos_gold_1\t1e-10\t50\t120\t0.8\t0.9\t0.35\t130\t0.78\n"
        "cand_1\tpos_gold_2\t1e-5\t40\t100\t0.7\t0.8\t0.30\t110\t0.70\n",
        encoding="utf-8",
    )
    df = parse_foldseek_tabular(f)
    assert list(df.columns)[0] == "query_id"
    assert int(df.iloc[0]["rank_within_query"]) == 1
    assert "raw_score" in df.columns
