from mcoxplorer.structure.features import has_foldseek


def test_foldseek_missing_graceful():
    cfg = {"external_tools": {"foldseek": "definitely_not_installed_foldseek"}}
    assert has_foldseek(cfg) is False
