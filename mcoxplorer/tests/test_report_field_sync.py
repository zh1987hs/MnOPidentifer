import pandas as pd

from mcoxplorer.reports.templates import structure_local_summary


def test_report_uses_new_local_fields():
    row = pd.Series(
        {
            "motif_count": 2,
            "sequence_local_support": 0.6,
            "structure_local_support_3d": 0.7,
            "combined_local_support": 0.65,
        }
    )
    text = structure_local_summary(row)
    assert "sequence_local_support" in text
    assert "structure_local_support_3d" in text
    assert "combined_local_support" in text
