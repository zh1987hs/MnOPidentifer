import pandas as pd

from mcoxplorer.structure.features import cluster_by_similarity


def test_structure_cluster_basic():
    ids = ["p1", "p2", "p3"]
    mat = pd.DataFrame(
        [[1, 0.9, 0.2], [0.9, 1, 0.1], [0.2, 0.1, 1]], index=ids, columns=ids
    )
    out = cluster_by_similarity(ids, mat, 0.5)
    c1 = out.set_index("protein_id").loc["p1", "structure_cluster"]
    c2 = out.set_index("protein_id").loc["p2", "structure_cluster"]
    c3 = out.set_index("protein_id").loc["p3", "structure_cluster"]
    assert c1 == c2
    assert c3 != c1
