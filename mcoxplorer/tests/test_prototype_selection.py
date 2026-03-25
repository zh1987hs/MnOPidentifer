import pandas as pd

from mcoxplorer.structure.features import choose_cluster_medoids


def test_choose_cluster_medoids():
    sim = pd.DataFrame(
        [[1.0, 0.9, 0.2], [0.9, 1.0, 0.1], [0.2, 0.1, 1.0]],
        index=["p1", "p2", "p3"],
        columns=["p1", "p2", "p3"],
    )
    cluster_df = pd.DataFrame(
        [
            {"protein_id": "p1", "structure_cluster_id": 1},
            {"protein_id": "p2", "structure_cluster_id": 1},
            {"protein_id": "p3", "structure_cluster_id": 2},
        ]
    )
    out = choose_cluster_medoids(sim, cluster_df)
    assert set(out["prototype_id"]) <= {"p1", "p2", "p3"}
