from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def parse_structure_qc(path: Path) -> dict:
    """Extract minimal QC metrics from PDB/mmCIF-like text.

    For AF-style PDBs, B-factor often stores per-atom confidence (pLDDT proxy).
    """

    modeled = 0
    b_factors = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    modeled += 1
                try:
                    b_factors.append(float(line[60:66].strip()))
                except ValueError:
                    pass
    avg_conf = sum(b_factors) / len(b_factors) if b_factors else None
    return {
        "protein_id": path.stem,
        "structure_path": str(path),
        "modeled_residue_count": modeled,
        "structure_coverage": 1.0 if modeled > 0 else 0.0,
        "mean_structure_confidence": avg_conf,
        "structure_quality_pass": bool(modeled > 0),
    }


def symmetrize_similarity_matrix(matrix: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Symmetrize query-target matrix.

    Foldseek/TM-align direction can differ; default uses arithmetic mean.
    """

    a = matrix.copy().astype(float)
    if strategy == "max":
        sym = np.maximum(a.values, a.values.T)
    else:
        sym = (a.values + a.values.T) / 2.0
    return pd.DataFrame(sym, index=a.index, columns=a.columns)


def hierarchical_structure_clustering(similarity: pd.DataFrame, linkage_method: str, cut_threshold: float) -> pd.DataFrame:
    """Hierarchical clustering from similarity matrix using distance = 1 - sim."""

    if len(similarity) == 1:
        pid = similarity.index[0]
        return pd.DataFrame([{"protein_id": pid, "structure_cluster_id": 1}])
    distance = 1 - similarity.clip(lower=0.0, upper=1.0)
    np.fill_diagonal(distance.values, 0.0)
    condensed = squareform(distance.values, checks=False)
    z = linkage(condensed, method=linkage_method)
    # threshold interpreted in distance space, so 1-similarity_cutoff
    labels = fcluster(z, t=cut_threshold, criterion="distance")
    return pd.DataFrame({"protein_id": similarity.index.tolist(), "structure_cluster_id": labels})


def choose_cluster_medoids(similarity: pd.DataFrame, cluster_df: pd.DataFrame) -> pd.DataFrame:
    """Choose medoid per cluster: max average intra-cluster similarity.

    Medoid is real member and robust to outliers.
    """

    rows = []
    for cid, sub in cluster_df.groupby("structure_cluster_id"):
        members = sub["protein_id"].tolist()
        medoid = max(members, key=lambda m: float(similarity.loc[m, members].mean()))
        rows.append({"structure_cluster_id": int(cid), "prototype_id": medoid, "cluster_size": len(members)})
    return pd.DataFrame(rows)


def cluster_by_similarity(ids: list[str], matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Backward-compatible simple threshold clustering."""

    sim = matrix.loc[ids, ids]
    return hierarchical_structure_clustering(similarity=sim, linkage_method="average", cut_threshold=1-threshold).rename(columns={"structure_cluster_id":"structure_cluster"})


from mcoxplorer.utils.tools import resolve_tool_path

def has_foldseek(cfg: dict) -> bool:
    return resolve_tool_path(cfg["external_tools"]["foldseek"]) is not None
