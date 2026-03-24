from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, PDBParser
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def _extract_modeled_residues_and_confidence(path: Path) -> tuple[int, float | None]:
    """Parse PDB/mmCIF to estimate modeled residues and mean confidence."""

    ext = path.suffix.lower()
    structure = None
    try:
        if ext in {".cif", ".mmcif"}:
            structure = MMCIFParser(QUIET=True).get_structure(path.stem, str(path))
        else:
            structure = PDBParser(QUIET=True).get_structure(path.stem, str(path))
    except Exception:
        # fallback to line-based parser
        modeled = 0
        b_factors = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    modeled += 1
                try:
                    b_factors.append(float(line[60:66].strip()))
                except ValueError:
                    pass
        return modeled, (sum(b_factors) / len(b_factors) if b_factors else None)

    residues = []
    bvals = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " ":
                    residues.append((chain.id, residue.id[1]))
                    for atom in residue:
                        bvals.append(float(atom.get_bfactor()))
    modeled_count = len(set(residues))
    mean_conf = sum(bvals) / len(bvals) if bvals else None
    return modeled_count, mean_conf


def parse_structure_qc(
    path: Path,
    seq_length: int | None = None,
    metadata_row: pd.Series | None = None,
    min_modeled_residue_count: int = 1,
    min_structure_confidence: float = 0.0,
    min_structure_coverage: float = 0.0,
) -> dict:
    """Extract structure QC metrics with metadata fusion and thresholding."""

    modeled_count, parsed_conf = _extract_modeled_residues_and_confidence(path)
    parsed_cov = (modeled_count / seq_length) if (seq_length and seq_length > 0) else None

    meta_cov = None
    meta_conf = None
    if metadata_row is not None and not metadata_row.empty:
        meta_cov = pd.to_numeric(metadata_row.get("coverage"), errors="coerce")
        meta_conf = pd.to_numeric(metadata_row.get("model_confidence"), errors="coerce")

    coverage = float(meta_cov) if pd.notna(meta_cov) else (float(parsed_cov) if parsed_cov is not None else np.nan)
    confidence = float(meta_conf) if pd.notna(meta_conf) else (float(parsed_conf) if parsed_conf is not None else np.nan)

    quality_pass = (
        modeled_count >= min_modeled_residue_count
        and (pd.isna(confidence) or confidence >= min_structure_confidence)
        and (pd.isna(coverage) or coverage >= min_structure_coverage)
    )

    return {
        "protein_id": path.stem,
        "structure_path": str(path),
        "modeled_residue_count": modeled_count,
        "sequence_length": seq_length,
        "structure_coverage": coverage,
        "mean_structure_confidence": confidence,
        "structure_quality_pass": bool(quality_pass),
    }


def compute_local_metal_features(sequence: str) -> dict[str, float]:
    """Simple local motif features for Mn-binding propensity heuristics."""

    seq = sequence.upper()
    motif_positions = []
    for i in range(len(seq) - 2):
        tri = seq[i : i + 3]
        if tri[0] in {"H", "D", "E"} and tri[2] in {"H", "D", "E"}:
            motif_positions.append(i + 1)

    windows = []
    for pos in motif_positions:
        l = max(0, pos - 5)
        r = min(len(seq), pos + 6)
        windows.append(seq[l:r])
    joined = "".join(windows) if windows else seq

    acidic = (joined.count("D") + joined.count("E")) / max(len(joined), 1)
    basic = (joined.count("K") + joined.count("R") + joined.count("H")) / max(len(joined), 1)
    polar = sum(joined.count(x) for x in "STNQ") / max(len(joined), 1)
    local_support = min(1.0, 0.6 * acidic + 0.2 * basic + 0.2 * polar + 0.1 * len(motif_positions))

    return {
        "motif_count": float(len(motif_positions)),
        "local_acidic_density": acidic,
        "local_basic_density": basic,
        "local_polar_density": polar,
        "local_structural_support": local_support,
    }


def symmetrize_similarity_matrix(matrix: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    a = matrix.copy().astype(float)
    if strategy == "max":
        sym = np.maximum(a.values, a.values.T)
    else:
        sym = (a.values + a.values.T) / 2.0
    return pd.DataFrame(sym, index=a.index, columns=a.columns)


def hierarchical_structure_clustering(similarity: pd.DataFrame, linkage_method: str, cut_threshold: float) -> pd.DataFrame:
    if len(similarity) == 1:
        pid = similarity.index[0]
        return pd.DataFrame([{"protein_id": pid, "structure_cluster_id": 1}])
    distance = 1 - similarity.clip(lower=0.0, upper=1.0)
    np.fill_diagonal(distance.values, 0.0)
    condensed = squareform(distance.values, checks=False)
    z = linkage(condensed, method=linkage_method)
    labels = fcluster(z, t=cut_threshold, criterion="distance")
    return pd.DataFrame({"protein_id": similarity.index.tolist(), "structure_cluster_id": labels})


def choose_cluster_medoids(similarity: pd.DataFrame, cluster_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, sub in cluster_df.groupby("structure_cluster_id"):
        members = sub["protein_id"].tolist()
        medoid = max(members, key=lambda m: float(similarity.loc[m, members].mean()))
        rows.append({"structure_cluster_id": int(cid), "prototype_id": medoid, "cluster_size": len(members), "cluster_compactness": float(similarity.loc[members, members].mean())})
    return pd.DataFrame(rows)


def cluster_by_similarity(ids: list[str], matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    sim = matrix.loc[ids, ids]
    return hierarchical_structure_clustering(similarity=sim, linkage_method="average", cut_threshold=1 - threshold).rename(columns={"structure_cluster_id": "structure_cluster"})


from mcoxplorer.utils.tools import resolve_tool_path


def has_foldseek(cfg: dict) -> bool:
    return resolve_tool_path(cfg["external_tools"]["foldseek"]) is not None
