from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mcoxplorer.io.readers import SequenceRecord


@dataclass
class SimilarityHit:
    query_id: str
    target_id: str
    identity: float


def pairwise_identity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / n


def cluster_sequences(records: list[SequenceRecord], threshold: float) -> pd.DataFrame:
    cluster_ids: dict[str, int] = {}
    cid = 0
    for rec in records:
        assigned = None
        for rep_id, rep_cid in list(cluster_ids.items()):
            rep_seq = next(r.sequence for r in records if r.protein_id == rep_id)
            if pairwise_identity(rec.sequence, rep_seq) >= threshold:
                assigned = rep_cid
                break
        if assigned is None:
            cid += 1
            assigned = cid
            cluster_ids[rec.protein_id] = assigned
    rows = []
    for rec in records:
        c = None
        for rep_id, rep_c in cluster_ids.items():
            rep_seq = next(r.sequence for r in records if r.protein_id == rep_id)
            if pairwise_identity(rec.sequence, rep_seq) >= threshold:
                c = rep_c
                break
        rows.append({"protein_id": rec.protein_id, "sequence_cluster": c})
    return pd.DataFrame(rows)


def similarity_search(candidates: list[SequenceRecord], positives: list[SequenceRecord]) -> pd.DataFrame:
    rows: list[dict] = []
    for c in candidates:
        best = max((pairwise_identity(c.sequence, p.sequence), p.protein_id) for p in positives)
        rows.append(
            {
                "protein_id": c.protein_id,
                "best_positive_id": best[1],
                "best_identity_to_positive": round(best[0], 4),
            }
        )
    return pd.DataFrame(rows)


def mock_esm_embedding(seq: str, dim: int = 16) -> np.ndarray:
    h = hashlib.sha256(seq.encode("utf-8")).digest()
    values = np.array([b for b in h[:dim]], dtype=float)
    return values / 255.0


def embedding_features(candidates: list[SequenceRecord], positives: list[SequenceRecord]) -> pd.DataFrame:
    pos_matrix = np.stack([mock_esm_embedding(p.sequence) for p in positives], axis=0)
    centroid = pos_matrix.mean(axis=0)
    rows = []
    for c in candidates:
        emb = mock_esm_embedding(c.sequence)
        dist = float(np.linalg.norm(emb - centroid))
        novelty = min(1.0, dist / 2.0)
        rows.append({"protein_id": c.protein_id, "embedding_distance_to_positive_centroid": dist, "novelty": novelty})
    return pd.DataFrame(rows)


def hmm_placeholder_scores(candidates: list[SequenceRecord]) -> pd.DataFrame:
    # TODO: integrate HMMER hmmbuild/hmmsearch for real profile HMM scoring.
    return pd.DataFrame({"protein_id": [c.protein_id for c in candidates], "hmm_score": [0.5] * len(candidates)})
