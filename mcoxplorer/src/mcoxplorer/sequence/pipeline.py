from __future__ import annotations

from pathlib import Path

import pandas as pd

from mcoxplorer.io.idmap import IdMapper
from mcoxplorer.io.qc import sequence_qc
from mcoxplorer.io.readers import read_csv, read_fasta
from mcoxplorer.sequence.features import (
    cluster_sequences_mmseqs,
    embedding_features,
    hmm_scores_hmmer,
    resolve_sequence_tools,
    similarity_search_fallback,
    similarity_search_mmseqs,
)
from mcoxplorer.sequence.structure_aware_embedding import structure_aware_embedding_features


def run_sequence_module(cfg: dict) -> dict[str, pd.DataFrame]:
    inputs = cfg["inputs"]
    seq_cfg = cfg["sequence"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    mapper = IdMapper(inputs.get("id_mapping_csv"))
    mapper.export_resolved(out_dir / "id_mapping_resolved.csv")

    pos = read_fasta(inputs["positive_fasta"])
    cand = read_fasta(inputs["candidate_fasta"])
    for r in pos:
        r.protein_id = mapper.canonical(r.protein_id)
    for r in cand:
        r.protein_id = mapper.canonical(r.protein_id)

    pos_md = read_csv(inputs["positive_metadata_csv"])
    pos_md = mapper.apply_to_dataframe(pos_md, "protein_id")

    pos_clean, pos_qc = sequence_qc(pos, seq_cfg["min_length"], seq_cfg["max_length"])
    cand_clean, cand_qc = sequence_qc(cand, seq_cfg["min_length"], seq_cfg["max_length"])

    inter_dir = out_dir / "intermediate" / "sequence"
    inter_dir.mkdir(parents=True, exist_ok=True)

    tools = resolve_sequence_tools(cfg.get("external_tools", {}))

    pos_clusters = cluster_sequences_mmseqs(
        pos_clean,
        tools.get("mmseqs"),
        inter_dir / "cluster",
        min_seq_id=seq_cfg["identity_cluster_threshold"],
    ).merge(pos_md[["protein_id", "family", "label_type"]], on="protein_id", how="left")
    sim = similarity_search_mmseqs(
        cand_clean,
        pos_clean,
        tools["mmseqs"],
        inter_dir,
        mapper=mapper,
    )
    if sim.empty:
        sim = similarity_search_fallback(cand_clean, pos_clean)

    emb = embedding_features(cand_clean, pos_clean, seq_cfg.get("embedding", {}))
    struct_aware = structure_aware_embedding_features(cand_clean, pos_clean, seq_cfg.get("structure_aware_embedding", {}))
    hmm = hmm_scores_hmmer(
        cand_clean,
        pos_clean,
        pos_clusters,
        tools,
        inter_dir / "hmmer",
    )

    features = (
        sim.merge(emb, on="protein_id", how="left")
        .merge(struct_aware, on="protein_id", how="left")
        .merge(hmm, on="protein_id", how="left")
    )
    structure_aware_weight = min(max(float(seq_cfg.get("structure_aware_embedding_weight", 0.0)), 0.0), 1.0)
    features["sequence_score"] = (
        0.55 * (1.0 - structure_aware_weight) * features["best_identity_to_positive"].fillna(0)
        + 0.30 * (1.0 - structure_aware_weight) * features["hmm_score"].fillna(0)
        + 0.15 * (1.0 - structure_aware_weight) * (1 - features["embedding_distance_to_positive_centroid"].fillna(1).clip(upper=1))
        + structure_aware_weight * (1 - features["structure_aware_distance_to_positive_centroid"].fillna(1).clip(upper=1))
    ).clip(lower=0, upper=1)
    features["nearest_positive_family"] = features["best_positive_id"].map(pos_md.set_index("protein_id")["family"].to_dict())
    ranked = features.sort_values("sequence_score", ascending=False).reset_index(drop=True)
    ranked["sequence_only_rank"] = ranked.index + 1

    pos_clusters.to_csv(out_dir / "positive_sequence_clusters.csv", index=False)
    features.to_csv(out_dir / "candidate_sequence_features.csv", index=False)
    ranked.to_csv(out_dir / "ranked_candidates_sequence_view.csv", index=False)
    pos_qc.to_csv(out_dir / "positive_sequence_qc.csv", index=False)
    cand_qc.to_csv(out_dir / "candidate_sequence_qc.csv", index=False)

    return {
        "positive_clusters": pos_clusters,
        "candidate_sequence_features": features,
        "ranked_sequence": ranked,
    }
