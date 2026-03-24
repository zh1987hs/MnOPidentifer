from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from mcoxplorer.io.readers import read_csv, read_fasta, read_structure_dir
from mcoxplorer.structure.features import (
    choose_cluster_medoids,
    hierarchical_structure_clustering,
    parse_structure_qc,
    symmetrize_similarity_matrix,
)
from mcoxplorer.structure.foldseek_runner import (
    build_structure_db,
    get_foldseek_runtime,
    search_candidates_against_positive_db,
    search_self_against_db,
)
from mcoxplorer.structure.tmalign_runner import refine_top_hits_with_tmalign

LOGGER = logging.getLogger(__name__)


DEFAULT_SIMILARITY = 0.0


def _normalize_foldseek_hit_table(hits: pd.DataFrame) -> pd.DataFrame:
    if hits.empty:
        return hits
    table = hits.copy()
    table["target_id"] = table["target_id"].astype(str).map(lambda x: Path(x).stem)
    table["query_id"] = table["query_id"].astype(str).map(lambda x: Path(x).stem)
    return table


def _build_pairwise_matrix(positive_ids: list[str], pair_hits: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.DataFrame(DEFAULT_SIMILARITY, index=positive_ids, columns=positive_ids, dtype=float)
    for _, row in pair_hits.iterrows():
        q, t = row["query_id"], row["target_id"]
        if q in matrix.index and t in matrix.columns:
            score = row.get("prob", None)
            if pd.isna(score):
                score = row.get("raw_score", 0.0)
            if pd.isna(score):
                score = 0.0
            # If Foldseek score > 1, squash to [0,1] using a stable transform.
            sim = float(score if score <= 1 else score / (score + 100.0))
            matrix.loc[q, t] = max(matrix.loc[q, t], sim)
    for pid in positive_ids:
        matrix.loc[pid, pid] = 1.0
    return symmetrize_similarity_matrix(matrix, strategy="mean")


def _rank_structure_features(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df
    out = candidate_df.copy()
    out["structure_score"] = (
        0.35 * out["best_structure_similarity_to_any_positive"].fillna(0)
        + 0.25 * out["best_structure_similarity_to_gold_positive"].fillna(0)
        + 0.15 * out["similarity_to_nearest_cluster_prototype"].fillna(0)
        + 0.10 * out["mean_topk_structure_similarity"].fillna(0)
        + 0.05 * out["sequence_structure_agreement_score"].fillna(0)
        + 0.10 * (1 - out["possible_generic_mco_risk_structure"].fillna(1.0))
    ).clip(lower=0, upper=1)
    out = out.sort_values("structure_score", ascending=False).reset_index(drop=True)
    out["structure_only_rank"] = out.index + 1
    return out


def run_structure_module(cfg: dict, sequence_features: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    inter_dir = out_dir / "intermediate" / "structure"
    inter_dir.mkdir(parents=True, exist_ok=True)

    inputs = cfg["inputs"]
    runtime_cfg = cfg.get("runtime", {})
    structure_cfg = cfg["structure"]
    reuse_existing = runtime_cfg.get("reuse_existing", True)
    force_rerun = runtime_cfg.get("force_rerun", False)

    pos_struct = read_structure_dir(inputs["positive_structure_dir"])
    cand_struct = read_structure_dir(inputs["candidate_structure_dir"])
    positives = read_fasta(inputs["positive_fasta"])
    candidates = read_fasta(inputs["candidate_fasta"])
    pos_md = read_csv(inputs["positive_metadata_csv"])

    all_candidate_ids = [c.protein_id for c in candidates]

    qc_rows = [parse_structure_qc(p) for p in list(pos_struct.values()) + list(cand_struct.values())]
    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(out_dir / "structure_qc_summary.csv", index=False)

    foldseek_rt = get_foldseek_runtime(cfg["external_tools"]["foldseek"])
    LOGGER.info("Foldseek path: %s", foldseek_rt.executable)

    if not foldseek_rt.available:
        message = "Foldseek not found; structure module falls back to NA features."
        if runtime_cfg.get("structure_required", False):
            raise RuntimeError(message)
        LOGGER.warning(message)
        empty_struct = pd.DataFrame({"candidate_id": all_candidate_ids})
        empty_struct["protein_id"] = empty_struct["candidate_id"]
        for col in [
            "best_structure_similarity_to_any_positive",
            "best_structure_similarity_to_gold_positive",
            "mean_topk_structure_similarity",
            "nearest_positive_structure_cluster",
            "similarity_to_nearest_cluster_prototype",
            "structure_support_count_above_threshold",
            "structure_novelty_score",
            "structure_family_consistency",
            "structure_gold_bias",
            "structure_silver_bias",
            "possible_generic_mco_risk_structure",
            "sequence_structure_agreement_score",
            "structure_coverage",
            "modeled_residue_count",
            "mean_structure_confidence",
            "structure_quality_pass",
            "structure_reason_summary",
            "structure_score",
            "structure_only_rank",
            "best_structure_similarity_to_positive",
        ]:
            empty_struct[col] = pd.NA
        empty_struct.to_csv(out_dir / "candidate_structure_features.csv", index=False)
        empty_struct.to_csv(out_dir / "ranked_candidates_structure_view.csv", index=False)
        return {"candidate_structure_features": empty_struct, "ranked_structure": empty_struct}

    positive_db = inter_dir / "positive_db"
    build_structure_db(inputs["positive_structure_dir"], positive_db, foldseek_rt.executable, force=force_rerun)

    pair_hits_path = inter_dir / "positive_vs_positive_foldseek.tsv"
    pair_hits = search_self_against_db(
        inputs["positive_structure_dir"],
        positive_db,
        pair_hits_path,
        inter_dir / "tmp_pos",
        foldseek_rt.executable,
        force=force_rerun or (not reuse_existing),
    )
    pair_hits = _normalize_foldseek_hit_table(pair_hits)

    pos_ids = [p.protein_id for p in positives if p.protein_id in pos_struct]
    similarity_matrix = _build_pairwise_matrix(pos_ids, pair_hits)
    similarity_matrix.to_csv(out_dir / "positive_structure_similarity_matrix.csv")

    cluster_cfg = structure_cfg.get("clustering", {})
    cluster_df = hierarchical_structure_clustering(
        similarity_matrix,
        linkage_method=cluster_cfg.get("linkage_method", "average"),
        cut_threshold=cluster_cfg.get("cut_distance_threshold", 0.45),
    )
    proto_df = choose_cluster_medoids(similarity_matrix, cluster_df)

    cluster_df = cluster_df.merge(pos_md[["protein_id", "family", "label_type"]], on="protein_id", how="left")
    cluster_df = cluster_df.merge(proto_df[["structure_cluster_id", "prototype_id", "cluster_size"]], on="structure_cluster_id", how="left")
    cluster_df["is_prototype"] = cluster_df["protein_id"] == cluster_df["prototype_id"]
    cluster_df["nearest_prototype_id"] = cluster_df["prototype_id"]
    cluster_df.to_csv(out_dir / "positive_structure_clusters.csv", index=False)

    proto_with_meta = proto_df.merge(
        cluster_df[["protein_id", "family", "label_type"]].drop_duplicates(),
        left_on="prototype_id",
        right_on="protein_id",
        how="left",
    ).drop(columns=["protein_id"])
    proto_with_meta["prototype_structure_path"] = proto_with_meta["prototype_id"].map({k: str(v) for k, v in pos_struct.items()})
    proto_with_meta.to_csv(out_dir / "positive_structure_prototypes.csv", index=False)

    fam_summary = (
        cluster_df.groupby(["family", "structure_cluster_id", "label_type"]) 
        .size()
        .reset_index(name="count")
        .sort_values(["structure_cluster_id", "count"], ascending=[True, False])
    )
    fam_summary.to_csv(out_dir / "positive_structure_cluster_family_summary.csv", index=False)

    cand_hits_path = inter_dir / "candidates_vs_positive_foldseek.tsv"
    cand_hits = search_candidates_against_positive_db(
        inputs["candidate_structure_dir"],
        positive_db,
        cand_hits_path,
        inter_dir / "tmp_cand",
        foldseek_rt.executable,
        params=structure_cfg.get("foldseek", {}),
        force=force_rerun or (not reuse_existing),
    )
    cand_hits = _normalize_foldseek_hit_table(cand_hits)

    structure_lookup = {**pos_struct, **cand_struct}
    tma_df = pd.DataFrame()
    tma_cfg = structure_cfg.get("tmalign", {})
    if tma_cfg.get("enabled", True):
        tma_df = refine_top_hits_with_tmalign(
            cand_hits,
            top_k_per_query=int(tma_cfg.get("top_k_per_query", 3)),
            structure_lookup=structure_lookup,
            tmalign_config=cfg["external_tools"]["tmalign"],
            raw_text_dir=inter_dir / "tmalign_raw",
        )
    if not tma_df.empty:
        tma_score = tma_df.copy()
        tma_score["tm_sym"] = tma_score[["tm_score_query_norm", "tm_score_target_norm"]].mean(axis=1)
        tma_best = tma_score.groupby(["query_id", "target_id"], as_index=False)["tm_sym"].max()
        cand_hits = cand_hits.merge(tma_best, on=["query_id", "target_id"], how="left")
        cand_hits["similarity_final"] = cand_hits["tm_sym"].fillna(cand_hits["prob"].fillna(0.0))
    else:
        cand_hits["similarity_final"] = cand_hits["prob"].fillna(0.0)

    pos_label_map = pos_md.set_index("protein_id")["label_type"].to_dict()
    pos_family_map = pos_md.set_index("protein_id")["family"].to_dict()
    pos_cluster_map = cluster_df.set_index("protein_id")["structure_cluster_id"].to_dict()
    prototype_map = proto_df.set_index("structure_cluster_id")["prototype_id"].to_dict()

    qc_by_id = qc_df.set_index("protein_id") if not qc_df.empty else pd.DataFrame()
    seq_family_map = {}
    if sequence_features is not None and (not sequence_features.empty):
        seq_family_map = sequence_features.set_index("protein_id")["nearest_positive_family"].to_dict()

    support_threshold = structure_cfg.get("support_similarity_threshold", 0.45)
    topk = int(structure_cfg.get("top_k_support", 3))

    rows = []
    for cid in all_candidate_ids:
        sub = cand_hits[cand_hits["query_id"] == cid].sort_values("similarity_final", ascending=False)
        has_structure = cid in cand_struct
        if sub.empty:
            rows.append(
                {
                    "candidate_id": cid,
                    "protein_id": cid,
                    "best_structure_similarity_to_any_positive": pd.NA,
                    "best_structure_similarity_to_gold_positive": pd.NA,
                    "mean_topk_structure_similarity": pd.NA,
                    "nearest_positive_structure_cluster": pd.NA,
                    "similarity_to_nearest_cluster_prototype": pd.NA,
                    "structure_support_count_above_threshold": 0,
                    "structure_novelty_score": pd.NA,
                    "structure_family_consistency": pd.NA,
                    "structure_gold_bias": pd.NA,
                    "structure_silver_bias": pd.NA,
                    "possible_generic_mco_risk_structure": pd.NA,
                    "sequence_structure_agreement_score": pd.NA,
                    "structure_coverage": qc_by_id.loc[cid, "structure_coverage"] if has_structure and cid in qc_by_id.index else pd.NA,
                    "modeled_residue_count": qc_by_id.loc[cid, "modeled_residue_count"] if has_structure and cid in qc_by_id.index else pd.NA,
                    "mean_structure_confidence": qc_by_id.loc[cid, "mean_structure_confidence"] if has_structure and cid in qc_by_id.index else pd.NA,
                    "structure_quality_pass": qc_by_id.loc[cid, "structure_quality_pass"] if has_structure and cid in qc_by_id.index else False,
                    "structure_reason_summary": "No structure evidence available; fallback to sequence-only.",
                    "best_structure_similarity_to_positive": pd.NA,
                }
            )
            continue

        best = sub.iloc[0]
        top = sub.head(topk)
        best_any = float(best["similarity_final"])
        best_gold = float(sub[sub["target_id"].map(pos_label_map) == "gold"]["similarity_final"].max()) if not sub[sub["target_id"].map(pos_label_map) == "gold"].empty else 0.0
        nearest_target = str(best["target_id"])
        nearest_cluster = pos_cluster_map.get(nearest_target)
        prototype_id = prototype_map.get(nearest_cluster)
        proto_sim = float(sub[sub["target_id"] == prototype_id]["similarity_final"].max()) if prototype_id else 0.0
        mean_topk = float(top["similarity_final"].mean())
        support_count = int((sub["similarity_final"] >= support_threshold).sum())
        novelty = float(1 - best_any)

        nearest_family = pos_family_map.get(nearest_target)
        seq_family = seq_family_map.get(cid)
        family_consistency = 1.0 if seq_family and nearest_family and seq_family == nearest_family else 0.0
        agreement = 0.5 + 0.5 * family_consistency
        if seq_family and nearest_family and seq_family != nearest_family and best_any > 0.7:
            agreement = 0.25

        gold_bias = float((sub["target_id"].map(pos_label_map) == "gold").mean())
        silver_bias = float((sub["target_id"].map(pos_label_map) == "silver").mean())

        generic_risk = 1.0 if (best_any < 0.45 and mean_topk < 0.4) else (0.6 if family_consistency == 0 and best_any < 0.55 else 0.2)

        if best_any >= 0.7 and (sequence_features is not None) and cid in seq_family_map and seq_family_map.get(cid) is not None:
            reason = "Strong structure support with positive cluster/prototype proximity."
        elif best_any >= 0.6:
            reason = "Moderate structure support; prioritize with sequence evidence."
        else:
            reason = "Weak structure support; possible generic MCO background risk."

        rows.append(
            {
                "candidate_id": cid,
                "protein_id": cid,
                "best_structure_similarity_to_any_positive": best_any,
                "best_structure_similarity_to_gold_positive": best_gold,
                "mean_topk_structure_similarity": mean_topk,
                "nearest_positive_structure_cluster": nearest_cluster,
                "similarity_to_nearest_cluster_prototype": proto_sim,
                "structure_support_count_above_threshold": support_count,
                "structure_novelty_score": novelty,
                "structure_family_consistency": family_consistency,
                "structure_gold_bias": gold_bias,
                "structure_silver_bias": silver_bias,
                "possible_generic_mco_risk_structure": generic_risk,
                "sequence_structure_agreement_score": agreement,
                "structure_coverage": qc_by_id.loc[cid, "structure_coverage"] if cid in qc_by_id.index else pd.NA,
                "modeled_residue_count": qc_by_id.loc[cid, "modeled_residue_count"] if cid in qc_by_id.index else pd.NA,
                "mean_structure_confidence": qc_by_id.loc[cid, "mean_structure_confidence"] if cid in qc_by_id.index else pd.NA,
                "structure_quality_pass": qc_by_id.loc[cid, "structure_quality_pass"] if cid in qc_by_id.index else False,
                "structure_reason_summary": reason,
                "best_structure_similarity_to_positive": best_any,
            }
        )

    feature_df = pd.DataFrame(rows)
    ranked_df = _rank_structure_features(feature_df)

    feature_df.to_csv(out_dir / "candidate_structure_features.csv", index=False)
    ranked_df.to_csv(out_dir / "ranked_candidates_structure_view.csv", index=False)
    return {
        "candidate_structure_features": feature_df,
        "ranked_structure": ranked_df,
        "positive_structure_clusters": cluster_df,
        "positive_structure_prototypes": proto_with_meta,
        "positive_structure_similarity_matrix": similarity_matrix,
    }
