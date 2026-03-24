from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mcoxplorer.io.idmap import IdMapper
from mcoxplorer.io.readers import read_csv, read_fasta, read_structure_dir
from mcoxplorer.structure.features import (
    choose_cluster_medoids,
    compute_local_motif_sequence_support,
    compute_local_support_3d,
    hierarchical_structure_clustering,
    parse_structure_qc,
    symmetrize_similarity_matrix,
)
from mcoxplorer.structure.foldseek_runner import build_structure_db, get_foldseek_runtime, search_candidates_against_positive_db, search_self_against_db
from mcoxplorer.structure.tmalign_runner import refine_top_hits_with_tmalign

LOGGER = logging.getLogger(__name__)
DEFAULT_SIMILARITY = 0.0


def _normalize_foldseek_hit_table(hits: pd.DataFrame, mapper: IdMapper) -> pd.DataFrame:
    if hits.empty:
        return hits
    table = hits.copy()
    table["target_id"] = table["target_id"].astype(str).map(lambda x: mapper.canonical(Path(x).stem))
    table["query_id"] = table["query_id"].astype(str).map(lambda x: mapper.canonical(Path(x).stem))
    return table


def _build_pairwise_matrix(positive_ids: list[str], pair_hits: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.DataFrame(DEFAULT_SIMILARITY, index=positive_ids, columns=positive_ids, dtype=float)
    for _, row in pair_hits.iterrows():
        q, t = row["query_id"], row["target_id"]
        if q in matrix.index and t in matrix.columns:
            score = row.get("prob", row.get("raw_score", 0.0))
            score = 0.0 if pd.isna(score) else float(score)
            sim = float(score if score <= 1 else score / (score + 100.0))
            matrix.loc[q, t] = max(matrix.loc[q, t], sim)
    for pid in positive_ids:
        matrix.loc[pid, pid] = 1.0
    return symmetrize_similarity_matrix(matrix, strategy="mean")


def _rank_structure_features(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df
    out = candidate_df.copy()
    out["structure_score_raw"] = (
        0.22 * out["best_structure_similarity_to_any_positive"].fillna(0)
        + 0.15 * out["best_structure_similarity_to_gold_positive"].fillna(0)
        + 0.12 * out["similarity_to_nearest_cluster_prototype"].fillna(0)
        + 0.10 * out["mean_topk_structure_similarity"].fillna(0)
        + 0.08 * out["structure_family_consistency"].fillna(0)
        + 0.08 * out["combined_local_support"].fillna(0)
        + 0.08 * out["structure_gold_bias"].fillna(0)
        + 0.07 * out["sequence_structure_agreement_score"].fillna(0)
        + 0.10 * (1 - out["possible_generic_mco_risk_structure"].fillna(1.0))
    ).clip(lower=0, upper=1)
    out["structure_quality_penalty"] = out.get("structure_quality_penalty", 0.0).fillna(0.0)
    out["structure_score"] = (out["structure_score_raw"] * (1 - out["structure_quality_penalty"])).clip(lower=0, upper=1)
    out["structure_evidence_usable"] = out["structure_quality_penalty"] < 0.5
    out["remote_but_structure_supported"] = (out["best_identity_to_positive"].fillna(1.0) < 0.35) & (out["best_structure_similarity_to_any_positive"].fillna(0.0) >= 0.65)
    out["high_confidence_first_batch"] = (out["structure_score"].fillna(0) >= 0.65) & (out["possible_generic_mco_risk_structure"].fillna(1) < 0.5)
    out = out.sort_values("structure_score", ascending=False).reset_index(drop=True)
    out["structure_only_rank"] = out.index + 1
    return out


def run_structure_module(cfg: dict, sequence_features: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    inter_dir = out_dir / "intermediate" / "structure"
    inter_dir.mkdir(parents=True, exist_ok=True)

    inputs = cfg["inputs"]
    mapper = IdMapper(inputs.get("id_mapping_csv"))
    mapper.export_resolved(out_dir / "id_mapping_resolved.csv")

    runtime_cfg = cfg.get("runtime", {})
    structure_cfg = cfg["structure"]
    reuse_existing = runtime_cfg.get("reuse_existing", True)
    force_rerun = runtime_cfg.get("force_rerun", False)

    pos_struct_raw = read_structure_dir(inputs["positive_structure_dir"])
    cand_struct_raw = read_structure_dir(inputs["candidate_structure_dir"])
    pos_struct = {mapper.canonical(k): v for k, v in pos_struct_raw.items()}
    cand_struct = {mapper.canonical(k): v for k, v in cand_struct_raw.items()}

    positives = read_fasta(inputs["positive_fasta"])
    candidates = read_fasta(inputs["candidate_fasta"])
    for r in positives:
        r.protein_id = mapper.canonical(r.protein_id)
    for r in candidates:
        r.protein_id = mapper.canonical(r.protein_id)

    pos_md = mapper.apply_to_dataframe(read_csv(inputs["positive_metadata_csv"]), "protein_id")
    struct_md = mapper.apply_to_dataframe(read_csv(inputs["structure_metadata_csv"]), "protein_id") if Path(inputs.get("structure_metadata_csv", "")).exists() else pd.DataFrame()

    all_candidate_ids = [c.protein_id for c in candidates]
    seq_len_map = {r.protein_id: len(r.sequence) for r in positives + candidates}
    md_map = struct_md.set_index("protein_id") if not struct_md.empty and "protein_id" in struct_md.columns else pd.DataFrame()
    quality_cfg = structure_cfg.get("quality", {})

    qc_rows = []
    for sid, spath in {**pos_struct, **cand_struct}.items():
        row = md_map.loc[sid] if (not md_map.empty and sid in md_map.index) else None
        qc = parse_structure_qc(
            spath,
            seq_length=seq_len_map.get(sid),
            metadata_row=row,
            min_modeled_residue_count=int(quality_cfg.get("min_modeled_residue_count", 1)),
            min_structure_confidence=float(quality_cfg.get("min_structure_confidence", 0.0)),
            min_structure_coverage=float(quality_cfg.get("min_structure_coverage", 0.0)),
        )
        qc["protein_id"] = sid
        qc_rows.append(qc)
    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(out_dir / "structure_qc_summary.csv", index=False)

    foldseek_rt = get_foldseek_runtime(cfg["external_tools"]["foldseek"])
    LOGGER.info("Foldseek path: %s", foldseek_rt.executable)

    if not foldseek_rt.available:
        message = "Foldseek not found; structure module falls back to NA features."
        if runtime_cfg.get("structure_required", False):
            raise RuntimeError(message)
        LOGGER.warning(message)
        empty_struct = pd.DataFrame({"candidate_id": all_candidate_ids, "protein_id": all_candidate_ids})
        empty_struct.to_csv(out_dir / "candidate_structure_features.csv", index=False)
        empty_struct.to_csv(out_dir / "ranked_candidates_structure_view.csv", index=False)
        return {"candidate_structure_features": empty_struct, "ranked_structure": empty_struct}

    positive_db = inter_dir / "positive_db"
    build_structure_db(inputs["positive_structure_dir"], positive_db, foldseek_rt.executable, force=force_rerun)

    pair_hits = _normalize_foldseek_hit_table(
        search_self_against_db(inputs["positive_structure_dir"], positive_db, inter_dir / "positive_vs_positive_foldseek.tsv", inter_dir / "tmp_pos", foldseek_rt.executable, force=force_rerun or (not reuse_existing)),
        mapper,
    )

    pos_ids = [p.protein_id for p in positives if p.protein_id in pos_struct]
    similarity_matrix = _build_pairwise_matrix(pos_ids, pair_hits)
    similarity_matrix.to_csv(out_dir / "positive_structure_similarity_matrix.csv")

    cluster_cfg = structure_cfg.get("clustering", {})
    cluster_df = hierarchical_structure_clustering(similarity_matrix, linkage_method=cluster_cfg.get("linkage_method", "average"), cut_threshold=cluster_cfg.get("cut_distance_threshold", 0.45))
    proto_df = choose_cluster_medoids(similarity_matrix, cluster_df)

    cluster_df = cluster_df.merge(pos_md[["protein_id", "family", "label_type"]], on="protein_id", how="left")
    cluster_df = cluster_df.merge(proto_df[["structure_cluster_id", "prototype_id", "cluster_size", "cluster_compactness"]], on="structure_cluster_id", how="left")
    cluster_df["is_prototype"] = cluster_df["protein_id"] == cluster_df["prototype_id"]
    cluster_df["nearest_prototype_id"] = cluster_df["prototype_id"]
    cluster_df.to_csv(out_dir / "positive_structure_clusters.csv", index=False)

    proto_with_meta = proto_df.merge(cluster_df[["protein_id", "family", "label_type"]].drop_duplicates(), left_on="prototype_id", right_on="protein_id", how="left").drop(columns=["protein_id"])
    proto_with_meta["prototype_structure_path"] = proto_with_meta["prototype_id"].map({k: str(v) for k, v in pos_struct.items()})
    proto_with_meta.to_csv(out_dir / "positive_structure_prototypes.csv", index=False)

    fam_summary = cluster_df.groupby(["family", "structure_cluster_id", "label_type"]).size().reset_index(name="count").sort_values(["structure_cluster_id", "count"], ascending=[True, False])
    fam_summary.to_csv(out_dir / "positive_structure_cluster_family_summary.csv", index=False)

    cand_hits = _normalize_foldseek_hit_table(
        search_candidates_against_positive_db(inputs["candidate_structure_dir"], positive_db, inter_dir / "candidates_vs_positive_foldseek.tsv", inter_dir / "tmp_cand", foldseek_rt.executable, params=structure_cfg.get("foldseek", {}), force=force_rerun or (not reuse_existing)),
        mapper,
    )

    structure_lookup = {**pos_struct, **cand_struct}
    tma_cfg = structure_cfg.get("tmalign", {})
    tma_df = refine_top_hits_with_tmalign(cand_hits, top_k_per_query=int(tma_cfg.get("top_k_per_query", 3)), structure_lookup=structure_lookup, tmalign_config=cfg["external_tools"]["tmalign"], raw_text_dir=inter_dir / "tmalign_raw") if tma_cfg.get("enabled", True) else pd.DataFrame()

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
    seq_identity_map = {}
    if sequence_features is not None and not sequence_features.empty:
        seq_family_map = sequence_features.set_index("protein_id")["nearest_positive_family"].to_dict()
        if "best_identity_to_positive" in sequence_features.columns:
            seq_identity_map = sequence_features.set_index("protein_id")["best_identity_to_positive"].to_dict()

    seq_map = {c.protein_id: c.sequence for c in candidates}
    support_threshold = structure_cfg.get("support_similarity_threshold", 0.45)
    topk = int(structure_cfg.get("top_k_support", 3))

    rows = []
    for cid in all_candidate_ids:
        sub = cand_hits[cand_hits["query_id"] == cid].sort_values("similarity_final", ascending=False)
        seq_local = compute_local_motif_sequence_support(seq_map.get(cid, ""))
        str_local = compute_local_support_3d(cand_struct.get(cid, Path("")), seq_map.get(cid, ""), seq_local.get("motif_positions", []), radius=float(structure_cfg.get("local_radius", 8.0)))
        combined_local = np.nanmean([seq_local["sequence_local_support"], str_local.get("structure_local_support_3d")]) if pd.notna(str_local.get("structure_local_support_3d")) else seq_local["sequence_local_support"]

        if sub.empty:
            row = {
                "candidate_id": cid,
                "protein_id": cid,
                **{k: v for k, v in seq_local.items() if k != "motif_positions"},
                **str_local,
                "combined_local_support": combined_local,
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
                "structure_reason_summary": "No structure evidence available; fallback to sequence-only.",
                "best_structure_similarity_to_positive": pd.NA,
                "best_identity_to_positive": seq_identity_map.get(cid, pd.NA),
                "structure_quality_penalty": 1.0,
            }
            rows.append(row)
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
            agreement = 0.2

        gold_bias = float((sub["target_id"].map(pos_label_map) == "gold").mean())
        silver_bias = float((sub["target_id"].map(pos_label_map) == "silver").mean())
        generic_risk = 1.0 if (best_any < 0.45 and mean_topk < 0.4 and combined_local < 0.45) else (0.6 if family_consistency == 0 and best_any < 0.55 else 0.2)

        quality_penalty = 0.0
        if cid in qc_by_id.index:
            qrow = qc_by_id.loc[cid]
            qcfg = structure_cfg.get("quality", {})
            if not bool(qrow.get("structure_quality_pass", False)):
                quality_penalty = float(qcfg.get("penalty_if_quality_fail", 0.45))
            if pd.notna(qrow.get("structure_coverage")) and qrow.get("structure_coverage") < qcfg.get("min_structure_coverage", 0.5):
                quality_penalty = max(quality_penalty, float(qcfg.get("penalty_if_low_coverage", 0.30)))
            if pd.notna(qrow.get("mean_structure_confidence")) and qrow.get("mean_structure_confidence") < qcfg.get("min_structure_confidence", 50.0):
                quality_penalty = max(quality_penalty, float(qcfg.get("penalty_if_low_confidence", 0.30)))

        reason = "Strong global+local structural support." if (best_any >= 0.7 and combined_local >= 0.5) else ("Moderate structure support; validate with sequence evidence." if best_any >= 0.6 else "Weak structure support; possible generic MCO background risk.")

        row = {
            "candidate_id": cid,
            "protein_id": cid,
            **{k: v for k, v in seq_local.items() if k != "motif_positions"},
            **str_local,
            "combined_local_support": combined_local,
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
            "structure_reason_summary": reason,
            "best_structure_similarity_to_positive": best_any,
            "best_identity_to_positive": seq_identity_map.get(cid, pd.NA),
            "structure_quality_penalty": quality_penalty,
        }
        if cid in qc_by_id.index:
            row.update(
                {
                    "structure_coverage": qc_by_id.loc[cid, "structure_coverage"],
                    "modeled_residue_count": qc_by_id.loc[cid, "modeled_residue_count"],
                    "mean_structure_confidence": qc_by_id.loc[cid, "mean_structure_confidence"],
                    "structure_quality_pass": qc_by_id.loc[cid, "structure_quality_pass"],
                }
            )
        rows.append(row)

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
