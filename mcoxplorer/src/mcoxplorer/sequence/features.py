from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from mcoxplorer.io.readers import SequenceRecord
from mcoxplorer.utils.subprocess import run_command
from mcoxplorer.utils.tools import resolve_tool_path

LOGGER = logging.getLogger(__name__)


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


def _write_fasta(records: list[SequenceRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(f">{r.protein_id}\n{r.sequence}\n")


def similarity_search_mmseqs(
    candidates: list[SequenceRecord],
    positives: list[SequenceRecord],
    mmseqs_exe: str | None,
    tmp_root: Path,
) -> pd.DataFrame:
    """Real-first MMseqs2 search; fallback to in-memory identity if unavailable."""

    if mmseqs_exe is None:
        LOGGER.warning("MMseqs2 not available. Fallback to pairwise identity.")
        return similarity_search_fallback(candidates, positives)

    tmp_root.mkdir(parents=True, exist_ok=True)
    q_fa, t_fa = tmp_root / "cand.fasta", tmp_root / "pos.fasta"
    _write_fasta(candidates, q_fa)
    _write_fasta(positives, t_fa)

    qdb, tdb, res = tmp_root / "qdb", tmp_root / "tdb", tmp_root / "resdb"
    out_tsv = tmp_root / "mmseqs_result.tsv"
    run_command([mmseqs_exe, "createdb", str(q_fa), str(qdb)])
    run_command([mmseqs_exe, "createdb", str(t_fa), str(tdb)])
    run_command([mmseqs_exe, "search", str(qdb), str(tdb), str(res), str(tmp_root / "tmp"), "--max-seqs", "50"])
    run_command([mmseqs_exe, "convertalis", str(qdb), str(tdb), str(res), str(out_tsv), "--format-output", "query,target,fident,bits,evalue"])

    raw = pd.read_csv(out_tsv, sep="\t", header=None, names=["query", "target", "fident", "bits", "evalue"])
    raw["fident"] = pd.to_numeric(raw["fident"], errors="coerce").fillna(0.0)
    best = raw.sort_values(["query", "fident", "bits"], ascending=[True, False, False]).groupby("query", as_index=False).first()
    out = best.rename(columns={"query": "protein_id", "target": "best_positive_id", "fident": "best_identity_to_positive"})
    out["best_identity_to_positive"] = out["best_identity_to_positive"] / 100.0
    out["sequence_backend"] = "mmseqs2"
    return out[["protein_id", "best_positive_id", "best_identity_to_positive", "sequence_backend"]]


def similarity_search_fallback(candidates: list[SequenceRecord], positives: list[SequenceRecord]) -> pd.DataFrame:
    rows: list[dict] = []
    for c in candidates:
        best = max((pairwise_identity(c.sequence, p.sequence), p.protein_id) for p in positives)
        rows.append(
            {
                "protein_id": c.protein_id,
                "best_positive_id": best[1],
                "best_identity_to_positive": round(best[0], 4),
                "sequence_backend": "fallback_pairwise",
            }
        )
    return pd.DataFrame(rows)


def hmm_scores_hmmer(
    candidates: list[SequenceRecord],
    positives: list[SequenceRecord],
    clusters: pd.DataFrame,
    hmmbuild_exe: str | None,
    hmmsearch_exe: str | None,
    tmp_root: Path,
) -> pd.DataFrame:
    """Build cluster HMMs and run hmmsearch; fallback to constant score."""

    if hmmbuild_exe is None or hmmsearch_exe is None:
        LOGGER.warning("HMMER tools not available. Fallback hmm_score=0.5")
        return pd.DataFrame({"protein_id": [c.protein_id for c in candidates], "hmm_score": [0.5] * len(candidates), "hmm_backend": "fallback"})

    tmp_root.mkdir(parents=True, exist_ok=True)
    cand_fa = tmp_root / "cand.fasta"
    _write_fasta(candidates, cand_fa)
    pos_map = {p.protein_id: p for p in positives}

    hmm_scores: dict[str, float] = {c.protein_id: 0.0 for c in candidates}
    with tempfile.TemporaryDirectory(dir=tmp_root) as td:
        td_path = Path(td)
        for cid, sub in clusters.groupby("sequence_cluster"):
            member_ids = [pid for pid in sub["protein_id"].tolist() if pid in pos_map]
            if len(member_ids) < 2:
                continue
            cluster_fa = td_path / f"cluster_{cid}.fa"
            _write_fasta([pos_map[x] for x in member_ids], cluster_fa)
            hmm_path = td_path / f"cluster_{cid}.hmm"
            out_tbl = td_path / f"cluster_{cid}.tbl"
            run_command([hmmbuild_exe, str(hmm_path), str(cluster_fa)])
            run_command([hmmsearch_exe, "--tblout", str(out_tbl), str(hmm_path), str(cand_fa)])
            if out_tbl.exists():
                for line in out_tbl.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split()
                    target = parts[0]
                    bit = float(parts[5]) if len(parts) > 5 else 0.0
                    hmm_scores[target] = max(hmm_scores.get(target, 0.0), bit)
    max_bit = max(hmm_scores.values()) if hmm_scores else 1.0
    rows = []
    for c in candidates:
        rows.append({"protein_id": c.protein_id, "hmm_score": (hmm_scores.get(c.protein_id, 0.0) / max_bit if max_bit > 0 else 0.0), "hmm_backend": "hmmer"})
    return pd.DataFrame(rows)


def mock_esm_embedding(seq: str, dim: int = 64) -> np.ndarray:
    h = hashlib.sha256(seq.encode("utf-8")).digest()
    values = np.array([b for b in h[:dim]], dtype=float)
    return values / 255.0


def _esm2_embedding_single(seq: str, model_name: str, cache_dir: Path, device: str) -> np.ndarray:
    """Try extracting real ESM2 embedding from local model cache.

    Requires `torch` and `transformers` installed with local cache available.
    """

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(f"ESM2 runtime deps missing: {exc}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir), local_files_only=True)
    model = AutoModel.from_pretrained(model_name, cache_dir=str(cache_dir), local_files_only=True)
    model.eval()
    model.to(device)
    with torch.no_grad():
        tokens = tokenizer(seq, return_tensors="pt", truncation=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        out = model(**tokens)
        emb = out.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy()
    return emb


def embedding_features(
    candidates: list[SequenceRecord],
    positives: list[SequenceRecord],
    embedding_cfg: dict,
) -> pd.DataFrame:
    """Real-first ESM2 embedding; deterministic mock fallback."""

    model_name = embedding_cfg.get("model_name", "facebook/esm2_t6_8M_UR50D")
    cache_dir = Path(embedding_cfg.get("cache_dir", "~/.cache/mcoxplorer")).expanduser()
    device = embedding_cfg.get("device", "cpu")
    force_mock = embedding_cfg.get("force_mock", False)

    backend = "mock"
    pos_embs, cand_embs = [], []
    if not force_mock:
        try:
            pos_embs = [_esm2_embedding_single(p.sequence, model_name, cache_dir, device) for p in positives]
            cand_embs = [_esm2_embedding_single(c.sequence, model_name, cache_dir, device) for c in candidates]
            backend = "esm2"
        except Exception as exc:
            LOGGER.warning("ESM2 embedding unavailable (%s), fallback to mock embedding.", exc)

    if backend == "mock":
        pos_embs = [mock_esm_embedding(p.sequence) for p in positives]
        cand_embs = [mock_esm_embedding(c.sequence) for c in candidates]

    pos_matrix = np.stack(pos_embs, axis=0)
    centroid = pos_matrix.mean(axis=0)
    rows = []
    for c, emb in zip(candidates, cand_embs):
        dist = float(np.linalg.norm(emb - centroid))
        novelty = min(1.0, dist / max(1.0, np.sqrt(len(emb))))
        rows.append(
            {
                "protein_id": c.protein_id,
                "embedding_distance_to_positive_centroid": dist,
                "novelty": novelty,
                "embedding_backend": backend,
            }
        )
    return pd.DataFrame(rows)


def resolve_sequence_tools(external_cfg: dict) -> dict[str, str | None]:
    return {
        "mmseqs": resolve_tool_path(external_cfg.get("mmseqs2", "mmseqs")),
        "hmmbuild": resolve_tool_path(external_cfg.get("hmmbuild", "hmmbuild")),
        "hmmsearch": resolve_tool_path(external_cfg.get("hmmsearch", "hmmsearch")),
    }
