from __future__ import annotations

import hashlib
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mcoxplorer.io.idmap import IdMapper
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




def cluster_sequences_mmseqs(records: list[SequenceRecord], mmseqs_exe: str | None, tmp_root: Path, min_seq_id: float = 0.4) -> pd.DataFrame:
    """Use MMseqs2 easy-cluster when available; fallback otherwise."""

    if mmseqs_exe is None:
        out = cluster_sequences(records, min_seq_id)
        out["sequence_cluster_backend"] = "fallback_greedy"
        return out
    tmp_root.mkdir(parents=True, exist_ok=True)
    in_fa = tmp_root / "positive.fa"
    _write_fasta(records, in_fa)
    out_prefix = tmp_root / "cluster"
    run_command([mmseqs_exe, "easy-cluster", str(in_fa), str(out_prefix), str(tmp_root / "tmp"), "--min-seq-id", str(min_seq_id), "-c", "0.8"])

    tsv = Path(str(out_prefix) + "_cluster.tsv")
    if not tsv.exists():
        out = cluster_sequences(records, min_seq_id)
        out["sequence_cluster_backend"] = "fallback_greedy"
        return out
    raw = pd.read_csv(tsv, sep="	", header=None, names=["rep", "member"])
    rep_to_cluster = {rep: i + 1 for i, rep in enumerate(sorted(raw["rep"].unique()))}
    rows = []
    for r in records:
        hit = raw[raw["member"] == r.protein_id]
        if hit.empty:
            cid = len(rep_to_cluster) + 1
        else:
            cid = rep_to_cluster[hit.iloc[0]["rep"]]
        rows.append({"protein_id": r.protein_id, "sequence_cluster": cid, "sequence_cluster_backend": "mmseqs2"})
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
    mapper: IdMapper | None = None,
) -> pd.DataFrame:
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
    raw["query"] = raw["query"].astype(str).map(lambda x: mapper.canonical(x) if mapper else x)
    raw["target"] = raw["target"].astype(str).map(lambda x: mapper.canonical(x) if mapper else x)
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


def _run_msa(cluster_fa: Path, msa_fa: Path, tools: dict[str, str | None]) -> str:
    mafft = tools.get("mafft")
    muscle = tools.get("muscle")
    if mafft:
        result = run_command([mafft, "--auto", str(cluster_fa)])
        msa_fa.write_text(result.stdout, encoding="utf-8")
        return "mafft"
    if muscle:
        run_command([muscle, "-align", str(cluster_fa), "-output", str(msa_fa)])
        return "muscle"
    raise RuntimeError("No MSA tool available (mafft/muscle)")


def hmm_scores_hmmer(
    candidates: list[SequenceRecord],
    positives: list[SequenceRecord],
    clusters: pd.DataFrame,
    tools: dict[str, str | None],
    tmp_root: Path,
) -> pd.DataFrame:
    hmmbuild_exe = tools.get("hmmbuild")
    hmmsearch_exe = tools.get("hmmsearch")
    if hmmbuild_exe is None or hmmsearch_exe is None:
        LOGGER.warning("HMMER tools not available. Fallback hmm_score=0.5")
        return pd.DataFrame({"protein_id": [c.protein_id for c in candidates], "hmm_score": [0.5] * len(candidates), "hmm_backend": "fallback"})

    tmp_root.mkdir(parents=True, exist_ok=True)
    cand_fa = tmp_root / "cand.fasta"
    _write_fasta(candidates, cand_fa)
    pos_map = {p.protein_id: p for p in positives}

    hmm_scores: dict[str, float] = {c.protein_id: 0.0 for c in candidates}
    used_msa_backend = "fallback"
    with tempfile.TemporaryDirectory(dir=tmp_root) as td:
        td_path = Path(td)
        for cid, sub in clusters.groupby("sequence_cluster"):
            member_ids = [pid for pid in sub["protein_id"].tolist() if pid in pos_map]
            if len(member_ids) < 2:
                continue
            cluster_fa = td_path / f"cluster_{cid}.fa"
            msa_fa = td_path / f"cluster_{cid}.msa.fa"
            _write_fasta([pos_map[x] for x in member_ids], cluster_fa)
            try:
                msa_backend = _run_msa(cluster_fa, msa_fa, tools)
                used_msa_backend = msa_backend
            except Exception as exc:
                LOGGER.warning("MSA tool unavailable for HMMER cluster %s: %s", cid, exc)
                return pd.DataFrame({"protein_id": [c.protein_id for c in candidates], "hmm_score": [0.5] * len(candidates), "hmm_backend": "fallback"})

            hmm_path = td_path / f"cluster_{cid}.hmm"
            out_tbl = td_path / f"cluster_{cid}.tbl"
            run_command([hmmbuild_exe, str(hmm_path), str(msa_fa)])
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
        rows.append({"protein_id": c.protein_id, "hmm_score": (hmm_scores.get(c.protein_id, 0.0) / max_bit if max_bit > 0 else 0.0), "hmm_backend": f"hmmer+{used_msa_backend}"})
    return pd.DataFrame(rows)


def mock_esm_embedding(seq: str, dim: int = 64) -> np.ndarray:
    h = hashlib.sha256(seq.encode("utf-8")).digest()
    values = np.array([b for b in h[:dim]], dtype=float)
    return values / 255.0


@dataclass
class ESM2EmbeddingRuntime:
    model_name: str
    cache_dir: Path
    device: str
    batch_size: int = 8
    tokenizer: object | None = None
    model: object | None = None

    def initialize(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=str(self.cache_dir), local_files_only=True)
        self.model = AutoModel.from_pretrained(self.model_name, cache_dir=str(self.cache_dir), local_files_only=True)
        self.model.eval()
        self.model.to(self.device)

    def embed_batch(self, sequences: list[str]) -> np.ndarray:
        if self.tokenizer is None or self.model is None:
            self.initialize()
        outputs = []
        from contextlib import nullcontext
        try:
            import torch
            guard = torch.no_grad
        except Exception:
            guard = nullcontext
        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i : i + self.batch_size]
            toks = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with guard():
                out = self.model(**toks)
            emb = out.last_hidden_state.mean(dim=1).detach().cpu().numpy()
            outputs.append(emb)
        return np.vstack(outputs)


def embedding_features(candidates: list[SequenceRecord], positives: list[SequenceRecord], embedding_cfg: dict) -> pd.DataFrame:
    model_name = embedding_cfg.get("model_name", "facebook/esm2_t6_8M_UR50D")
    cache_dir = Path(embedding_cfg.get("cache_dir", "~/.cache/mcoxplorer")).expanduser()
    device = embedding_cfg.get("device", "cpu")
    force_mock = embedding_cfg.get("force_mock", False)
    batch_size = int(embedding_cfg.get("batch_size", 8))

    backend = "mock"
    if not force_mock:
        try:
            rt = ESM2EmbeddingRuntime(model_name=model_name, cache_dir=cache_dir, device=device, batch_size=batch_size)
            pos_matrix = rt.embed_batch([p.sequence for p in positives])
            cand_matrix = rt.embed_batch([c.sequence for c in candidates])
            backend = "esm2"
        except Exception as exc:
            LOGGER.warning("ESM2 embedding unavailable (%s), fallback to mock embedding.", exc)

    if backend == "mock":
        pos_matrix = np.stack([mock_esm_embedding(p.sequence) for p in positives], axis=0)
        cand_matrix = np.stack([mock_esm_embedding(c.sequence) for c in candidates], axis=0)

    centroid = pos_matrix.mean(axis=0)
    rows = []
    for c, emb in zip(candidates, cand_matrix):
        dist = float(np.linalg.norm(emb - centroid))
        novelty = min(1.0, dist / max(1.0, np.sqrt(len(emb))))
        rows.append({"protein_id": c.protein_id, "embedding_distance_to_positive_centroid": dist, "novelty": novelty, "embedding_backend": backend})
    return pd.DataFrame(rows)


def resolve_sequence_tools(external_cfg: dict) -> dict[str, str | None]:
    return {
        "mmseqs": resolve_tool_path(external_cfg.get("mmseqs2", "mmseqs")),
        "hmmbuild": resolve_tool_path(external_cfg.get("hmmbuild", "hmmbuild")),
        "hmmsearch": resolve_tool_path(external_cfg.get("hmmsearch", "hmmsearch")),
        "mafft": resolve_tool_path(external_cfg.get("mafft", "mafft")),
        "muscle": resolve_tool_path(external_cfg.get("muscle", "muscle")),
    }
