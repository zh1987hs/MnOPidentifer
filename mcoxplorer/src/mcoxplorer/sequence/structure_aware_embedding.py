from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mcoxplorer.io.readers import SequenceRecord

LOGGER = logging.getLogger(__name__)


def _mock_embedding(seq: str, dim: int = 128) -> np.ndarray:
    h = hashlib.sha256(("structaware:" + seq).encode("utf-8")).digest()
    vals = np.array([h[i % len(h)] for i in range(dim)], dtype=float)
    return vals / 255.0


@dataclass
class StructureAwareEmbeddingRuntime:
    backend: str
    model_name: str
    cache_dir: Path
    device: str
    batch_size: int
    tokenizer: object | None = None
    model: object | None = None

    def initialize(self) -> None:
        from transformers import AutoModel, AutoTokenizer, T5EncoderModel

        if self.backend == "prostt5":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=str(self.cache_dir), local_files_only=True)
            self.model = T5EncoderModel.from_pretrained(self.model_name, cache_dir=str(self.cache_dir), local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=str(self.cache_dir), local_files_only=True)
            self.model = AutoModel.from_pretrained(self.model_name, cache_dir=str(self.cache_dir), local_files_only=True)
        self.model.eval()
        self.model.to(self.device)

    def embed_batch(self, sequences: list[str]) -> np.ndarray:
        if self.tokenizer is None or self.model is None:
            self.initialize()
        from contextlib import nullcontext

        try:
            import torch

            guard = torch.no_grad
        except Exception:
            guard = nullcontext

        outs = []
        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i : i + self.batch_size]
            toks = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with guard():
                output = self.model(**toks)
            emb = output.last_hidden_state.mean(dim=1).detach().cpu().numpy()
            outs.append(emb)
        return np.vstack(outs)


def structure_aware_embedding_features(candidates: list[SequenceRecord], positives: list[SequenceRecord], cfg: dict) -> pd.DataFrame:
    enabled = cfg.get("enabled", False)
    if not enabled:
        return pd.DataFrame(
            {
                "protein_id": [c.protein_id for c in candidates],
                "structure_aware_embedding_backend": ["disabled"] * len(candidates),
                "structure_aware_distance_to_positive_centroid": [np.nan] * len(candidates),
                "structure_aware_novelty": [np.nan] * len(candidates),
                "nearest_positive_structure_aware_distance": [np.nan] * len(candidates),
            }
        )

    backend = cfg.get("backend", "prostt5")
    force_mock = cfg.get("force_mock", False)

    if force_mock or backend == "mock":
        pos_matrix = np.stack([_mock_embedding(p.sequence) for p in positives], axis=0)
        cand_matrix = np.stack([_mock_embedding(c.sequence) for c in candidates], axis=0)
        used_backend = "mock"
    else:
        try:
            rt = StructureAwareEmbeddingRuntime(
                backend=backend,
                model_name=cfg.get("model_name", "Rostlab/ProstT5"),
                cache_dir=Path(cfg.get("cache_dir", "~/.cache/mcoxplorer")).expanduser(),
                device=cfg.get("device", "cpu"),
                batch_size=int(cfg.get("batch_size", 8)),
            )
            pos_matrix = rt.embed_batch([p.sequence for p in positives])
            cand_matrix = rt.embed_batch([c.sequence for c in candidates])
            used_backend = backend
        except Exception as exc:
            LOGGER.warning("Structure-aware embedding backend '%s' unavailable (%s), fallback to mock.", backend, exc)
            pos_matrix = np.stack([_mock_embedding(p.sequence) for p in positives], axis=0)
            cand_matrix = np.stack([_mock_embedding(c.sequence) for c in candidates], axis=0)
            used_backend = "mock"

    centroid = pos_matrix.mean(axis=0)
    rows = []
    for c, emb in zip(candidates, cand_matrix):
        dist_centroid = float(np.linalg.norm(emb - centroid))
        all_d = np.linalg.norm(pos_matrix - emb.reshape(1, -1), axis=1)
        nearest_d = float(np.min(all_d)) if len(all_d) else np.nan
        novelty = min(1.0, dist_centroid / max(1.0, np.sqrt(len(emb))))
        rows.append(
            {
                "protein_id": c.protein_id,
                "structure_aware_embedding_backend": used_backend,
                "structure_aware_distance_to_positive_centroid": dist_centroid,
                "structure_aware_novelty": novelty,
                "nearest_positive_structure_aware_distance": nearest_d,
            }
        )
    return pd.DataFrame(rows)
