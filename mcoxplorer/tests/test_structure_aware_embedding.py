from pathlib import Path

import numpy as np

from mcoxplorer.io.readers import SequenceRecord
from mcoxplorer.sequence.structure_aware_embedding import (
    StructureAwareEmbeddingRuntime,
    structure_aware_embedding_features,
)


def test_structure_aware_runtime_batches():
    rt = StructureAwareEmbeddingRuntime(
        backend="prostt5",
        model_name="x",
        cache_dir=Path("."),
        device="cpu",
        batch_size=2,
    )

    class Tok:
        def __call__(self, batch, return_tensors=None, truncation=None, padding=None):
            n = len(batch)
            return {"input_ids": Dummy(np.ones((n, 4))), "attention_mask": Dummy(np.ones((n, 4)))}

    class Dummy:
        def __init__(self, arr):
            self.arr = arr

        def to(self, device):
            return self

    class Model:
        def eval(self):
            pass

        def to(self, device):
            pass

        def __call__(self, **kwargs):
            n = kwargs["input_ids"].arr.shape[0]

            class Out:
                def __init__(self, n):
                    self.last_hidden_state = Arr(n)

            return Out(n)

    class Arr:
        def __init__(self, n):
            self.n = n

        def mean(self, dim=1):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.ones((self.n, 3))

    rt.tokenizer = Tok()
    rt.model = Model()
    out = rt.embed_batch(["AAA", "BBB", "CCC"])
    assert out.shape == (3, 3)


def test_structure_aware_embedding_features_disabled():
    pos = [SequenceRecord(protein_id="p1", sequence="AAAA")]
    cand = [SequenceRecord(protein_id="c1", sequence="AAAT")]
    out = structure_aware_embedding_features(cand, pos, {"enabled": False})
    assert out.loc[0, "structure_aware_embedding_backend"] == "disabled"
    assert "nearest_positive_structure_aware_distance" in out.columns


def test_structure_aware_embedding_features_mock_backend():
    pos = [SequenceRecord(protein_id="p1", sequence="AAAA"), SequenceRecord(protein_id="p2", sequence="TTTT")]
    cand = [SequenceRecord(protein_id="c1", sequence="AAAT")]
    out = structure_aware_embedding_features(
        cand,
        pos,
        {
            "enabled": True,
            "backend": "mock",
            "force_mock": True,
        },
    )
    assert out.loc[0, "structure_aware_embedding_backend"] == "mock"
    assert out.loc[0, "structure_aware_distance_to_positive_centroid"] >= 0
