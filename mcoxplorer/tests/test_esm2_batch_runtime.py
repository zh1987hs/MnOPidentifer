from pathlib import Path
import numpy as np

from mcoxplorer.sequence.features import ESM2EmbeddingRuntime


def test_esm2_runtime_batches(monkeypatch):
    rt = ESM2EmbeddingRuntime(model_name="x", cache_dir=Path("."), device="cpu", batch_size=2)

    class Tok:
        def __call__(self, batch, return_tensors=None, truncation=None, padding=None):
            import numpy as np
            n = len(batch)
            return {"input_ids": Dummy(np.ones((n, 4))), "attention_mask": Dummy(np.ones((n, 4)))}

    class Dummy:
        def __init__(self, arr): self.arr = arr
        def to(self, device): return self

    class Model:
        def eval(self): pass
        def to(self, device): pass
        def __call__(self, **kwargs):
            n = kwargs["input_ids"].arr.shape[0]
            class O:
                def __init__(self, n): self.last_hidden_state = Arr(n)
            return O(n)

    class Arr:
        def __init__(self, n): self.n = n
        def mean(self, dim=1): return self
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return np.ones((self.n, 3))

    rt.tokenizer = Tok()
    rt.model = Model()
    out = rt.embed_batch(["AAA", "BBB", "CCC"])
    assert out.shape == (3, 3)
