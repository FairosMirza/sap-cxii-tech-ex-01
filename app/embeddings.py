"""Sentence-embedding text representations: products are encoded offline, queries online.

Products and free-text queries go through the same pinned model so their vectors share one space.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EMBEDDING_CACHE = Path("artifacts/embeddings")
# Only the PyTorch safetensors weights, tokenizer and sentence-transformers configs; skips ONNX/OpenVINO/TF copies.
MODEL_FILES = ["*.json", "*.txt", "*.model", "model.safetensors", "1_Pooling/*", "2_Normalize/*"]


@dataclass(frozen=True)
class EmbeddingSpec:
    key: str
    model_id: str
    dim: int
    # Prepended to short search queries only (asymmetric retrieval), never to product texts.
    query_prefix: str = ""
    max_seq_length: int = 128


EMBEDDING_MODELS = {
    spec.key: spec for spec in (
        EmbeddingSpec("minilm", "sentence-transformers/all-MiniLM-L6-v2", 384),
        EmbeddingSpec("bge-small", "BAAI/bge-small-en-v1.5", 384,
                      query_prefix="Represent this sentence for searching relevant passages: "),
        EmbeddingSpec("mpnet", "sentence-transformers/all-mpnet-base-v2", 768),
    )
}


class TextEncoder:
    """One loaded model; `revision` is the resolved Hugging Face commit, recorded for reproducibility."""

    def __init__(self, spec: EmbeddingSpec, revision: str | None = None, device: str = "cpu",
                 local_path: str | Path | None = None) -> None:
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        path = Path(local_path) if local_path else Path(
            snapshot_download(spec.model_id, revision=revision, allow_patterns=MODEL_FILES))
        self.spec = spec
        self.path = path
        self.revision = path.name
        self.model = SentenceTransformer(str(path), device=device)
        self.model.max_seq_length = spec.max_seq_length

    @property
    def parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def encode_products(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        vectors = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        vector = self.model.encode([self.spec.query_prefix + text], normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vector[0], dtype=np.float32)


def cached_product_vectors(encoder: TextEncoder, texts: list[str]) -> tuple[np.ndarray, bool]:
    """Product embeddings and whether they came from disk (same model revision and texts)."""
    text_hash = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:12]
    path = EMBEDDING_CACHE / f"{encoder.spec.key}-{encoder.revision[:10]}-{text_hash}.npy"
    if path.exists():
        return np.load(path), True
    vectors = encoder.encode_products(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors, False
