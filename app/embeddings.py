"""Sentence-embedding text representations: products are encoded offline, queries online.

Products and free-text queries go through the same pinned model and the same encoder, so their vectors
share one space. The model runs on ONNX Runtime from the model repository's own ONNX export, and the rest
of the sentence-transformers pipeline (tokenizer, truncation, pooling, L2 normalisation) is reproduced from
the model's config files. PyTorch is not used: its bundled OpenMP runtime crashes next to FAISS's (D17).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ONNX_MODEL = "onnx/model.onnx"
# Tokenizer, sentence-transformers configs and the fp32 ONNX export; skips PyTorch/TF/OpenVINO/quantised copies.
MODEL_FILES = ["*.json", "*.txt", ONNX_MODEL, "1_Pooling/*", "2_Normalize/*"]
POOLING_MODES = {"pooling_mode_cls_token": "cls", "pooling_mode_mean_tokens": "mean"}


@dataclass(frozen=True)
class EmbeddingSpec:
    key: str
    model_id: str
    revision: str  # pinned Hugging Face commit
    dim: int
    # Prepended to short search queries only (asymmetric retrieval), never to product texts.
    query_prefix: str = ""
    max_seq_length: int = 128


EMBEDDING_MODELS = {
    spec.key: spec for spec in (
        EmbeddingSpec("minilm", "sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", 384),
        EmbeddingSpec("bge-small", "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", 384,
                      query_prefix="Represent this sentence for searching relevant passages: "),
        EmbeddingSpec("mpnet", "sentence-transformers/all-mpnet-base-v2", "e8c3b32edf5434bc2275fc9bab85f82640a19130", 768),
    )
}


def embedding_cache() -> Path:
    return Path(os.environ.get("EMBEDDING_CACHE_DIR", ".cache/embeddings"))


def download(spec: EmbeddingSpec) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(spec.model_id, revision=spec.revision, allow_patterns=MODEL_FILES))


def read_pooling(path: Path, dim: int) -> str:
    """The pooling mode from the model's own sentence-transformers config; anything not reproduced here is rejected."""
    modules = json.loads((path / "modules.json").read_text())
    kinds = [module["type"].rsplit(".", 1)[-1] for module in modules]
    if kinds[:2] != ["Transformer", "Pooling"] or not set(kinds[2:]) <= {"Normalize"}:
        raise ValueError(f"unsupported sentence-transformers modules {kinds} in {path}")
    config = json.loads((path / modules[1]["path"] / "config.json").read_text())
    enabled = [key for key, on in config.items() if key.startswith("pooling_mode_") and on]
    if len(enabled) != 1 or enabled[0] not in POOLING_MODES:
        raise ValueError(f"unsupported pooling {enabled} in {path}")
    if config.get("word_embedding_dimension") != dim:
        raise ValueError(f"model dimension {config.get('word_embedding_dimension')} does not match the spec ({dim})")
    return POOLING_MODES[enabled[0]]


def pool(tokens: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    """sentence-transformers pooling: the [CLS] token, or the mean over real (unpadded) tokens."""
    if mode == "cls":
        return tokens[:, 0]
    weights = mask[..., None].astype(tokens.dtype)
    return (tokens * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


class TextEncoder:
    """One pinned model on ONNX Runtime. `threads=0` lets ONNX Runtime use every physical core."""

    def __init__(self, spec: EmbeddingSpec, local_path: str | Path | None = None, threads: int = 0) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        path = Path(local_path) if local_path else download(spec)
        self.spec, self.path, self.revision = spec, path, spec.revision
        self.pooling = read_pooling(path, spec.dim)
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        if json.loads((path / "sentence_bert_config.json").read_text()).get("do_lower_case"):
            normalizer = self.tokenizer.normalizer
            if normalizer is None or normalizer.normalize_str("A") != "a":
                raise ValueError(f"{spec.model_id} expects lowercased input but its tokenizer does not lowercase")
        # Like sentence-transformers: pad to the batch's longest text, cut at max_seq_length (overrides tokenizer.json).
        self.tokenizer.enable_truncation(spec.max_seq_length)
        pad = json.loads((path / "tokenizer_config.json").read_text()).get("pad_token", "[PAD]")
        pad = pad["content"] if isinstance(pad, dict) else pad
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id(pad), pad_token=pad)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(path / ONNX_MODEL), options, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}

    def _embed(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feeds = {"input_ids": np.array([e.ids for e in encodings], dtype=np.int64), "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feeds["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
        tokens = self.session.run(["last_hidden_state"], feeds)[0]
        return l2_normalize(pool(tokens, mask, self.pooling)).astype(np.float32)

    def copy_model_files(self, target: Path) -> None:
        """Copy only the files the model needs at runtime (resolving Hugging Face cache symlinks)."""
        for source in self.path.rglob("*"):
            relative = source.relative_to(self.path).as_posix()
            if source.is_file() and any(fnmatch.fnmatch(relative, pattern) for pattern in MODEL_FILES):
                (target / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target / relative)

    def encode_products(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vectors = np.empty((len(texts), self.spec.dim), dtype=np.float32)
        order = np.argsort([-len(text) for text in texts], kind="stable")  # similar lengths per batch: less padding
        for start in range(0, len(texts), batch_size):
            rows = order[start:start + batch_size]
            vectors[rows] = self._embed([texts[i] for i in rows])
        return vectors

    def encode_query(self, text: str) -> np.ndarray:
        return self._embed([self.spec.query_prefix + text])[0]


def cached_product_vectors(encoder: TextEncoder, texts: list[str]) -> tuple[np.ndarray, bool]:
    """Product embeddings and whether they came from disk (same model revision, runtime, truncation and texts)."""
    text_hash = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:12]
    spec = encoder.spec
    path = embedding_cache() / f"{spec.key}-{spec.revision[:10]}-onnx-{spec.max_seq_length}-{text_hash}.npy"
    if path.exists():
        return np.load(path), True
    vectors = encoder.encode_products(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors, False
