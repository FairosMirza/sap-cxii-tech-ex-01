"""Versioned serving artifacts: derived vectors, indexes and model files, tied to the dataset by checksum.

The catalogue itself is rebuilt from the raw LDJSON at startup (parsing is code, not data); everything
expensive or model-dependent is built offline here and only loaded by the service.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse

from app.ann import HNSWIndex, HNSWParams
from app.data import DEFAULT_DATA_FILE, load_records
from app.embeddings import EMBEDDING_MODELS, ONNX_MODEL, TextEncoder
from app.features import build_catalog
from app.retrieval import SimilarityEngine
from app.scoring import Anchors, DenseModality, FeatureStore
from app.semantic import SemanticSearcher
from app.settings import DEFAULT_CONFIG, load_config, load_serving

MANIFEST = "manifest.json"
FORMAT_VERSION = 2  # 2: models stored as ONNX for ONNX Runtime (1 held PyTorch weights)


class ArtifactError(RuntimeError):
    """Artifacts are missing, malformed, or were built from a different dataset."""


def _git_commit() -> str | None:
    """HEAD, marked -dirty when code or config differs from it; GIT_COMMIT where there is no repository (Docker)."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        changed = subprocess.run(["git", "status", "--porcelain", "--", "app", "configs", "scripts", "requirements*.txt"],
                                 capture_output=True, text=True, check=True).stdout.strip()
        return f"{head}-dirty" if changed else head
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GIT_COMMIT") or None


def required_models(config_path: Path = DEFAULT_CONFIG) -> list[str]:
    serving = load_serving(config_path)
    text_model = load_config(config_path, serving["similar_method"]).text_model
    return sorted({serving["semantic_model"], text_model} - {"tfidf"})


def build(out: Path, config_path: Path = DEFAULT_CONFIG, data_file: Path = DEFAULT_DATA_FILE,
          params: HNSWParams = HNSWParams()) -> dict:
    """Write TF-IDF vectors, embeddings, HNSW indexes, model files and the manifest into `out`."""
    import faiss
    import onnxruntime

    from app.embeddings import cached_product_vectors
    from app.pipeline import embedding_modality, tfidf_modality

    records, report = load_records(data_file)
    catalog = build_catalog(records)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "format": FORMAT_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": _git_commit(),
        "dataset": {"file": str(data_file), "sha256": report.sha256, "rows": report.rows},
        "serving": load_serving(config_path),
        "representations": {},
    }
    tfidf = tfidf_modality(catalog)
    sparse.save_npz(out / "tfidf.npz", tfidf.vectors)
    manifest["representations"]["tfidf"] = {"kind": "tfidf", "file": "tfidf.npz",
                                            "dimensions": int(tfidf.vectors.shape[1]), "anchors": asdict(tfidf.anchors)}
    faiss.omp_set_num_threads(1)  # single-threaded insertion gives a reproducible graph
    for key in required_models(config_path):
        spec = EMBEDDING_MODELS[key]
        encoder = TextEncoder(spec)
        vectors, _ = cached_product_vectors(encoder, catalog.texts)
        modality = embedding_modality(catalog, vectors)
        np.save(out / f"{key}.npy", modality.vectors)
        started = time.perf_counter()
        index = HNSWIndex.build(modality.vectors, params)
        build_seconds = time.perf_counter() - started
        index.save(out / f"{key}.hnsw")
        model_dir = out / "models" / key
        shutil.rmtree(model_dir, ignore_errors=True)
        encoder.copy_model_files(model_dir)
        manifest["representations"][key] = {
            "kind": "embedding", "model_id": spec.model_id, "revision": spec.revision, "file": f"{key}.npy",
            "dimensions": spec.dim, "anchors": asdict(modality.anchors), "model_dir": f"models/{key}",
            "encoder": {"runtime": f"onnxruntime {onnxruntime.__version__}", "model_file": ONNX_MODEL,
                        "pooling": encoder.pooling, "max_seq_length": spec.max_seq_length,
                        "query_prefix": spec.query_prefix, "normalize": "l2"},
            "index": {"file": f"{key}.hnsw", "type": "faiss.IndexHNSWFlat, inner product", "params": asdict(params),
                      "build_seconds": round(build_seconds, 2), "bytes": index.memory_bytes},
        }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load(artifacts: Path, data_file: Path = DEFAULT_DATA_FILE, encoder_threads: int = 1) -> tuple[FeatureStore, dict]:
    """Rebuild the catalogue and attach the stored representations; raise ArtifactError if anything is off."""
    path = artifacts / MANIFEST
    if not path.exists():
        raise ArtifactError(f"no manifest at {path}; run `python -m scripts.prepare` first")
    manifest = json.loads(path.read_text())
    if manifest.get("format") != FORMAT_VERSION:
        raise ArtifactError(f"unsupported artifact format {manifest.get('format')!r}")
    records, report = load_records(data_file)
    if report.sha256 != manifest["dataset"]["sha256"]:
        raise ArtifactError("the dataset does not match the one the artifacts were built from")
    store = FeatureStore(build_catalog(records))
    n = len(store.catalog)
    for key, rep in manifest["representations"].items():
        anchors = Anchors(**rep["anchors"])
        if rep["kind"] == "tfidf":
            vectors = sparse.load_npz(artifacts / rep["file"]).tocsr()
            store.modalities[key] = DenseModality(vectors, np.asarray(vectors.getnnz(axis=1) > 0), anchors)
        else:
            spec = EMBEDDING_MODELS[key]
            if rep["revision"] != spec.revision:
                raise ArtifactError(f"{key} was built from revision {rep['revision']}, the code pins {spec.revision}")
            vectors = np.load(artifacts / rep["file"])
            encoder = TextEncoder(spec, local_path=artifacts / rep["model_dir"], threads=encoder_threads)
            store.modalities[key] = DenseModality(vectors, np.linalg.norm(vectors, axis=1) > 0, anchors, encoder.encode_query)
            index = HNSWIndex.load(artifacts / rep["index"]["file"], HNSWParams(**rep["index"]["params"]))
            if index.size != n:
                raise ArtifactError(f"index {key} holds {index.size} vectors, catalogue has {n}")
            store.indexes[key] = index
        if store.modalities[key].vectors.shape[0] != n:
            raise ArtifactError(f"representation {key} has {store.modalities[key].vectors.shape[0]} rows, catalogue has {n}")
    return store, manifest


def load_service() -> tuple[SimilarityEngine, SemanticSearcher, dict]:
    """The served engines over the prepared artifacts, set by CONFIG_PATH, ARTIFACTS_DIR and ENCODER_THREADS."""
    config_path = Path(os.environ.get("CONFIG_PATH", str(DEFAULT_CONFIG)))
    # One encoder thread per request by default: concurrency comes from parallel requests, not from one query.
    threads = int(os.environ.get("ENCODER_THREADS", "1"))
    store, manifest = load(Path(os.environ.get("ARTIFACTS_DIR", "artifacts")), encoder_threads=threads)
    serving = load_serving(config_path)
    engine = SimilarityEngine(store, load_config(config_path, serving["similar_method"]))
    searcher = SemanticSearcher(store, serving["semantic_model"], load_config(config_path))
    return engine, searcher, manifest
