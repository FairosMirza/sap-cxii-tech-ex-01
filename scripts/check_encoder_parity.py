"""Check that the ONNX Runtime encoder reproduces the previous PyTorch (sentence-transformers) encoder.

PyTorch runs in a child process: its OpenMP runtime crashes next to FAISS's (D17), and this process needs
the retrieval code, which loads FAISS. Compared on the full catalogue and the development queries only:
token IDs, product and query embeddings, the exact top-10 of both endpoints (with the score gaps behind any
reordering), ONNX top-10 pairs missing from the existing label pools, and encoding speed.
The label pools are read, never written.
Run from the repository root:  python -m scripts.check_encoder_parity
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np

from app.embeddings import EMBEDDING_MODELS, MODEL_FILES, TextEncoder, embedding_cache

SIMILAR_METHOD = {"bge-small": "C_bge", "minilm": "C_minilm"}
K = 10
WARMUP, RUNS = 5, 200
CHUNK = 1024


def snapshot(key: str) -> Path:
    """The pinned snapshot plus the PyTorch weights, which only the reference needs."""
    from huggingface_hub import snapshot_download

    spec = EMBEDDING_MODELS[key]
    return Path(snapshot_download(spec.model_id, revision=spec.revision, allow_patterns=MODEL_FILES + ["model.safetensors"]))


def latency(encode, texts: list[str]) -> dict:
    for text in texts[:WARMUP]:
        encode(text)
    timings = []
    for i in range(RUNS):
        started = time.perf_counter()
        encode(texts[i % len(texts)])
        timings.append(1000 * (time.perf_counter() - started))
    return {"median_ms": round(float(np.median(timings)), 2), "p95_ms": round(float(np.percentile(timings, 95)), 2)}


def reference(workdir: Path) -> None:
    """The previous encoder, unchanged: sentence-transformers on PyTorch, max_seq_length 128, L2-normalised."""
    import sentence_transformers
    import torch

    request = json.loads((workdir / "request.json").read_text())
    texts, queries = request["texts"], request["queries"]
    for key in request["models"]:
        spec = EMBEDDING_MODELS[key]
        model = sentence_transformers.SentenceTransformer(str(snapshot(key)), device="cpu")
        model.max_seq_length = spec.max_seq_length

        def encode_query(text: str) -> np.ndarray:
            return model.encode([spec.query_prefix + text], normalize_embeddings=True, convert_to_numpy=True)[0]

        started = time.perf_counter()
        products = model.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
        seconds = time.perf_counter() - started
        ids, lengths = [], []
        for start in range(0, len(texts), CHUNK):
            features = model.preprocess(texts[start:start + CHUNK])
            for row, mask in zip(features["input_ids"], features["attention_mask"].bool()):
                ids.append(row[mask].numpy())
                lengths.append(int(mask.sum()))
        np.savez(workdir / f"{key}.npz", products=products, queries=np.stack([encode_query(q) for q in queries]),
                 ids=np.concatenate(ids), lengths=np.array(lengths))
        (workdir / f"{key}.json").write_text(json.dumps({
            "runtime": f"torch {torch.__version__}, sentence-transformers {sentence_transformers.__version__}",
            "threads": torch.get_num_threads(), "product_encoding_seconds": round(seconds, 1),
            "query_latency": latency(encode_query, queries)}))


def token_ids(encoder: TextEncoder, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ids, lengths = [], []
    for start in range(0, len(texts), CHUNK):
        for encoding in encoder.tokenizer.encode_batch(texts[start:start + CHUNK]):
            n = sum(encoding.attention_mask)
            ids.append(np.array(encoding.ids[:n]))
            lengths.append(n)
    return np.concatenate(ids), np.array(lengths)


def agreement(a: np.ndarray, b: np.ndarray) -> dict:
    cosine = np.einsum("ij,ij->i", a, b)
    return {"rows": len(a), "min_cosine": round(float(cosine.min()), 8), "mean_cosine": round(float(cosine.mean()), 8),
            "max_abs_diff": float(f"{np.abs(a - b).max():.2e}")}


def compare_rankings(a: list[int], b: list[int], score_a, score_b) -> dict:
    """a, b: PyTorch and ONNX top-k rows; score_a, score_b: each side's full scores for the same query.

    A reordering can only happen between items whose scores differ by less than the score change itself,
    so `max_swap_gap` next to `score_noise` shows whether differences are near-ties.
    """
    union = set(a) | set(b)
    gaps = [abs(score_a[x] - score_a[y]) for x, y in zip(a, b) if x != y]
    return {"identical": a == b, "overlap": len(set(a) & set(b)),
            "score_noise": max((abs(score_a[r] - score_b[r]) for r in union), default=0.0),
            "max_swap_gap": max(gaps, default=0.0)}


def summarize(rows: list[dict]) -> dict:
    differing = [{**r, "score_noise": float(f"{r['score_noise']:.2e}"), "max_swap_gap": float(f"{r['max_swap_gap']:.2e}")}
                 for r in rows if not r["identical"]]
    return {"queries": len(rows), "identical_top10": len(rows) - len(differing),
            "min_overlap": min(r["overlap"] for r in rows), "mean_overlap": round(float(np.mean([r["overlap"] for r in rows])), 2),
            "max_score_noise": float(f"{max(r['score_noise'] for r in rows):.2e}"),
            "max_swap_gap": float(f"{max(r['max_swap_gap'] for r in rows):.2e}"),
            "pytorch_pairs_outside_pool": sum(r["pytorch_outside_pool"] for r in rows),
            "onnx_pairs_outside_pool": sum(r["onnx_outside_pool"] for r in rows),
            "differences": [{k: v for k, v in r.items() if k != "identical"} for r in differing]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(SIMILAR_METHOD), choices=list(SIMILAR_METHOD))
    parser.add_argument("--out", default="reports/encoder_parity.json")
    parser.add_argument("--reference", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reference:
        reference(Path(args.reference))
        return

    import onnxruntime

    from app.pipeline import build_store, embedding_modality
    from app.query_parser import parse_query
    from app.retrieval import SimilarityEngine
    from app.scoring import FeatureStore, score_all
    from app.semantic import SemanticSearcher
    from app.settings import load_config

    catalog = build_store(text=False).catalog
    texts = catalog.texts

    def changes(a: list[int], b: list[int], score_a, score_b) -> list[dict]:
        def item(row: int) -> dict:
            return {"id": catalog.ids[row], "name": catalog.names[row][:70],
                    "score_pytorch": round(float(score_a[row]), 8), "score_onnx": round(float(score_b[row]), 8)}
        return [{"rank": i + 1, "same_text": texts[x] == texts[y], "pytorch": item(x), "onnx": item(y)}
                for i, (x, y) in enumerate(zip(a, b)) if x != y]
    dev_products = [q["id"] for q in json.loads(Path("eval/queries.json").read_text())["queries"] if q["split"] == "dev"]
    dev_queries = json.loads(Path("eval/semantic_queries.json").read_text())["dev"]
    semantic_texts = sorted({parse_query(q).semantic_text for q in dev_queries})
    pools = {task: {q["query"]: set(q["candidates"]) for q in json.loads(Path(f"eval/pools/{task}_dev.json").read_text())["queries"]}
             for task in ("similar", "semantic")}

    Path(".cache").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="encoder-parity-", dir=".cache") as tmp:
        workdir = Path(tmp)
        (workdir / "request.json").write_text(json.dumps({"models": args.models, "texts": texts, "queries": semantic_texts}))
        subprocess.run([sys.executable, "-m", "scripts.check_encoder_parity", "--reference", tmp], check=True)
        references = {}
        for key in args.models:
            with np.load(workdir / f"{key}.npz") as data:
                references[key] = ({name: data[name] for name in data.files}, json.loads((workdir / f"{key}.json").read_text()))

    report = {
        "created": date.today().isoformat(),
        "environment": f"{platform.machine()} {platform.platform()}, Python {platform.python_version()}, CPU",
        "candidate": f"onnxruntime {onnxruntime.__version__} running the model repository's onnx/model.onnx",
        "reference": "the previous encoder: sentence-transformers on PyTorch, run in a child process",
        "scope": f"all {len(texts)} catalogue texts for tokens and product embeddings; development queries only "
                 f"({len(dev_products)} products, {len(dev_queries)} text queries) for rankings; exact search, k={K}",
        "latency_method": f"median/p95 of {RUNS} single-query encodings after {WARMUP} warm-up runs, dev query texts",
        "models": {},
    }
    for key in args.models:
        spec = EMBEDDING_MODELS[key]
        ref, ref_meta = references[key]
        encoder = TextEncoder(spec)
        started = time.perf_counter()
        products = encoder.encode_products(texts)
        onnx_seconds = time.perf_counter() - started
        query_table = {text: encoder.encode_query(text) for text in semantic_texts}
        ids, lengths = token_ids(encoder, texts)
        result = {
            "revision": spec.revision, "pooling": encoder.pooling, "max_seq_length": spec.max_seq_length,
            "pytorch_runtime": ref_meta["runtime"],
            "token_ids": {"texts": len(texts), "identical": bool(np.array_equal(lengths, ref["lengths"]) and np.array_equal(ids, ref["ids"])),
                          "truncated_at_max_seq_length": int((lengths == spec.max_seq_length).sum())},
            "product_embeddings": agreement(ref["products"], products),
            "query_embeddings": agreement(ref["queries"], np.stack([query_table[t] for t in semantic_texts])),
        }
        text_hash = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:12]
        stored_path = embedding_cache() / f"{key}-{spec.revision[:10]}-{text_hash}.npy"  # PyTorch-era cache name
        if stored_path.exists():
            stored = np.load(stored_path)
            result["stored_pytorch_vectors"] = {"reference_max_abs_diff": float(f"{np.abs(stored - ref['products']).max():.2e}"),
                                                "onnx_vs_stored": agreement(stored, products)}

        stores, tables = {}, {"pytorch": dict(zip(semantic_texts, ref["queries"])), "onnx": query_table}
        for side, vectors in (("pytorch", ref["products"]), ("onnx", products)):
            stores[side] = FeatureStore(catalog)
            stores[side].modalities[key] = embedding_modality(catalog, vectors, tables[side].__getitem__)

        config = load_config(method=SIMILAR_METHOD[key])
        engines = {side: SimilarityEngine(store, config) for side, store in stores.items()}
        similar = []
        for pid in dev_products:
            row = engines["onnx"].row_of(pid)
            tops = {side: [r for r, _ in engine.search(row, K).hits] for side, engine in engines.items()}
            scores = {side: score_all(store, row, config) for side, store in stores.items()}
            entry = {"query": pid, **compare_rankings(tops["pytorch"], tops["onnx"], scores["pytorch"], scores["onnx"])}
            for side in tops:
                entry[f"{side}_outside_pool"] = len({catalog.ids[r] for r in tops[side]} - pools["similar"][pid])
            if not entry["identical"]:
                entry["changes"] = changes(tops["pytorch"], tops["onnx"], scores["pytorch"], scores["onnx"])
            similar.append(entry)

        searchers = {side: SemanticSearcher(store, key, load_config()) for side, store in stores.items()}
        semantic = []
        for query in dev_queries:
            found = {side: searcher.search(query, K) for side, searcher in searchers.items()}
            tops = {side: [int(r) for r in result_.rows] for side, result_ in found.items()}
            union = np.array(sorted(set(tops["pytorch"]) | set(tops["onnx"])), dtype=np.int64)
            scores = {}
            for side, searcher in searchers.items():
                parsed = found[side].parsed
                values = stores[side].modalities[key].vectors[union] @ tables[side][parsed.semantic_text]
                scores[side] = dict(zip(union.tolist(), (values + searcher.bonuses(parsed, union)).tolist()))
            entry = {"query": query, **compare_rankings(tops["pytorch"], tops["onnx"], scores["pytorch"], scores["onnx"])}
            for side in tops:
                entry[f"{side}_outside_pool"] = len({catalog.ids[r] for r in tops[side]} - pools["semantic"][query])
            if not entry["identical"]:
                entry["changes"] = changes(tops["pytorch"], tops["onnx"], scores["pytorch"], scores["onnx"])
            semantic.append(entry)

        result["similar_products"] = {"method": SIMILAR_METHOD[key], **summarize(similar)}
        result["semantic_search"] = summarize(semantic)
        result["query_latency_ms"] = {f"pytorch ({ref_meta['threads']} threads)": ref_meta["query_latency"],
                                      "onnx (1 thread, serving default)": latency(TextEncoder(spec, threads=1).encode_query, semantic_texts),
                                      "onnx (all cores)": latency(encoder.encode_query, semantic_texts)}
        result["product_encoding_seconds"] = {"pytorch": ref_meta["product_encoding_seconds"], "onnx": round(onnx_seconds, 1)}
        report["models"][key] = result
        print(key, json.dumps({k: v for k, v in result.items() if k not in ("similar_products", "semantic_search")}), flush=True)
        for task in ("similar_products", "semantic_search"):
            print(f"  {task}:", json.dumps({k: v for k, v in result[task].items() if k != "differences"}), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
