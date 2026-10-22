"""Run the semantic search on the sample queries and record parsed constraints and top results.

For manual sanity checks only; relevance is judged later with blind human labels.
Run from the repository root:  python -m scripts.run_sample_queries --model minilm
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from app.embeddings import EMBEDDING_MODELS, TextEncoder, cached_product_vectors
from app.features import COLOR_FAMILIES
from app.pipeline import build_store, embedding_modality
from app.semantic import SemanticSearcher
from app.settings import load_config

AUDIENCES = {1: "men", 2: "women", 4: "boys", 8: "girls", 16: "baby"}


def describe(parsed) -> dict:
    info = asdict(parsed)
    info["audience"] = [name for bit, name in AUDIENCES.items() if parsed.audience & bit]
    info["colors"], info["materials"] = sorted(parsed.colors), sorted(parsed.materials)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="minilm", choices=list(EMBEDDING_MODELS))
    parser.add_argument("--queries", default="eval/sample_queries.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default=None, help="default: reports/semantic_samples_<model>.json")
    args = parser.parse_args()
    args.out = args.out or f"reports/semantic_samples_{args.model}.json"

    store = build_store(text=False)
    c = store.catalog
    encoder = TextEncoder(EMBEDDING_MODELS[args.model])
    vectors, _ = cached_product_vectors(encoder, c.texts)
    store.modalities[args.model] = embedding_modality(c, vectors)
    searcher = SemanticSearcher(store, args.model, encoder.encode_query, load_config())

    report = {"model": encoder.spec.model_id, "revision": encoder.revision, "k": args.k, "results": []}
    for query in json.loads(Path(args.queries).read_text())["queries"]:
        result = searcher.search(query, args.k)
        hits = []
        for row, score in zip(result.rows, result.scores):
            colours = [COLOR_FAMILIES[j] for j in np.flatnonzero(c.color_matrix[row])]
            categories = [c.category_names[j] for j in c.category_matrix[row].indices]
            hits.append({"id": c.ids[row], "score": round(score, 4), "name": c.names[row],
                         "price": None if np.isnan(c.price[row]) else float(c.price[row]),
                         "categories": categories, "colours": colours})
        report["results"].append({"query": query, "parsed": describe(result.parsed), "eligible": result.eligible,
                                  "excluded_by_filter": result.excluded, "hits": hits})
        print(f"\n{query}\n  parsed: price={result.parsed.price and (result.parsed.price.minimum, result.parsed.price.maximum)}"
              f" pref={result.parsed.price_preference} audience={describe(result.parsed)['audience']} "
              f"colors={sorted(result.parsed.colors)} sizes={result.parsed.sizes} | eligible={result.eligible}")
        for hit in hits:
            print(f"  {hit['score']:.3f} ₹{hit['price']} {hit['name'][:70]} {hit['categories'][:1]}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
