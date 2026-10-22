"""Smoke-test a running service over HTTP: both endpoints in both search modes, filters and error handling.

Records client-side latency (median/p95 per endpoint and mode) and every returned ID list, so two
deployments (for example macOS and the Linux container) can be compared request by request.
Run from the repository root:  python -m scripts.smoke_api --base-url http://127.0.0.1:8000 --out <file>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np

K = 10
WARMUP, RUNS = 5, 50


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare", help="an earlier --out file whose returned IDs should be compared")
    parser.add_argument("--wait", type=float, default=180, help="seconds to wait for /ready")
    args = parser.parse_args()

    products = [q["id"] for q in json.loads(Path("eval/queries.json").read_text())["queries"] if q["split"] == "dev"]
    queries = json.loads(Path("eval/semantic_queries.json").read_text())["dev"]
    failures: list[str] = []

    def check(ok: bool, message: str) -> None:
        if not ok:
            failures.append(message)

    with httpx.Client(base_url=args.base_url, timeout=60) as client:
        deadline = time.monotonic() + args.wait
        while True:
            try:
                if client.get("/ready").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                sys.exit(f"{args.base_url} did not become ready within {args.wait:.0f}s")
            time.sleep(1)
        ready = client.get("/ready").json()
        check(client.get("/health").status_code == 200, "/health")

        results: dict[str, dict[str, list[str]]] = {}
        for mode in ("exact", "ann"):
            results[f"similar/{mode}"] = {}
            for pid in products:
                response = client.get("/find_similar_products", params={"product_id": pid, "num_similar": K, "mode": mode})
                ids = response.json() if response.status_code == 200 else []
                check(len(ids) == K == len(set(ids)) and pid not in ids, f"similar/{mode} {pid}: HTTP {response.status_code}, {len(ids)} ids")
                results[f"similar/{mode}"][pid] = ids
            results[f"semantic/{mode}"] = {}
            for query in queries:
                response = client.get("/semantic_search", params={"query": query, "num_results": K, "mode": mode})
                ids = response.json() if response.status_code == 200 else []
                check(response.status_code == 200 and len(ids) <= K and len(ids) == len(set(ids)), f"semantic/{mode} {query!r}: HTTP {response.status_code}")
                results[f"semantic/{mode}"][query] = ids
        for pid in products[:5]:
            response = client.get("/find_similar_products", params={"product_id": pid, "num_similar": K, "max_price": 1000})
            check(response.status_code == 200 and len(response.json()) <= K, f"max_price filter {pid}: HTTP {response.status_code}")
            response = client.get("/find_similar_products", params={"product_id": pid, "num_similar": K, "same_brand": True})
            check(response.status_code in (200, 422), f"same_brand filter {pid}: HTTP {response.status_code}")
        for path, params, expected in [
            ("/find_similar_products", {"product_id": "not-hex", "num_similar": 5}, 422),
            ("/find_similar_products", {"product_id": "0" * 32, "num_similar": 5}, 404),
            ("/find_similar_products", {"product_id": products[0], "num_similar": 0}, 422),
            ("/find_similar_products", {"product_id": products[0], "num_similar": 101}, 422),
            ("/find_similar_products", {"product_id": products[0], "num_similar": 5, "mode": "fast"}, 422),
            ("/semantic_search", {"query": "", "num_results": 5}, 422),
            ("/semantic_search", {"query": "x" * 201, "num_results": 5}, 422),
            ("/semantic_search", {"query": "shoes", "num_results": 0}, 422),
        ]:
            status = client.get(path, params=params).status_code
            check(status == expected, f"{path} {params} -> HTTP {status}, expected {expected}")

        latency = {}
        for name, path, field, items, size in [("similar", "/find_similar_products", "product_id", products, "num_similar"),
                                               ("semantic", "/semantic_search", "query", queries, "num_results")]:
            for mode in ("exact", "ann"):
                timings = []
                for i in range(WARMUP + RUNS):
                    started = time.perf_counter()
                    client.get(path, params={field: items[i % len(items)], size: K, "mode": mode})
                    if i >= WARMUP:
                        timings.append(1000 * (time.perf_counter() - started))
                latency[f"{name}/{mode}"] = {"median": round(float(np.median(timings)), 2),
                                             "p95": round(float(np.percentile(timings, 95)), 2)}

    overlap = {task: round(float(np.mean([len(set(results[f"{task}/exact"][q]) & set(results[f"{task}/ann"][q]))
                                          / max(len(results[f"{task}/exact"][q]), 1) for q in results[f"{task}/exact"]])), 4)
               for task in ("similar", "semantic")}
    report = {"base_url": args.base_url, "ready": ready, "failures": failures,
              "latency_ms": {"method": f"client-side HTTP, median/p95 of {RUNS} requests after {WARMUP} warm-up, k={K}", **latency},
              "ann_top10_overlap_with_exact": overlap, "results": results}
    if args.compare:
        other = json.loads(Path(args.compare).read_text())["results"]
        report["comparison"] = {key: {"identical_lists": sum(ids == other[key].get(q) for q, ids in lists.items()), "of": len(lists)}
                                for key, lists in results.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
