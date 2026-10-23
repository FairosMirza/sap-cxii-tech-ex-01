# Product similarity search: SAP CX technical exercise

Given a product's `uniq_id`, return similar products from the Amazon Fashion 2020 catalogue (30,000 products), and
answer free-text searches such as "red lehenga choli for wedding" with filters read from the query. One codebase
covers the [original exercise](https://github.com/punitmishra/sap-cxii-tech-ex-01) (kept verbatim in
[docs/assignment.md](docs/assignment.md)) and its [v2 specification](https://github.com/andrijaperovic/sap-cxii-tech-ex-01/tree/v2-update).

Every decision, with the evidence and the alternatives rejected, is in [docs/decisions.md](docs/decisions.md)
(D01–D17). The one-page trade-off note (v2 Part 6) is [docs/tradeoff_note.md](docs/tradeoff_note.md).

## Two submissions

| | Core submission | Experimental submission |
|---|---|---|
| What | The service as it runs: hybrid similarity, semantic search, approximate search | Image features (multimodal bonus) and semantic IDs |
| Status | Frozen at commit `77372d7` ([eval/freeze.json](eval/freeze.json)), selected on development labels, assessed once on an untouched holdout | Pre-registered before running (D14, D15). Semantic IDs: tested, not adopted. Images: development labels in progress |
| Served | Yes | No. Adoption would need new held-out queries, because the holdout is spent (D07) |

Everything below describes the core submission unless it says otherwise.

## Quickstart

**Docker** (recommended; tested with Docker 29 on Colima, Linux arm64):

```bash
docker build -t product-similarity .
docker run --rm -p 8000:8000 product-similarity
curl -s localhost:8000/ready        # 200 once the artifacts are loaded, a few seconds after start
```

The build needs network access and takes about 10 minutes from scratch: it installs the pinned dependencies,
downloads the pinned bge-small model, encodes all 30,000 products and builds the FAISS index. From a git clone,
add `--build-arg GIT_COMMIT=$(git rev-parse HEAD)` to report the commit in `/ready`.

**Local Python 3.12** (the model download and the embedding cache stay under `.cache/` in the repository):

```bash
python3.12 -m zipfile -e data/archive.zip data/
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-build.txt
.venv/bin/python -m scripts.prepare        # about 2 minutes: writes artifacts/
.venv/bin/uvicorn app:app --port 8000
```

**Kubernetes on kind** ([deploy/kubernetes.yaml](deploy/kubernetes.yaml): 2 replicas, startup, readiness and
liveness probes, non-root, read-only root filesystem, CPU and memory limits):

```bash
docker build -t product-similarity:1.1.0 .
kind create cluster --name similarity
kind load docker-image product-similarity:1.1.0 --name similarity
kubectl apply -f deploy/kubernetes.yaml
kubectl -n product-similarity rollout status deployment/product-similarity
kubectl -n product-similarity port-forward service/product-similarity 8000:80
```

## API

| Endpoint | Parameters | Returns |
|---|---|---|
| `GET /find_similar_products` | `product_id` (32 hex characters), `num_similar` (1–100); optional `same_brand`, `min_price`, `max_price` (rupees), `mode` (`exact` or `ann`) | Product IDs, most similar first; exactly `num_similar` unless fewer products are eligible |
| `GET /semantic_search` | `query` (1–200 characters), `num_results` (1–100); optional `mode` | Product IDs, most relevant first; at most `num_results`, fewer when filters or the relevance cutoff leave fewer |
| `GET /parse_query` | `query` | The price, audience, colour, material and size constraints read from the query |
| `GET /health`, `GET /ready` | | Liveness; readiness (503 until the artifacts are loaded, then the dataset checksum, serving configuration and commit) |

Errors: 404 for an unknown product; 422 for invalid parameters (malformed ID, count out of range, unknown mode,
`min_price` above `max_price`, `same_brand` for a product without a known brand); 503 while not ready; 500 with a
request ID for anything unexpected. Every response carries an `x-request-id` header.

Examples, as returned by the frozen configuration:

```bash
curl -s "localhost:8000/find_similar_products?product_id=ab6a6605b0a017c3ca05f793b2ec4cd1&num_similar=3"
# ["2998128e6877dd3fc96fa203ca63dabb","a4b8c19421d8ae0010269161fcaa174e","3e0dd6fa257554220d4e3fce121edcea"]
# Query: Juniper Women's Straight Kurta (₹599). Results: three other Juniper straight kurtas (₹599–799).

curl -s "localhost:8000/find_similar_products?product_id=ab6a6605b0a017c3ca05f793b2ec4cd1&num_similar=3&max_price=500"
# W for Woman straight kurtas at ₹454–494.

curl -s "localhost:8000/semantic_search?query=red%20lehenga%20choli%20for%20wedding&num_results=3"
# Three embroidered lehengas (₹1,299 and ₹6,999; one listing has no price).

curl -s "localhost:8000/semantic_search?query=comfortable%20running%20shoes%20under%20%24100&num_results=5"
# Four results, not five: gloves and bra pads under ₹100. Amounts are read as rupees (D11), the catalogue holds
# only 2 running shoes (D13), and the relevance cutoff drops the rest (D12).

curl -s "localhost:8000/parse_query?query=cotton%20kurti%20for%20women%20around%201200%20rupees"
# price 960–1440 (around = ±20%), audience women, material cotton; "cotton kurti for women" is what gets embedded.
```

The Part 1 function returns exactly what the endpoint returns (run `scripts.prepare` first):

```python
from app.retrieval import find_similar_products
find_similar_products("ab6a6605b0a017c3ca05f793b2ec4cd1", 5)
```

## Approach

- **Data** (D03): weight placeholders, normalised brands, valid prices, colours read from titles, categories from
  the sales-rank map or the browse node. Missing values stay missing; missing never matches missing.
- **Similar products** (D01, D04–D06): a plausible substitute for the query product. The score is an evidence-aware
  weighted average of a text block (bge-small cosine, calibrated) and an attribute block (category 0.40, price
  0.20, brand 0.15, colour 0.15, weight 0.10; price and weight on a log scale), with block weights 0.5 and 0.5.
  A component missing for either product drops out for that pair. The query's own ASIN and variant family are
  excluded, with at most one listing per family. Ties are broken by rating, review count and product ID.
- **Text model** (D09, D17): bge-small-en-v1.5 (384 dimensions, 33 M parameters), pinned to a Hugging Face commit
  and run with ONNX Runtime; the serving image has no PyTorch.
- **Approximate search** (D08): FAISS HNSW (Malkov & Yashunin, 2018) proposes candidates, 800 for similar products
  and 200 for search, and the exact scorer re-ranks them. `mode=exact` scores every product.
- **Semantic search** (D10–D12): a rule-based parser reads price limits ("under", "up to", "around", ranges,
  "under a hundred bucks"), audience, colour, material and size. Filters apply first; products are then ranked by
  cosine similarity plus small bonuses, and a relevance cutoff learned from development labels (0.5593) drops
  clearly unrelated results.
- **Service** (D16): FastAPI with plain handlers run in the thread pool (the work is CPU-bound), input validation
  and request logging; a two-stage Docker image that runs as a non-root user with a readiness health check.

## Results

**Method** (D07). Blind, pooled human judgements, graded 0–3, with nDCG@10 and P@5 as metrics. The selection rule
was fixed before any metric was computed; 95% intervals are bootstrapped over queries, and comparisons use paired
differences. Selection used the development queries (20 products, 10 searches). The holdout (10 products, 10
searches) was labelled after the configuration was frozen and evaluated once. Judgements: 844 development and 417
holdout, from a single labeller.

**Development (selection).** P@5 is shown over the scored queries and over all queries, counting queries without
any relevant product as 0.

| Similar products, 20 queries | nDCG@10 | P@5 | | Search, 8 of 10 scored | nDCG@10 | P@5 (scored / all) |
|---|---|---|---|---|---|---|
| **C_bge** (chosen) | 0.760 | 0.82 | | **bge-small** (chosen) | 0.747 | 0.78 / 0.62 |
| C_minilm | 0.749 | 0.76 | | TF-IDF | 0.601 | 0.68 / 0.54 |
| B_tfidf | 0.707 | 0.78 | | MiniLM | 0.564 | 0.55 / 0.44 |
| A_structured (attributes only) | 0.497 | 0.63 | | | | |

**Holdout (assessment of the frozen selection).**

| Similar products, 9 of 10 scored | nDCG@10 | P@5 (scored / all) | | Search, 10 scored | nDCG@10 | P@5 |
|---|---|---|---|---|---|---|
| **C_bge** (chosen) | 0.743 | 0.62 / 0.56 | | **bge-small** (chosen) | 0.758 | 0.82 |
| C_minilm | 0.665 | 0.62 / 0.56 | | MiniLM | 0.778 | 0.76 |
| B_tfidf | 0.636 | 0.67 / 0.60 | | TF-IDF | 0.724 | 0.76 |

- **Similar products:** C_bge leads C_minilm by 0.077 nDCG@10 (95% interval 0.006 to 0.170) and B_tfidf by 0.107
  (0.008 to 0.200). P@5 separates no pair. This is encouraging evidence with limited generalisability: 9 scored
  queries. The tenth, the catalogue's only book, has no relevant product.
- **Search:** no pair of models is separated on the holdout, and bge-small's development lead over MiniLM did not
  replicate. Selection happened on development data; the holdout assessed it and does not change it (D09).
- **Deployed versus exact search:** identical nDCG@10 and P@5 on every scored holdout query. On 1,000 sampled
  products, approximate search returns 99.1% of the exact top-10 (D08).
- **Latency, measured:** in process, similar products take 1.5 ms (approximate) against 6.3 ms (exact), median on
  Apple silicon. Over HTTP on macOS (client side, k = 10), similar products take 3.2 ms against 6.9 ms and search
  6.1 ms against 8.1 ms, median.

**Verified deployments** (23 Oct).
- **Clean Docker build** of the committed tree, with no build cache: 10.5 minutes. The image is 342 MB compressed
  (810 MB unpacked, of which data 71 MB and artifacts 227 MB) and runs as a non-root user; it uses 560 MiB after the
  smoke test, and its serving code is identical to the frozen commit. The
  container rebuilt its own artifacts and returned the same 60 result lists as the frozen artifacts (both endpoints,
  both search modes).
- **kind** (Kubernetes 1.37): both replicas were ready within 6 seconds, with the read-only root filesystem and all
  capabilities dropped. The same 60 lists came back identical, and the Service spread 40 in-cluster requests over
  both pods (16 and 24).

## Experimental submission

Pre-registered in the decision log before anything was computed. The code is in `experiments/`, with results in
`reports/experiments/`; none of it is served.

- **Semantic IDs** (D15, `python -m experiments.semantic_ids`). Residual k-means codes over the product vectors
  (3 levels of 64) serve as a candidate index, compared with HNSW under the D08 benchmark's conditions. *Not
  adopted.* They build in 0.18 s instead of 6.2 s and take 0.9 MB instead of an 8.2 MB graph, but at the deployed
  candidate counts they cover far less of the exact top-10: 0.789 against 0.991 for similar products, and 0.438
  against 0.991 for title searches. At equal latency HNSW still wins (0.991 in 1.71 ms against 0.944 in 1.85 ms).
- **Image features** (D14, `python -m experiments.image_features`, then `experiments.image_experiment`). ResNet-50
  vectors for all 29,861 usable images (38 minutes on CPU), and C_bge with an image block of weight 0.1, 0.2 or
  0.3. The 56 new pairs these configurations retrieve for the development products are being judged blind; the
  pre-registered rule then decides. Development evidence only, because the holdout is spent.

## Limitations

- **Small, single-labeller evaluation:** 20 + 10 development and 10 + 10 holdout queries, judged by one person.
  The intervals are wide, and on the holdout the search models cannot be separated.
- **Catalogue:** 17 footwear products, no wallets and no laptop bags. Some listings are out of domain or misfiled
  (a garden plant filed as caps, a book), and 9.6% have no price and 79% no weight (D03, D13). Results for such
  queries reflect the catalogue, not ranking errors.
- **Currency:** query amounts are read as rupees, so "under $100" means ₹100 (D11).
- **Search errors** (holdout error analysis):
  - Product type can lose to colour and audience: "navy blue polo t-shirt for men" returns navy shirts that are
    not polos.
  - A query with no suitable product still gets results; the cutoff does not empty them.
- **No product descriptions:** the dataset has none, so product text is the title plus keywords.
- **Holdout spent:** any further change needs new held-out queries.

## Reproduce

```bash
.venv/bin/pip install -r requirements-dev.txt     # tests and research scripts (adds PyTorch for the parity reference)
.venv/bin/python -m pytest                        # 262 tests (3 expected failures document D06); real-artifact tests skip until scripts.prepare has run
.venv/bin/python -m scripts.benchmark_ann         # reports/ann_benchmark.json (D08)
.venv/bin/python -m scripts.check_labels --rankings   # validates eval/labels.jsonl, writes eval/qrels_dev.json
.venv/bin/python -m scripts.evaluate              # reports/dev_evaluation.json
.venv/bin/python -m scripts.evaluation_checks     # reports/evaluation_checks.json (all-query P@5, every pair's interval)
.venv/bin/python -m scripts.smoke_api --base-url http://127.0.0.1:8000 --out smoke.json   # against a running service
```

The holdout was evaluated once (`scripts.evaluate --split holdout`, which refuses to overwrite
`reports/holdout_evaluation.json`), with the code, configuration, artifacts and packages fingerprinted in
`eval/freeze.json` by `scripts.freeze`. Anything that re-ranks the holdout first checks those fingerprints; artifacts
rebuilt on another platform can differ in the last floating-point bits and fail the check. The committed reports
are the record.

## Repository

| Path | Contents |
|---|---|
| `app/` | Data parsing, features, scoring, approximate search, query parser, semantic search, HTTP service |
| `scripts/` | Artifact preparation, benchmarks, label pools, labelling page, label checks, evaluation, freeze, smoke test |
| `configs/default.yaml` | Scoring weights, eligibility rules, compared methods and the serving configuration |
| `deploy/kubernetes.yaml` | Namespace, Deployment and Service for kind |
| `docs/` | Decision log, trade-off note, original assignment |
| `eval/` | Query sets, blind pools, labels, merged judgements, freeze record |
| `reports/` | Every number quoted here and in the decision log |
| `tests/` | Unit, API and real-artifact integration tests |

## Requirements coverage

| Requirement | Where |
|---|---|
| Part 1: `find_similar_products` | `app/retrieval.py`; D01, D05 |
| Part 2: FastAPI service, Dockerfile, optional Kubernetes | `app/api.py`, `Dockerfile`, `deploy/kubernetes.yaml`; D16 |
| v2 Part 3: embeddings fused with normalised numeric features | D05 (fusion maths and normalisation), D09 (model: corpus, dimensions, latency, alternatives) |
| Part 4 (bonus): approximate search, filtered search | `app/ann.py` (HNSW reference), `same_brand` / `min_price` / `max_price`; D08 |
| v2 Part 5: `/semantic_search` with parsed filters | `app/query_parser.py`, `app/semantic.py`; D10–D12 |
| v2 Part 6: trade-off note | `docs/tradeoff_note.md` |
| Multimodal similarity (bonus) | Experimental submission; the scorer already supports a weighted image block with a per-pair fallback (D14) |
| Suggested: dimensionality reduction | Not used: the 384-dimension vectors already take only 1.5 ms per approximate query, and reducing them would change the evaluated scorer |
| Suggested: caching | Offline only (product vectors keyed by model revision and texts). Responses are deterministic, so a response cache would be valid, but at 2–7 ms per request none was needed |
