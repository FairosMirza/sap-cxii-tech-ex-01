# Decision summary

Every decision in one line, then the key trade-offs. Each ID links to the full entry in
[docs/decisions.md](docs/decisions.md), with its evidence and the alternatives rejected; [solutions.md](solutions.md)
explains the approach end to end. The single most consequential decision has its own one-page note, as v2 Part 6
asks: [docs/tradeoff_note.md](docs/tradeoff_note.md).

## Decisions

| ID | Area | Decision | Key evidence |
|---|---|---|---|
| [D01](docs/decisions.md#d01-what-similar-means) | Definition | "Similar" means a plausible substitute: type and use dominate; style, colour, brand and price refine | Makes relevance judgeable; co-purchase data mixes complements with substitutes |
| [D02](docs/decisions.md#d02-scope-and-priorities) | Scope | One service for the original and v2 specifications; essentials first, experiments after | v2's 5–7 hour budget rewards depth over breadth |
| [D03](docs/decisions.md#d03-data-parsing-rules) | Data | An explicit parsing rule per field; placeholders become missing; missing never matches missing | Weight is a placeholder for 79%, brand missing for 27%, item colour unknown for 57% |
| [D04](docs/decisions.md#d04-duplicates-and-variants) | Duplicates | Exclude the query's ASIN and variant family; one listing per family; title cap available, off | 471 duplicate-ASIN rows; the cap changed nDCG@10 by −0.001 |
| [D05](docs/decisions.md#d05-scoring-and-fusion-maths) | Scoring | Evidence-aware weighted average of calibrated components; text 0.5, attributes 0.5 | No text weight beat 0.5 by more than one standard error |
| [D06](docs/decisions.md#d06-product-type-must-dominate) | Product type | No type penalty | Only 3 of C_bge's 200 development results are irrelevant products of another category |
| [D07](docs/decisions.md#d07-evaluation-protocol) | Evaluation | Blind pooled human labels; rules fixed before metrics; the frozen configuration checked once on a holdout | 844 development and 417 holdout judgements |
| [D08](docs/decisions.md#d08-default-search-mode-rule) | Search mode | Approximate (HNSW) search by default: 800 candidates for similar products, 200 for search | 99.1% of the exact top-10 in 1.5 ms instead of 6.3 ms; no relevance cost on the holdout |
| [D09](docs/decisions.md#d09-text-representation) | Text model | bge-small for both endpoints; TF-IDF as baseline; mpnet dropped | Development search +0.183 nDCG@10 over MiniLM; holdout: ahead for similar products, search not separated |
| [D10](docs/decisions.md#d10-semantic-search) | Query parsing | Rule-based filter parsing; filter first, then rank | Deterministic, free and offline; 36 unit tests; no API key for reviewers |
| [D11](docs/decisions.md#d11-currency) | Currency | Query amounts are read as rupees | The catalogue is from amazon.in; "under $100" means ₹100, a documented limitation |
| [D12](docs/decisions.md#d12-relevance-cutoff-for-semantic-search) | Relevance cutoff | Search drops results below cosine 0.5593, learned from labels | Removes 12 of 30 irrelevant development results and no useful one |
| [D13](docs/decisions.md#d13-catalogue-limitations) | Catalogue | Gaps documented, not hidden | 17 footwear products, no wallets or laptop backpacks, misfiled listings |
| [D14](docs/decisions.md#d14-images) | Images | ResNet-50 image block: weight 0.1, the smallest meeting the adoption rule on development labels; not served | +0.048 nDCG@10 on development; fallback not exact, coverage 0.930 at 800 candidates, holdout spent |
| [D15](docs/decisions.md#d15-semantic-ids) | Semantic IDs | Tested as a candidate index; not adopted | Coverage 0.789 against HNSW's 0.991 at 800 candidates |
| [D16](docs/decisions.md#d16-engineering-choices) | Engineering | Python 3.12; build offline, serve online; two-stage Docker image; Kubernetes on kind | Fresh-clone, no-cache build in about 10 minutes; non-root, read-only root filesystem |
| [D17](docs/decisions.md#d17-inference-runtime-onnx-runtime-instead-of-pytorch) | Runtime | ONNX Runtime instead of PyTorch for text encoding | Fixed a FAISS–PyTorch crash; dependencies 64% smaller; embeddings agree to cosine 0.9999997 |

## Key trade-offs

| Trade-off | Chosen | Gained | Cost or risk accepted |
|---|---|---|---|
| Text runtime (D17) | ONNX Runtime over PyTorch | No crash; dependencies 1,436 MB down to 522 MB | Offline encoding about 20 s slower; we own pooling and normalisation, covered by a parity check |
| Text model (D09) | bge-small over MiniLM | Better development search (+0.183 nDCG@10); one model, vector set and index | About 1.9 ms more per search and 43 MB; the search lead did not replicate on the holdout |
| Search mode (D08) | Approximate over exhaustive | 1.5 ms instead of 6.3 ms per similar-products query, and room to grow | 0.9% of the exact top-10 lost on 1,000 sampled products |
| Fusion (D05) | Late fusion with fixed attribute weights, over concatenated vectors or learned weights | Interpretable scores; missing data handled per pair; stable with 20 queries | Weights are not learned; on attributes alone, product type can lose (D06) |
| Query parsing (D10) | Rules over an LLM | Deterministic, free, offline and testable | Phrasings outside the rules are missed; amounts are read as rupees (D11) |
| Relevance cutoff (D12) | Keep every useful development pair | Clearly irrelevant results removed without losing useful ones | Too low to empty a query that has no suitable product |
| De-duplication (D04) | One listing per variant family; title cap off | No size-variant repeats | Listings sharing a title still repeat (1.45 per top-10) |
| Evaluation (D07) | Fixed rules, blind labels, a once-only holdout | An unbiased check of the frozen configuration | Small samples, wide intervals and one person's judgements |
| Images (D14) | Not served | The served path stays verified and simple | The +0.048 nDCG@10 seen on development is not delivered |
| Candidate index (D15) | HNSW over semantic IDs | 0.991 coverage against 0.789 | A larger index (8.2 MB against 0.9 MB) and a slower build (6.2 s against 0.18 s) |
