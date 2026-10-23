"""Image experiment (D14, pre-registered): C_bge plus an image block, on the 20 development products only.

  pool      rank the development products with C_bge and C_bge+image_w (w = 0.1, 0.2, 0.3); pairs that no
            development pool holds go to a blind pool in eval/experiments/images/ (never overwritten)
  label     serve that pool on the blind labelling page; labels go to eval/experiments/images/labels.jsonl, apart
            from the core's development labels, which stay unchanged
  evaluate  validate the labels, then nDCG@10, P@5, grade-0 results shown and paired differences against C_bge, the
            D14 rule and the missing-image checks -> reports/experiments/images.json
  coverage  the D08 approximate-search benchmark for the chosen configuration: candidates still come from text, so
            serving image scores needs it -> reports/experiments/images_ann_coverage.json
  fallback  why some lists change when the query's image is removed: each differing list is recomputed with the
            evidence floor excluding the image block, then also weighted in float64, and attributed to the evidence
            floor, float32 weighting or both; plus how many lists float64 weighting changes with images present
            -> reports/experiments/images_fallback.json
Needs the image vectors first:  python -m experiments.image_features
Run from the repository root:  python -m experiments.image_experiment {pool,label,evaluate,coverage,fallback}
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from app.pipeline import fit_anchors
from app.retrieval import SimilarityEngine
from app.scoring import Anchors, DenseModality, FeatureStore
from app.settings import ScoringConfig, load_config
from experiments.image_features import vectors_path
from scripts.benchmark_ann import query_sets
from scripts.build_label_pools import K, SUPPLEMENT_SEED, store_with_embeddings, supplement_queries
from scripts.check_labels import check, holdout_queries, labels_file, pool_files
from scripts.evaluate import check_fresh, choose_text_weight, evaluate_task

WEIGHTS = (0.1, 0.2, 0.3)
POOL_DIR = Path("eval/experiments/images")
LABELS = POOL_DIR / "labels.jsonl"
SUPPLEMENT = "images"
POOL = POOL_DIR / f"similar_dev_supplement_{SUPPLEMENT}.json"


def image_store() -> FeatureStore:
    store = store_with_embeddings()
    vectors = np.load(vectors_path(list(store.catalog.ids)))
    available = np.linalg.norm(vectors, axis=1) > 0
    store.modalities["image"] = DenseModality(vectors, available, fit_anchors(vectors, available, store.catalog.asins))
    return store


@dataclass(frozen=True)
class Float64Anchors(Anchors):
    """The served calibration, returned in float64 so that weights multiply without float32 rounding."""

    def apply(self, raw: np.ndarray) -> np.ndarray:
        return super().apply(raw).astype(np.float64)


def float64_store(store: FeatureStore) -> FeatureStore:
    """The same store with dense similarities weighted in float64; availability arrays are shared, not copied."""
    return replace(store, modalities={key: replace(m, anchors=Float64Anchors(**vars(m.anchors))) for key, m in store.modalities.items()})


def configurations() -> dict[str, ScoringConfig]:
    base = load_config(method="C_bge")
    rest = {w: round((1 - w) / 2, 2) for w in WEIGHTS}
    return {"C_bge": base, **{f"C_bge+image_{w}": base.with_blocks(text=rest[w], image=w, attributes=rest[w]) for w in WEIGHTS}}


def dev_rows(store: FeatureStore) -> list[int]:
    queries = json.loads(Path("eval/queries.json").read_text())["queries"]
    return [store.catalog.index[q["id"]] for q in queries if q["split"] == "dev"]


def top10(store: FeatureStore, configs: dict[str, ScoringConfig], rows: list[int]) -> dict[str, dict[str, list[int]]]:
    engines = {name: SimilarityEngine(store, config) for name, config in configs.items()}
    return {name: {row: [r for r, _ in e.search(row, K, mode="exact").hits] for row in rows} for name, e in engines.items()}


def as_ids(store: FeatureStore, lists: dict[str, dict[int, list[int]]]) -> dict[str, dict[str, list[str]]]:
    ids = store.catalog.ids
    rows = next(iter(lists.values()))
    return {ids[row]: {name: [ids[r] for r in lists[name][row]] for name in lists} for row in rows}


def pool() -> None:
    if POOL.exists():
        raise SystemExit(f"{POOL} exists; pools that may already be labelled are never overwritten")
    store = image_store()
    ranked = as_ids(store, top10(store, configurations(), dev_rows(store)))
    pooled = {(q["query"], c) for path in pool_files(Path("eval/pools"), "similar", "dev")
              for q in json.loads(path.read_text())["queries"] for c in q["candidates"]}
    queries, membership = supplement_queries(ranked, pooled)
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    POOL.write_text(json.dumps({"task": "similar", "split": "dev", "k": K, "seed": SUPPLEMENT_SEED, "queries": queries},
                               indent=1, ensure_ascii=False) + "\n")
    POOL.with_name(f"{POOL.stem}_membership.json").write_text(
        json.dumps({"systems": list(configurations()), "queries": membership}, indent=1, ensure_ascii=False) + "\n")
    print(f"{sum(len(q['candidates']) for q in queries)} pairs over {len(queries)} queries written to {POOL}")


def label(port: int) -> None:
    import uvicorn

    from scripts.label_server import create_app

    uvicorn.run(create_app("dev", pools_dir=POOL_DIR, labels_path=LABELS, supplement=SUPPLEMENT),
                host="127.0.0.1", port=port, log_level="warning")


def missing_image_checks(store: FeatureStore, configs: dict[str, ScoringConfig], rows: list[int], sample: list[int]) -> dict:
    image = store.modalities["image"]
    variants = [name for name in configs if name != "C_bge"]
    report = {"catalogue_share_without_image": round(float(1 - image.available.mean()), 4)}
    dev_lists = top10(store, configs, rows)
    no_image = [row for row in rows if not image.available[row]]
    report["dev_queries_without_image"] = {store.catalog.ids[row]: {name: dev_lists[name][row] == dev_lists["C_bge"][row] for name in variants}
                                           for row in no_image}
    sample_lists = top10(store, configs, sample)
    report["share_of_top10_without_image"] = {
        name: {"dev": round(float(np.mean([not image.available[r] for row in rows for r in dev_lists[name][row]])), 4),
               "sample": round(float(np.mean([not image.available[r] for row in sample for r in sample_lists[name][row]])), 4)}
        for name in configs}
    with_image = [row for row in sample if image.available[row]]
    identical = dict.fromkeys(variants, 0)
    for row in with_image:
        image.available[row] = False
        try:
            removed = top10(store, {name: configs[name] for name in variants}, [row])
        finally:
            image.available[row] = True
        for name in variants:
            identical[name] += removed[name][row] == sample_lists["C_bge"][row]
    report["sample_query_image_removed"] = {"queries": len(with_image),
                                            "top10_identical_to_C_bge": identical}
    return report


def evaluate(out: Path) -> None:
    report, merged = check(LABELS, POOL_DIR, "dev", holdout_queries())
    if report["failures"]:
        raise SystemExit("experiment labels fail validation: " + json.dumps(report["failures"])[:2000])
    core = json.loads(Path("eval/qrels_dev.json").read_text())
    check_fresh(core, labels_file("dev"), Path("eval/pools"), "dev")
    judgements = dict(core["judgements"]["similar"])
    for query, grades in merged["similar"].items():
        judgements[query] = {**judgements.get(query, {}), **grades}

    store = image_store()
    configs = configurations()
    rows = dev_rows(store)
    ranked = as_ids(store, top10(store, configs, rows))
    unjudged = sorted({(q, pid) for q, lists in ranked.items() for ids in lists.values() for pid in ids if pid not in judgements.get(q, {})})
    if unjudged:
        raise SystemExit(f"{len(unjudged)} retrieved pairs are unjudged; pool them before evaluating")
    title_of = dict(zip(store.catalog.ids, store.catalog.title_codes.tolist()))
    result = evaluate_task("similar", ranked, judgements, title_of)
    sweep = {float(spec.partition("+image_")[2]): v["ndcg_difference"] for spec, v in result["variants"].items()}
    image = store.modalities["image"]
    sample = query_sets(SimilarityEngine(store, configs["C_bge"]))["sample_rows"]
    summary = {
        "protocol": "docs/decisions.md D14, pre-registered before any image vector was computed",
        "scope": "development evidence only; the holdout is spent (D07)",
        "image_vectors": {"products_with_vector": int(image.available.sum()), "anchors": vars(image.anchors)},
        "labels": {"experiment_pairs": report["expected_pairs"], "file": str(LABELS)},
        "rankers": result["rankers"], "versus_C_bge": result["variants"],
        "rule_d14": {"chosen_image_weight": choose_text_weight(0.0, sweep),
                     "rule": "adopt only a weight beating C_bge by more than one paired SE; if several, the smallest"},
        "missing_images": missing_image_checks(store, configs, rows, sample),
        "per_query": result["per_query"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    for spec, s in summary["rankers"].items():
        print(f"{spec:18s} nDCG@10 {s['ndcg']:.3f} {s['ndcg_ci95']}  P@5 {s['p5']:.3f}  grade 0 shown {s['grade_0_results_shown']}")
    for spec, v in summary["versus_C_bge"].items():
        print(f"{spec} - C_bge: nDCG {json.dumps(v['ndcg_difference'])}, P@5 {json.dumps(v['p5_difference'])}")
    print("rule:", json.dumps(summary["rule_d14"]))
    print("missing images:", json.dumps(summary["missing_images"]))
    print(f"written to {out}")


def coverage(out: Path, candidates: int = 800) -> None:
    from app.ann import HNSWIndex, HNSWParams
    from scripts.benchmark_ann import similar_benchmark

    store = image_store()
    index = json.loads(Path("artifacts/manifest.json").read_text())["representations"]["bge-small"]["index"]
    store.indexes["bge-small"] = HNSWIndex.load(Path("artifacts") / index["file"], HNSWParams(**index["params"]))
    chosen = json.loads(Path("reports/experiments/images.json").read_text())["rule_d14"]["chosen_image_weight"]
    name = f"C_bge+image_{chosen}"
    engine = SimilarityEngine(store, configurations()[name].with_search("ann", candidates))
    sets = query_sets(engine)
    report = {"configuration": name, "deployed_candidates": candidates, "bar": 0.95, "method": "scripts.benchmark_ann (D08)",
              "dev": similar_benchmark(engine, sets["dev_rows"]), "sample": similar_benchmark(engine, sets["sample_rows"])}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    for split in ("dev", "sample"):
        for n, e in report[split]["by_candidates"].items():
            print(f"{name} {split} N={n}: exact-text {e['exact_text']['mean']:.4f} | faiss {e['faiss']['mean']:.4f} | "
                  f"served {e['served']['mean']:.4f} {e['served']['median_ms']} ms")
    print(f"written to {out}")


CAUSES = ("evidence floor", "float32 weighting", "both", "unexplained")


def fallback(out: Path) -> None:
    from app.scoring import component_scores, score_all

    store = image_store()
    exact = float64_store(store)
    configs = configurations()
    base, image = configs["C_bge"], store.modalities["image"]
    weights = {name: w for name, w in base.component_weights().items() if w > 0}
    floor = base.min_evidence * sum(base.component_weights().values())
    sample = [r for r in query_sets(SimilarityEngine(store, base))["sample_rows"] if image.available[r]]
    variants = {name: config for name, config in configs.items() if name != "C_bge"}
    # The same text and attribute weights without the image block, so the evidence floor no longer counts it.
    without_block = {name: base.with_blocks(text=c.block_weights["text"], image=0.0, attributes=c.block_weights["attributes"])
                     for name, c in variants.items()}
    probe = component_scores(store, sample[0], configs["C_bge+image_0.1"], {"text", "image"})
    report = {"queries": len(sample), "floor": floor, "similarity_dtypes": {k: str(v.dtype) for k, v in probe.sims.items()},
              "weights": {}}
    for name in variants:
        report["weights"][name] = {"identical_to_C_bge": 0, "identical_once_the_floor_excludes_the_image_block": 0,
                                   "identical_once_also_weighted_in_float64": 0, "causes": dict.fromkeys(CAUSES, 0),
                                   "evidence_threshold": round(floor / (1 - variants[name].block_weights["image"]), 4), "differing": []}
    for row in sample:
        image.available[row] = False
        try:
            reference = top10(store, {"C_bge": base}, [row])["C_bge"][row]
            for name, config in variants.items():
                entry = report["weights"][name]
                lists = {k: v[row] for k, v in top10(store, {name: config, "without": without_block[name]}, [row]).items()}
                corrected = top10(exact, {"without": without_block[name]}, [row])["without"][row] == reference
                entry["identical_once_the_floor_excludes_the_image_block"] += lists["without"] == reference
                entry["identical_once_also_weighted_in_float64"] += corrected
                if lists[name] == reference:
                    entry["identical_to_C_bge"] += 1
                    continue
                if lists["without"] == reference:
                    cause = "evidence floor"
                elif top10(exact, {name: config}, [row])[name][row] == reference:
                    cause = "float32 weighting"
                else:
                    cause = "both" if corrected else "unexplained"
                entry["causes"][cause] += 1
                moved = [r for r in dict.fromkeys(reference + lists[name])
                         if (reference.index(r) if r in reference else None) != (lists[name].index(r) if r in lists[name] else None)]
                rows = np.array(moved)
                masks = component_scores(store, row, base, set(weights), rows).masks
                evidence = sum(w * masks[c] for c, w in weights.items())
                ratio = score_all(store, row, config, rows=rows) / score_all(store, row, base, rows=rows)
                entry["differing"].append({"query": store.catalog.ids[row], "name": store.catalog.names[row][:60], "cause": cause, "moved": [
                    {"id": store.catalog.ids[r], "rank_C_bge": reference.index(r) + 1 if r in reference else None,
                     "rank_with_image_weight": lists[name].index(r) + 1 if r in lists[name] else None,
                     "evidence": round(float(e), 4), "score_ratio": round(float(q), 4)}
                    for r, e, q in zip(moved, evidence, ratio)]})
        finally:
            image.available[row] = True
    for entry in report["weights"].values():
        entry["products_at_full_evidence_keep_their_score"] = all(
            m["score_ratio"] == 1.0 for d in entry["differing"] for m in d["moved"] if m["evidence"] >= entry["evidence_threshold"])
    # Would correcting both (float64 weighting; the floor excluding the image block when the query has none) change
    # the development lists the reported results were computed from?
    dev = dev_rows(store)
    served = top10(store, configs, dev)
    report["development_lists_unchanged_by_both_corrections"] = {"queries": len(dev), **{
        name: sum(top10(exact, {name: configs[name] if image.available[r] else without_block[name]}, [r])[name][r] == served[name][r]
                  for r in dev) for name in variants}}
    served, wide = top10(store, configs, sample), top10(exact, configs, sample)
    report["sample_lists_unchanged_by_float64_weighting_with_images_present"] = {"queries": len(sample), **{
        name: sum(served[name][r] == wide[name][r] for r in sample) for name in configs}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    n = report["queries"]
    for name, entry in report["weights"].items():
        print(f"{name}: identical {entry['identical_to_C_bge']}/{n}; once the floor excludes the image block "
              f"{entry['identical_once_the_floor_excludes_the_image_block']}/{n}; once also weighted in float64 "
              f"{entry['identical_once_also_weighted_in_float64']}/{n}; differing lists by cause {json.dumps(entry['causes'])}; "
              f"threshold E < {entry['evidence_threshold']}; full-evidence products unchanged: {entry['products_at_full_evidence_keep_their_score']}")
    for d in report["weights"]["C_bge+image_0.1"]["differing"]:
        print(f"  [{d['cause']}] {d['name']}: " + "; ".join(
            f"rank {m['rank_C_bge']}->{m['rank_with_image_weight']} E={m['evidence']} ratio={m['score_ratio']}" for m in d["moved"]))
    print("similarity dtypes:", json.dumps(report["similarity_dtypes"]))
    print("development lists unchanged by both corrections:", json.dumps(report["development_lists_unchanged_by_both_corrections"]))
    print("sample lists unchanged by float64 weighting, images present:",
          json.dumps(report["sample_lists_unchanged_by_float64_weighting_with_images_present"]))
    print(f"written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["pool", "label", "evaluate", "coverage", "fallback"])
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--out", help="default: reports/experiments/images.json (evaluate), images_<command>.json otherwise")
    args = parser.parse_args()
    if args.command == "pool":
        pool()
    elif args.command == "label":
        label(args.port)
    elif args.command == "evaluate":
        evaluate(Path(args.out or "reports/experiments/images.json"))
    elif args.command == "coverage":
        coverage(Path(args.out or "reports/experiments/images_ann_coverage.json"))
    else:
        fallback(Path(args.out or "reports/experiments/images_fallback.json"))


if __name__ == "__main__":
    main()
