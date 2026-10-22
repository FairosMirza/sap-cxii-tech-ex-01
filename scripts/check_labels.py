"""Validate the human relevance labels against the pools, then merge them into one judgement per pair.

Fails on: malformed lines or grades outside 0-3, labels for pairs no pool contains, pooled pairs without a
label, a pair pooled twice, conflicting grades for one pair (unless --latest-wins accepts them as deliberate
re-grades, which is the labelling page's rule) and labels on holdout queries. Repeats with the same grade are
only reported. With --rankings it also recomputes the current top-10 of every ranker that contributed to a pool
(systems and variants, read from the membership files) and fails if any retrieved pair is unlabelled. The merged
judgements are written only when every check passes.
Run from the repository root:  python -m scripts.check_labels --rankings
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

TASKS = ("similar", "semantic")
FIELDS = {"task", "split", "query", "candidate", "grade", "labelled_at"}
PRODUCT_ID = re.compile(r"[0-9a-f]{32}")


def pool_files(pools_dir: Path, task: str, split: str) -> list[Path]:
    """The pool of a task and split plus its supplements; membership files are not pools."""
    return sorted(p for p in pools_dir.glob(f"{task}_{split}*.json") if not p.stem.endswith("_membership"))


def ranker_specs(pools_dir: Path, task: str, split: str) -> list[str]:
    """Every ranker (a system, optionally SYSTEM+VARIANT) that contributed to the task's pools."""
    return sorted({spec for path in pool_files(pools_dir, task, split)
                   for spec in json.loads(path.with_name(f"{path.stem}_membership.json").read_text())["systems"]})


def holdout_queries(eval_dir: Path = Path("eval")) -> dict[str, set[str]]:
    products = json.loads((eval_dir / "queries.json").read_text())["queries"]
    return {"similar": {q["id"] for q in products if q["split"] == "holdout"},
            "semantic": set(json.loads((eval_dir / "semantic_queries.json").read_text())["holdout"])}


def _problem(row: object) -> str | None:
    if not isinstance(row, dict) or not FIELDS <= row.keys():
        return "missing fields"
    if row["task"] not in TASKS:
        return f"unknown task {row['task']!r}"
    if not PRODUCT_ID.fullmatch(str(row["candidate"])) or (row["task"] == "similar" and not PRODUCT_ID.fullmatch(str(row["query"]))):
        return "invalid product ID"
    if type(row["grade"]) is not int or not 0 <= row["grade"] <= 3:
        return f"grade {row['grade']!r} is not an integer 0-3"
    try:
        datetime.fromisoformat(row["labelled_at"])
    except (TypeError, ValueError):
        return "invalid labelled_at timestamp"
    return None


def check(labels_path: Path, pools_dir: Path, split: str, holdout: dict[str, set[str]],
          latest_wins: bool = False) -> tuple[dict, dict[str, dict[str, dict[str, int]]]]:
    """A report of every check, and the merged judgements {task: {query: {product: grade}}}."""
    failures: dict[str, list] = defaultdict(list)
    expected: dict[tuple[str, str, str], str] = {}
    for task in TASKS:
        for path in pool_files(pools_dir, task, split):
            for q in json.loads(path.read_text())["queries"]:
                for candidate in q["candidates"]:
                    if (task, q["query"], candidate) in expected:
                        failures["pooled_twice"].append([task, q["query"], candidate])
                    expected[(task, q["query"], candidate)] = path.name

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    lines = labels_path.read_text().splitlines() if labels_path.exists() else []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            failures["invalid_lines"].append([number, "not JSON"])
            continue
        problem = _problem(row)
        if problem:
            failures["invalid_lines"].append([number, problem])
        elif split != "holdout" and (row["split"] == "holdout" or row["query"] in holdout[row["task"]]):
            failures["holdout_labels"].append([number, row["task"], row["query"]])
        elif row["split"] == split:
            groups[(row["task"], row["query"], row["candidate"])].append(row)

    merged: dict[str, dict[str, dict[str, int]]] = {task: defaultdict(dict) for task in TASKS}
    repeats, regrades = 0, []
    for (task, query, candidate), group in groups.items():
        grades = [r["grade"] for r in group]
        if len(set(grades)) > 1:
            (regrades if latest_wins else failures["conflicts"]).append([task, query, candidate, grades])
        elif len(group) > 1:
            repeats += 1
        if (task, query, candidate) not in expected:
            failures["unexpected_pairs"].append([task, query, candidate])
        merged[task][query][candidate] = group[-1]["grade"]  # file order: the latest label counts
    failures["missing_labels"] = [list(key) for key in expected if key[2] not in merged[key[0]].get(key[1], {})]

    report = {
        "split": split,
        "pool_files": sorted(set(expected.values())),
        "expected_pairs": len(expected),
        "labelled_pairs": sum(len(products) for queries in merged.values() for products in queries.values()),
        "repeats_with_same_grade": repeats,
        "regrades_accepted": regrades,
        "failures": {name: items for name, items in failures.items() if items},
    }
    return report, {task: {q: dict(products) for q, products in queries.items()} for task, queries in merged.items()}


def unlabelled_retrieved(merged: dict[str, dict[str, dict[str, int]]], split: str, pools_dir: Path) -> list[list[str]]:
    """Pairs in any pooled ranker's current top-10 that have no label. Rankers are not named, to keep labelling blind."""
    from scripts.build_label_pools import semantic_rankings, similar_rankings, store_with_embeddings

    store = store_with_embeddings()
    gaps: set[tuple[str, str, str]] = set()
    for task, rank in (("similar", similar_rankings), ("semantic", semantic_rankings)):
        specs = tuple(ranker_specs(pools_dir, task, split))
        for query, ranked in (rank(store, split, specs) if specs else {}).items():
            gaps |= {(task, query, pid) for ids in ranked.values() for pid in ids if pid not in merged[task].get(query, {})}
    return [list(gap) for gap in sorted(gaps)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--labels", default="eval/labels.jsonl")
    parser.add_argument("--pools", default="eval/pools")
    parser.add_argument("--rankings", action="store_true", help="also confirm every current top-10 pair is labelled")
    parser.add_argument("--latest-wins", action="store_true", help="accept conflicting grades as deliberate re-grades")
    parser.add_argument("--out", help="merged judgements (default: eval/qrels_<split>.json)")
    args = parser.parse_args()

    labels, pools = Path(args.labels), Path(args.pools)
    report, merged = check(labels, pools, args.split, holdout_queries(), args.latest_wins)
    report["holdout_pool_files"] = sorted(p.name for p in pools.glob("*_holdout*.json"))
    if args.rankings:
        gaps = unlabelled_retrieved(merged, args.split, pools)
        report["rankings_checked"] = {task: ranker_specs(pools, task, args.split) for task in TASKS}
        if gaps:
            report["failures"]["unlabelled_retrieved_pairs"] = gaps
    report["passed"] = not report["failures"]
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        sys.exit(1)
    out = Path(args.out or f"eval/qrels_{args.split}.json")
    sources = {"labels": {"file": str(labels), "sha256": sha256(labels)},
               "pools": {name: sha256(pools / name) for name in report["pool_files"]}}
    out.write_text(json.dumps({"split": args.split, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "checks": "scripts/check_labels.py" + (" --rankings" if args.rankings else ""),
                               "sources": sources, "judgements": merged}, indent=1, ensure_ascii=False) + "\n")
    print(f"merged judgements written to {out}")


if __name__ == "__main__":
    main()
