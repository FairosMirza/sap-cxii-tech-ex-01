"""Freeze everything that decides a ranking before the holdout is pooled (D07).

The record holds the commit; the git objects of app/, configs/ and the pooling and evaluation scripts (scoring,
filters, parsing and de-duplication rules, metrics); SHA-256 fingerprints of the configuration, the serving artifacts,
the encoder files and cached product vectors the pooled rankers load, the dataset and the query sets; the numerical
package versions; and, for reading, the resolved settings (model revisions, weights, candidate counts, index
parameters, cutoff). Holdout pooling and evaluation call `check()` and stop if anything differs. The record is never
overwritten.
Run on a clean tree after rebuilding the artifacts at that commit (python -m scripts.prepare):
    python -m scripts.freeze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

RECORD = Path("eval/freeze.json")
CODE = ("app", "configs", "scripts/build_label_pools.py", "scripts/check_labels.py", "scripts/evaluate.py", "scripts/freeze.py",
        "scripts/label_server.py")
PACKAGES = ("numpy", "scipy", "scikit-learn", "faiss-cpu", "onnxruntime", "tokenizers")
MODELS = ("bge-small", "minilm")  # the embedding models of the pooled rankers
SECTIONS = ("code", "files", "packages", "settings")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _files() -> dict[str, Path]:
    from app.data import DEFAULT_DATA_FILE
    from app.embeddings import EMBEDDING_MODELS, MODEL_FILES, download, embedding_cache
    from app.settings import DEFAULT_CONFIG

    files = {str(p): p for p in (DEFAULT_CONFIG, DEFAULT_DATA_FILE, Path("eval/queries.json"), Path("eval/semantic_queries.json"))}
    files.update({str(p): p for p in sorted(Path("artifacts").rglob("*")) if p.is_file()})
    for key in MODELS:
        spec = EMBEDDING_MODELS[key]
        snapshot = download(spec)
        used = sorted({p for pattern in MODEL_FILES for p in snapshot.glob(pattern) if p.is_file()})
        files.update({f"model {key}/{p.relative_to(snapshot)}": p for p in used})
        cached = sorted(embedding_cache().glob(f"{key}-{spec.revision[:10]}-onnx-{spec.max_seq_length}-*.npy"))
        if len(cached) != 1:
            raise SystemExit(f"expected one cached product-vector file for {key}, found {len(cached)}")
        files[f"cached vectors {key}"] = cached[0]
    return files


def fingerprint() -> dict:
    from app.embeddings import EMBEDDING_MODELS
    from app.settings import load_config, load_serving, method_names

    manifest = json.loads(Path("artifacts/manifest.json").read_text())
    current = {
        "code": {path: _git("rev-parse", f"HEAD:{path}") for path in CODE},
        "files": {name: _sha256(path) for name, path in _files().items()},
        "packages": {name: version(name) for name in PACKAGES},
        "settings": {
            "serving": load_serving(),
            "methods": {name: asdict(load_config(method=name)) for name in method_names()},
            "models": {key: asdict(EMBEDDING_MODELS[key]) for key in MODELS},
            "index": {key: rep["index"]["params"] for key, rep in manifest["representations"].items() if "index" in rep},
            "artifacts_built_from": manifest["code_commit"],
        },
    }
    return json.loads(json.dumps(current))  # compare as stored: tuples become lists


def differences(frozen: dict, current: dict) -> list[str]:
    return [f"{section}: {key}" for section in SECTIONS
            for key in sorted(set(frozen.get(section, {})) | set(current.get(section, {})))
            if frozen.get(section, {}).get(key) != current.get(section, {}).get(key)]


def _uncommitted() -> str:
    return _git("status", "--porcelain", "--", *CODE)


def check(path: Path = RECORD) -> dict:
    """Stop unless the code, settings and every fingerprint still match the freeze record."""
    if not path.exists():
        raise SystemExit(f"no freeze record at {path}: run `python -m scripts.freeze` first (D07)")
    if _uncommitted():
        raise SystemExit("the frozen code has uncommitted changes: " + ", ".join(CODE))
    record = json.loads(path.read_text())
    changed = differences(record, fingerprint())
    if changed:
        raise SystemExit("the frozen configuration has changed since " + record["commit"] + ": " + "; ".join(changed))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(RECORD))
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists: the configuration is already frozen")
    if _uncommitted():
        raise SystemExit("commit the code first, the freeze records a commit: " + ", ".join(CODE))
    head = _git("rev-parse", "HEAD")
    built_from = json.loads(Path("artifacts/manifest.json").read_text())["code_commit"]
    if built_from != head:
        raise SystemExit(f"the artifacts were built from {built_from}, not {head}: run `python -m scripts.prepare`")
    record = {"frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": head, **fingerprint()}
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"frozen at {head}: {len(record['files'])} files fingerprinted; written to {out}")


if __name__ == "__main__":
    main()
