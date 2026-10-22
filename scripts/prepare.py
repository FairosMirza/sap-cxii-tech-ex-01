"""Build the serving artifacts (vectors, HNSW indexes, model files, manifest).

Run from the repository root:  python -m scripts.prepare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.artifacts import build
from app.data import DEFAULT_DATA_FILE
from app.settings import DEFAULT_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data", default=str(DEFAULT_DATA_FILE))
    args = parser.parse_args()
    manifest = build(Path(args.out), Path(args.config), Path(args.data))
    summary = {key: {k: v for k, v in rep.items() if k in ("kind", "model_id", "revision", "dimensions", "index")}
               for key, rep in manifest["representations"].items()}
    print(json.dumps({"dataset": manifest["dataset"], "serving": manifest["serving"], "representations": summary}, indent=2))


if __name__ == "__main__":
    main()
