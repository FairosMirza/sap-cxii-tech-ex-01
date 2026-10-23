"""Image vectors for the experimental image block (D14): ResNet-50, ImageNet weights, pooled and L2-normalised.

Runs in its own process with PyTorch only; FAISS must not be loaded alongside it (D17). Products without a usable
image get an all-zero row. Writes .cache/embeddings/images-resnet50-<weights>-<catalogue hash>.npy, which the
experiment scripts load without PyTorch.
Run from the repository root (needs requirements-dev.txt):  python -m experiments.image_features
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(".cache/torch").resolve()))

import numpy as np
import orjson
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

DATA_FILE = Path("data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson")
IMAGES = Path("data/images")
WEIGHTS = ResNet50_Weights.IMAGENET1K_V2
BATCH = 64


def catalogue_ids(path: Path = DATA_FILE) -> list[str]:
    """Product IDs in the catalogue's row order: the order app.data keeps after its validity checks."""
    from app.data import load_records  # pure parsing: no FAISS, no ONNX Runtime

    return [r["uniq_id"] for r in load_records(path)[0]]


def vectors_path(ids: list[str]) -> Path:
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()[:12]
    return Path(".cache/embeddings") / f"images-resnet50-{WEIGHTS.name.lower()}-{digest}.npy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    ids = catalogue_ids()
    out = vectors_path(ids)
    if out.exists():
        print(f"{out} exists")
        return
    model = resnet50(weights=WEIGHTS)
    model.fc = torch.nn.Identity()
    model.eval()
    preprocess = WEIGHTS.transforms()
    vectors = np.zeros((len(ids), 2048), dtype=np.float32)
    rows = [i for i, pid in enumerate(ids) if (IMAGES / f"{pid}.jpg").exists()]
    unreadable: list[str] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH):
            batch, kept = [], []
            for i in rows[start:start + BATCH]:
                try:
                    with Image.open(IMAGES / f"{ids[i]}.jpg") as image:
                        batch.append(preprocess(image.convert("RGB")))
                    kept.append(i)
                except OSError:
                    unreadable.append(ids[i])
            if batch:
                features = model(torch.stack(batch)).numpy()
                vectors[kept] = features / np.linalg.norm(features, axis=1, keepdims=True)
            if start // BATCH % 50 == 0:
                print(f"{start + len(batch)} / {len(rows)} images, {time.perf_counter() - started:.0f} s", flush=True)
    seconds = time.perf_counter() - started
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, vectors)
    summary = {"model": "torchvision resnet50", "weights": WEIGHTS.name, "torch": torch.__version__,
               "products": len(ids), "with_vector": int((np.linalg.norm(vectors, axis=1) > 0).sum()),
               "unreadable": unreadable, "seconds": round(seconds, 1), "threads": args.threads, "file": str(out)}
    Path("reports/experiments").mkdir(parents=True, exist_ok=True)
    Path("reports/experiments/image_features.json").write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2) + b"\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
