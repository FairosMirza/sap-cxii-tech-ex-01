"""End-to-end check on the real artifacts, in a child process like the service itself.

Covers what the other tests fake: ONNX query encoding and FAISS search behind both endpoints, in both
modes. Also asserts that PyTorch is never imported and at most one OpenMP runtime is loaded (D17).
Skipped until the artifacts are built (python -m scripts.prepare).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / os.environ.get("ARTIFACTS_DIR", "artifacts")
pytestmark = pytest.mark.skipif(not (ARTIFACTS / "manifest.json").exists(), reason="artifacts not built")

CHILD = r"""
import ctypes, json, re, sys
from fastapi.testclient import TestClient
from app.api import create_app
from app.retrieval import find_similar_products

calls = {}
with TestClient(create_app()) as client:
    calls["ready"] = client.get("/ready").status_code
    for mode in ("exact", "ann"):
        similar = client.get("/find_similar_products", params={"product_id": sys.argv[1], "num_similar": 10, "mode": mode})
        semantic = client.get("/semantic_search", params={"query": "red cotton saree for women", "num_results": 10, "mode": mode})
        calls[mode] = [similar.status_code, len(similar.json()), semantic.status_code, len(semantic.json())]
    served = client.get("/find_similar_products", params={"product_id": sys.argv[1], "num_similar": 10}).json()
    calls["part1_function_matches_endpoint"] = find_similar_products(sys.argv[1], 10) == served
if sys.platform == "darwin":
    dyld = ctypes.CDLL(None)
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    libraries = [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]
else:
    libraries = [line.split()[-1] for line in open("/proc/self/maps") if "/" in line]
print(json.dumps({
    "calls": calls,
    "heavy_modules": sorted({name.split(".")[0] for name in sys.modules} & {"torch", "transformers", "sentence_transformers"}),
    "openmp": sorted({lib for lib in libraries if re.search(r"/lib(g|i)?omp[^/]*\.(dylib|so)", lib)}),
}))
"""


def test_service_on_real_artifacts():
    queries = json.loads((ROOT / "eval" / "queries.json").read_text())["queries"]
    product = next(q["id"] for q in queries if q["split"] == "dev")
    child = subprocess.run([sys.executable, "-c", CHILD, product], cwd=ROOT, capture_output=True, text=True, timeout=600,
                           env={**os.environ, "ARTIFACTS_DIR": str(ARTIFACTS)})
    assert child.returncode == 0, child.stderr[-3000:]  # -11 is a segmentation fault
    report = json.loads(child.stdout.strip().splitlines()[-1])
    assert report["calls"] == {"ready": 200, "exact": [200, 10, 200, 10], "ann": [200, 10, 200, 10],
                               "part1_function_matches_endpoint": True}
    assert report["heavy_modules"] == []
    assert len(report["openmp"]) <= 1, report["openmp"]
