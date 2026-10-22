from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.ann import HNSWIndex, HNSWParams
from app.api import create_app
from app.pipeline import build_store, embedding_modality
from app.retrieval import SimilarityEngine
from app.semantic import SemanticSearcher
from tests.conftest import config, make_record
from tests.test_semantic import fake_encode

TITLES = ["Black Leather Wallet", "Men Leather Wallet", "Running Shoes", "Women Running Shoes",
          "Winter Jacket", "Men Winter Jacket", "Summer Dress", "Women Summer Dress"]
UNBRANDED = "ee".ljust(32, "0")


def loader():
    records = [make_record(f"a{i}", title, brand=f"B{i % 3}", price=f"{300 + 100 * i}.00") for i, title in enumerate(TITLES)]
    records.append(make_record("ee", "Leather Wallet Brown", brand=None))
    store = build_store(records, text=False)
    vectors = np.stack([fake_encode(t) for t in store.catalog.texts])
    store.modalities["fake"] = embedding_modality(store.catalog, vectors, fake_encode)
    store.indexes["fake"] = HNSWIndex.build(store.modalities["fake"].vectors, HNSWParams(exact_below=0))
    cfg = replace(config(text=0.5, attributes=0.5), text_model="fake")
    manifest = {"dataset": {"sha256": "abc"}, "serving": {"similar_method": "C_fake", "semantic_model": "fake"}}
    return SimilarityEngine(store, cfg), SemanticSearcher(store, "fake", cfg), manifest


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(loader)) as c:
        yield c


def pid(i: int) -> str:
    return f"a{i}".ljust(32, "0")


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["dataset_sha256"] == "abc"


@pytest.mark.parametrize("mode", [None, "exact", "ann"])
def test_find_similar_products_returns_requested_count(client, mode):
    params = {"product_id": pid(0), "num_similar": 4} | ({"mode": mode} if mode else {})
    response = client.get("/find_similar_products", params=params)
    assert response.status_code == 200
    ids = response.json()
    assert len(ids) == 4 and pid(0) not in ids and all(isinstance(i, str) for i in ids)
    assert "x-request-id" in response.headers


@pytest.mark.parametrize("params, status", [
    ({"product_id": "f" * 32, "num_similar": 3}, 404),
    ({"product_id": "not-an-id", "num_similar": 3}, 422),
    ({"product_id": pid(0), "num_similar": 0}, 422),
    ({"product_id": pid(0), "num_similar": 101}, 422),
    ({"product_id": pid(0), "num_similar": "three"}, 422),
    ({"product_id": pid(0)}, 422),
    ({"num_similar": 3}, 422),
    ({"product_id": pid(0), "num_similar": 3, "min_price": 900, "max_price": 100}, 422),
    ({"product_id": pid(0), "num_similar": 3, "min_price": -5}, 422),
    ({"product_id": UNBRANDED, "num_similar": 3, "same_brand": "true"}, 422),
    ({"product_id": pid(0), "num_similar": 3, "mode": "fast"}, 422),
])
def test_find_similar_products_errors(client, params, status):
    assert client.get("/find_similar_products", params=params).status_code == status


def test_find_similar_products_filters(client):
    ids = client.get("/find_similar_products", params={"product_id": pid(0), "num_similar": 5, "max_price": 600}).json()
    assert ids and set(ids) <= {pid(1), pid(2), pid(3), UNBRANDED}  # the only other products priced <= 600


def test_semantic_search(client):
    response = client.get("/semantic_search", params={"query": "men winter jacket", "num_results": 2})
    assert response.status_code == 200 and response.json()[0] == pid(5)
    priced = client.get("/semantic_search", params={"query": "running shoes under 450", "num_results": 5}).json()
    assert set(priced) <= {pid(0), pid(1)}  # only products under 450 pass the filter
    assert client.get("/semantic_search", params={"query": "running shoes under 250", "num_results": 5}).json() == []


@pytest.mark.parametrize("params", [
    {"query": "", "num_results": 3}, {"query": "x" * 201, "num_results": 3},
    {"query": "shoes", "num_results": 0}, {"num_results": 3},
])
def test_semantic_search_validation(client, params):
    assert client.get("/semantic_search", params=params).status_code == 422


def test_parse_query_endpoint(client):
    parsed = client.get("/parse_query", params={"query": "black leather wallet for men under 500"}).json()
    assert parsed["price"]["maximum"] == 500 and parsed["colors"] == ["black"] and parsed["materials"] == ["leather"]


def test_not_ready_service_reports_503_without_internal_details():
    def broken():
        raise FileNotFoundError("/secret/path/manifest.json")

    with TestClient(create_app(broken)) as c:
        assert c.get("/health").status_code == 200
        ready = c.get("/ready")
        assert ready.status_code == 503 and "secret" not in ready.text
        assert c.get("/find_similar_products", params={"product_id": pid(0), "num_similar": 3}).status_code == 503
        assert c.get("/semantic_search", params={"query": "shoes", "num_results": 3}).status_code == 503


def test_unexpected_errors_return_500_with_request_id():
    def failing_loader():
        engine, searcher, manifest = loader()

        def boom(*args, **kwargs):
            raise RuntimeError("internal detail")

        engine.search = boom
        return engine, searcher, manifest

    with TestClient(create_app(failing_loader), raise_server_exceptions=False) as c:
        response = c.get("/find_similar_products", params={"product_id": pid(0), "num_similar": 3})
        assert response.status_code == 500 and "internal detail" not in response.text
        assert response.json()["request_id"] == response.headers["x-request-id"]
