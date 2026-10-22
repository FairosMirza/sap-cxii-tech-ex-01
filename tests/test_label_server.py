import json
import re

import pytest
from fastapi.testclient import TestClient

from scripts.label_server import create_app, read_labels
from tests.conftest import make_record

QUERY, CAND_A, CAND_B = (s.ljust(32, "0") for s in ("a1", "b2", "c3"))


@pytest.fixture
def setup(tmp_path):
    records = [make_record("a1", "Black <b>Kurti</b>"), make_record("b2", "Red Kurti"), make_record("c3", "Wallet")]
    (tmp_path / "similar_dev.json").write_text(json.dumps({"queries": [{"query": QUERY, "candidates": [CAND_A, CAND_B]}]}))
    labels = tmp_path / "labels.jsonl"
    app = create_app("dev", records=records, pools_dir=tmp_path, labels_path=labels, images_dir=tmp_path)
    client = TestClient(app, follow_redirects=False)
    page = client.get("/item/similar/0/0").text
    token = re.search(r'name="token" value="([^"]+)"', page).group(1)
    return client, labels, token, page


def post(client, token, grade=2, ci=0, note=""):
    return client.post("/label", data={"task": "similar", "qi": 0, "ci": ci, "grade": grade, "token": token, "note": note})


def test_page_is_blind_and_escapes_titles(setup):
    _, _, _, page = setup
    assert "&lt;b&gt;Kurti&lt;/b&gt;" in page and "<b>Kurti</b>" not in page
    for hidden in ("A_structured", "B_tfidf", "C_minilm", "score", "WomensKurtasKurtis"):
        assert hidden not in page


def test_label_is_saved_and_moves_to_next_unlabelled(setup):
    client, labels, token, _ = setup
    response = post(client, token, grade=3)
    assert response.status_code == 303 and response.headers["location"] == "/next/similar"
    assert client.get("/next/similar").headers["location"] == "/item/similar/0/1"
    saved = read_labels(labels)
    assert saved[("similar", "dev", QUERY, CAND_A)]["grade"] == 3


def test_regrading_keeps_the_latest_label(setup):
    client, labels, token, _ = setup
    post(client, token, grade=1)
    post(client, token, grade=2)
    assert read_labels(labels)[("similar", "dev", QUERY, CAND_A)]["grade"] == 2


@pytest.mark.parametrize("kwargs, status", [({"grade": 4}, 422), ({"ci": 5}, 404), ({"note": "x" * 501}, 422)])
def test_invalid_labels_rejected(setup, kwargs, status):
    client, labels, token, _ = setup
    assert post(client, token, **kwargs).status_code == status
    assert not labels.exists()


def test_requests_without_the_session_token_are_rejected(setup):
    client, labels, _, _ = setup
    assert post(client, "forged").status_code == 403
    assert not labels.exists()


@pytest.mark.parametrize("pid", ["not-an-id", "..%2F..%2Fetc%2Fpasswd", "A" * 32])
def test_image_route_only_serves_product_ids(setup, pid):
    client, _, _, _ = setup
    assert client.get(f"/img/{pid}").status_code == 404


def test_missing_image_returns_placeholder(setup):
    client, _, _, _ = setup
    response = client.get(f"/img/{CAND_B}")
    assert response.status_code == 200 and response.headers["content-type"].startswith("image/svg+xml")


def test_supplemental_pool_is_served_on_its_own_and_labelled_for_the_split(tmp_path):
    records = [make_record("a1", "Black Kurti"), make_record("b2", "Red Kurti"), make_record("c3", "Green Kurti")]
    (tmp_path / "similar_dev.json").write_text(json.dumps({"queries": [{"query": QUERY, "candidates": [CAND_A]}]}))
    (tmp_path / "similar_dev_supplement.json").write_text(json.dumps({"queries": [{"query": QUERY, "candidates": [CAND_B]}]}))
    labels = tmp_path / "labels.jsonl"
    app = create_app("dev", records=records, pools_dir=tmp_path, labels_path=labels, images_dir=tmp_path, supplement="")
    client = TestClient(app, follow_redirects=False)
    page = client.get("/item/similar/0/0").text
    assert "Green Kurti" in page and "Red Kurti" not in page and "supplement" not in page.lower()
    token = re.search(r'name="token" value="([^"]+)"', page).group(1)
    post(client, token, grade=1)
    assert list(read_labels(labels)) == [("similar", "dev", QUERY, CAND_B)]
