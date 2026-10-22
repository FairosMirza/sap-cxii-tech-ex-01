import json
from pathlib import Path

import numpy as np
import pytest

from app.embeddings import EMBEDDING_MODELS, EmbeddingSpec, TextEncoder, l2_normalize, pool, read_pooling

ARTIFACT_MODELS = Path(__file__).resolve().parents[1] / "artifacts" / "models"
CLS = {"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": False, "pooling_mode_max_tokens": False}
MEAN = {"pooling_mode_cls_token": False, "pooling_mode_mean_tokens": True, "pooling_mode_max_tokens": False}


def model_dir(path: Path, modules: list[str], pooling: dict, dim: int = 4) -> Path:
    entries = [{"idx": i, "name": str(i), "path": "" if kind == "Transformer" else f"{i}_{kind}",
                "type": f"sentence_transformers.models.{kind}"} for i, kind in enumerate(modules)]
    (path / "modules.json").write_text(json.dumps(entries))
    (path / "1_Pooling").mkdir()
    (path / "1_Pooling" / "config.json").write_text(json.dumps({"word_embedding_dimension": dim, **pooling}))
    return path


@pytest.mark.parametrize("pooling, expected", [(CLS, "cls"), (MEAN, "mean")])
def test_pooling_comes_from_the_model_config(tmp_path, pooling, expected):
    assert read_pooling(model_dir(tmp_path, ["Transformer", "Pooling", "Normalize"], pooling), 4) == expected


@pytest.mark.parametrize("pooling", [{**CLS, "pooling_mode_mean_tokens": True},
                                     {**MEAN, "pooling_mode_mean_tokens": False, "pooling_mode_max_tokens": True}, {}])
def test_pooling_modes_not_reproduced_are_rejected(tmp_path, pooling):
    with pytest.raises(ValueError, match="pooling"):
        read_pooling(model_dir(tmp_path, ["Transformer", "Pooling"], pooling), 4)


def test_extra_modules_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="modules"):
        read_pooling(model_dir(tmp_path, ["Transformer", "Pooling", "Dense", "Normalize"], CLS), 4)


def test_dimension_must_match_the_spec(tmp_path):
    with pytest.raises(ValueError, match="dimension"):
        read_pooling(model_dir(tmp_path, ["Transformer", "Pooling"], CLS, dim=8), 4)


def test_cls_pooling_takes_the_first_token():
    tokens = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    assert np.array_equal(pool(tokens, np.ones((2, 3)), "cls"), tokens[:, 0])


def test_mean_pooling_ignores_padding():
    tokens = np.array([[[1.0, 1.0], [3.0, 5.0], [100.0, 100.0]]], dtype=np.float32)
    assert np.allclose(pool(tokens, np.array([[1, 1, 0]]), "mean"), [[2.0, 3.0]])


def test_l2_normalize_gives_unit_rows_and_keeps_zero_rows():
    assert np.allclose(l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]])), [[0.6, 0.8], [0.0, 0.0]])


def test_lowercasing_model_needs_a_lowercasing_tokenizer(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    model_dir(tmp_path, ["Transformer", "Pooling"], CLS)
    Tokenizer(WordLevel({"[PAD]": 0, "a": 1}, unk_token="[PAD]")).save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "sentence_bert_config.json").write_text(json.dumps({"do_lower_case": True}))
    with pytest.raises(ValueError, match="lowercase"):
        TextEncoder(EmbeddingSpec("tiny", "test/tiny", "0" * 40, 4), local_path=tmp_path)


def test_every_model_pins_a_full_commit():
    for spec in EMBEDDING_MODELS.values():
        assert len(spec.revision) == 40 and set(spec.revision) <= set("0123456789abcdef")


@pytest.mark.skipif(not (ARTIFACT_MODELS / "bge-small").exists(), reason="artifacts not built")
def test_bge_uses_cls_pooling_from_its_own_config():
    assert read_pooling(ARTIFACT_MODELS / "bge-small", 384) == "cls"
