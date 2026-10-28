"""Check model selection and cache reuse without importing the ML stack."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from api import models


@pytest.fixture(autouse=True)
def clear_model_cache():
    models._load_model.cache_clear()
    yield
    models._load_model.cache_clear()


def fake_ml_stack(monkeypatch):
    pipeline = Mock()
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=pipeline))
    return pipeline


def test_local_model_is_reused_without_downloading(monkeypatch, tmp_path):
    name = "tinybert-sentiment-analysis"
    (tmp_path / name).mkdir()
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    download = Mock(side_effect=AssertionError("Use local artifacts"))
    monkeypatch.setattr(models, "download_dir_from_s3", download)
    pipeline = fake_ml_stack(monkeypatch)
    first = models.get_model(name)
    assert models.get_model(name) is first
    pipeline.assert_called_once_with("sentiment-analysis", model=str(tmp_path / name), device=-1)
    download.assert_not_called()


def test_missing_model_uses_configured_s3_prefix(monkeypatch, tmp_path):
    name = "tinybert-disaster-tweets"
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("AWS_S3_MODEL_PREFIX", "custom/models/")
    download = Mock()
    monkeypatch.setattr(models, "download_dir_from_s3", download)
    fake_ml_stack(monkeypatch)
    models.get_model(name)
    download.assert_called_once_with(tmp_path / name, f"custom/models/{name}")


def test_failed_load_can_be_retried(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    download = Mock(side_effect=[OSError("interrupted"), None])
    monkeypatch.setattr(models, "download_dir_from_s3", download)
    pipeline = fake_ml_stack(monkeypatch)
    with pytest.raises(OSError):
        models.get_model("vit-human-pose-classification")
    assert models.get_model("vit-human-pose-classification") is pipeline.return_value
    assert download.call_count == 2


def test_unknown_model_is_rejected_before_download(monkeypatch):
    download = Mock()
    monkeypatch.setattr(models, "download_dir_from_s3", download)
    with pytest.raises(ValueError, match="Unknown model"):
        models.get_model("../arbitrary-path")
    download.assert_not_called()
