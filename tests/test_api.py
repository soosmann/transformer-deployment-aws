"""Exercise the HTTP contract without model weights or AWS access."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import api_main


@pytest.fixture
def client():
    return TestClient(api_main.app)


def test_docs_and_health_do_not_load_models(client, monkeypatch):
    loader = Mock(side_effect=AssertionError("No model should be loaded"))
    monkeypatch.setattr(api_main, "get_model", loader)
    assert client.get("/health").json() == {
        "status": "ok",
        "model_loading": "on_first_request",
    }
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    loader.assert_not_called()


@pytest.mark.parametrize(
    ("route", "model_name"),
    [
        ("sentiment_analysis", "tinybert-sentiment-analysis"),
        ("disaster_classifier", "tinybert-disaster-tweets"),
    ],
)
def test_text_predictions_preserve_batch_order(client, monkeypatch, route, model_name):
    classifier = Mock(
        return_value=[
            {"label": "positive", "score": 0.97},
            {"label": "negative", "score": 0.83},
        ]
    )
    loader = Mock(return_value=classifier)
    monkeypatch.setattr(api_main, "get_model", loader)
    response = client.post(
        f"/api/v1/{route}",
        json={
            "texts": ["great film", "poor film"],
            "user_id": "demo@example.com",
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["pred_model"] == model_name
    assert result["texts"] == ["great film", "poor film"]
    assert result["labels"] == ["positive", "negative"]
    assert result["scores"] == [0.97, 0.83]
    assert isinstance(result["prediction_time"], int)
    assert result["prediction_time"] >= 0
    loader.assert_called_once_with(model_name)
    classifier.assert_called_once_with(["great film", "poor film"], truncation=True)


@pytest.mark.parametrize(
    "payload",
    [
        {"texts": [], "user_id": "demo@example.com"},
        {"texts": ["  "], "user_id": "demo@example.com"},
        {"texts": ["a" * 10001], "user_id": "demo@example.com"},
        {"texts": ["text"] * 33, "user_id": "demo@example.com"},
        {"texts": ["text"], "user_id": "invalid"},
        {"texts": ["text"]},
    ],
)
def test_invalid_text_requests_do_not_load_models(client, monkeypatch, payload):
    loader = Mock()
    monkeypatch.setattr(api_main, "get_model", loader)
    assert client.post("/api/v1/sentiment_analysis", json=payload).status_code == 422
    loader.assert_not_called()


def test_pose_prediction_returns_top_label_per_image(client, monkeypatch):
    classifier = Mock(
        return_value=[
            [{"label": "sitting", "score": 0.92}, {"label": "standing", "score": 0.08}],
            [{"label": "running", "score": 0.84}],
        ]
    )
    monkeypatch.setattr(api_main, "get_model", Mock(return_value=classifier))
    urls = ["https://example.com/one.jpg", "https://example.com/two.jpg"]
    response = client.post(
        "/api/v1/pose_classifier",
        json={
            "urls": urls,
            "user_id": "demo@example.com",
        },
    )
    assert response.status_code == 200
    assert response.json()["labels"] == ["sitting", "running"]
    assert response.json()["urls"] == urls
    classifier.assert_called_once_with(urls, timeout=10)


@pytest.mark.parametrize(
    "urls",
    [[], ["file:///etc/passwd"], ["not-a-url"], ["https://example.com/a.jpg"] * 9],
)
def test_invalid_image_requests(client, urls):
    response = client.post(
        "/api/v1/pose_classifier",
        json={
            "urls": urls,
            "user_id": "demo@example.com",
        },
    )
    assert response.status_code == 422


def test_model_load_failure_is_actionable_without_leaking_details(client, monkeypatch):
    monkeypatch.setattr(
        api_main, "get_model", Mock(side_effect=RuntimeError("private detail"))
    )
    response = client.post(
        "/api/v1/sentiment_analysis",
        json={
            "texts": ["hello"],
            "user_id": "demo@example.com",
        },
    )
    assert response.status_code == 503
    assert "MODEL_DIR" in response.json()["detail"]
    assert "private detail" not in response.text


def test_unreadable_image_returns_validation_error(client, monkeypatch):
    monkeypatch.setattr(
        api_main, "get_model", Mock(return_value=Mock(side_effect=OSError))
    )
    response = client.post(
        "/api/v1/pose_classifier",
        json={
            "urls": ["https://example.com/missing.jpg"],
            "user_id": "demo@example.com",
        },
    )
    assert response.status_code == 422
