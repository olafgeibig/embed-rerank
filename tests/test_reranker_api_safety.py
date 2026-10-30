"""Isolated routers and fake managers only; no application lifespan/models."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.routers import cohere_router, openai_router, reranking_router, tei_router
from app.services.reranking_service import RerankingService


ENDPOINTS = ["/api/v1/rerank/", "/v1/rerank", "/v2/rerank", "/v1/openai/rerank", "/rerank"]


def payload(path, docs, query="query"):
    body = {"query": query, "return_documents": True, "top_n": 2}
    if path == "/rerank":
        body.update(texts=docs, return_text=True, top_k=2)
    else:
        body["documents"] = docs
    return body


@pytest.fixture
def api(monkeypatch):
    calls = []
    state = SimpleNamespace(scores=None, error=None)

    async def rerank_passages(query, passages):
        calls.append((query, passages))
        if state.error:
            raise state.error
        return state.scores if state.scores is not None else [i / max(len(passages), 1) for i in range(len(passages))]

    backend = SimpleNamespace(rerank_passages=rerank_passages)
    manager = SimpleNamespace(
        is_available=lambda: True, is_ready=lambda: True,
        get_current_backend=lambda: backend,
        get_current_backend_info=lambda: {"name": "fake", "device": "cpu", "rerank_model_name": "fake"},
    )
    service = RerankingService(manager)
    app = FastAPI()
    for module in (reranking_router, cohere_router, openai_router, tei_router):
        app.include_router(module.router)
    app.dependency_overrides[reranking_router.get_reranking_service] = lambda: service
    app.dependency_overrides[cohere_router.get_reranking_service] = lambda: service
    app.dependency_overrides[tei_router.get_backend_manager] = lambda: manager

    async def get_service():
        return service

    monkeypatch.setattr(openai_router, "get_reranking_service", get_service)
    monkeypatch.setattr(settings, "rerank_max_passage_chars", 65536, raising=False)
    monkeypatch.setattr(settings, "rerank_enable_chunking", False)
    monkeypatch.setattr(settings, "openai_rerank_auto_sigmoid", False)
    return TestClient(app), calls, state


@pytest.mark.parametrize("path", ENDPOINTS)
def test_long_document_accepted_and_original_index_and_whitespace_preserved(api, path):
    client, calls, _ = api
    docs = ["  " + "a" * 6834 + "  ", "  preferred document  "]
    response = client.post(path, json=payload(path, docs))
    assert response.status_code == 200, response.text
    result = response.json()
    results = result if path == "/rerank" else result.get("results", result.get("data"))
    assert [r["index"] for r in results] == [1, 0]
    assert calls[-1] == ("query", docs)
    assert docs[0] in response.text


@pytest.mark.parametrize("path", ENDPOINTS)
def test_configured_character_guard_exact_boundary(api, monkeypatch, path):
    client, calls, _ = api
    monkeypatch.setattr(settings, "rerank_max_passage_chars", 5000)
    assert client.post(path, json=payload(path, ["a" * 5000])).status_code == 200
    calls.clear()
    response = client.post(path, json=payload(path, ["a" * 5001]))
    assert response.status_code in (400, 422), response.text
    assert not calls


@pytest.mark.parametrize("path", ENDPOINTS)
@pytest.mark.parametrize("docs,query", [(["   "], "q"), (["doc"], "   "), (["doc"], "q" * 2049)])
def test_blank_and_query_safety_guards(api, path, docs, query):
    client, calls, _ = api
    response = client.post(path, json=payload(path, docs, query))
    assert response.status_code in (400, 422), response.text
    assert not calls


def test_cohere_chunking_cannot_bypass_original_document_guards(api, monkeypatch):
    client, calls, _ = api
    monkeypatch.setattr(settings, "rerank_enable_chunking", True)
    monkeypatch.setattr(settings, "rerank_max_tokens_per_doc", 10)
    monkeypatch.setattr(settings, "rerank_max_passage_chars", 5000)
    for docs, query in [(["a" * 5001], "q"), (["  "], "q"), (["doc"], "  ")]:
        response = client.post("/v1/rerank", json=payload("/v1/rerank", docs, query))
        assert response.status_code in (400, 422), response.text
    assert not calls


def test_current_cohere_chunking_aggregates_original_indices_and_documents(api, monkeypatch):
    client, calls, state = api
    monkeypatch.setattr(settings, "rerank_enable_chunking", True)
    monkeypatch.setattr(settings, "rerank_max_tokens_per_doc", 100)
    docs = ["a" * 5000, "short"]
    response = client.post("/v2/rerank", json=payload("/v2/rerank", docs))
    assert response.status_code == 200, response.text
    results = response.json()["results"]
    assert [r["index"] for r in results] == [1, 0]
    assert [r["document"]["text"] for r in results] == [docs[1], docs[0]]
    assert len(calls[-1][1]) > 2


@pytest.mark.parametrize("path", ENDPOINTS)
def test_query_budget_value_error_reaches_http400(api, path):
    client, _, state = api
    state.error = ValueError("Reranker query/template leaves no document token space")
    response = client.post(path, json=payload(path, ["doc"]))
    assert response.status_code == 400, response.text


@pytest.mark.parametrize("path", ENDPOINTS)
def test_model_failure_is_server_error(api, path):
    client, _, state = api
    state.error = RuntimeError("MLX reranker inference failed")
    response = client.post(path, json=payload(path, ["doc"]))
    assert response.status_code in (500, 503), response.text


@pytest.mark.parametrize("path", ENDPOINTS)
@pytest.mark.parametrize("scores", [[], [0.1], [0.1, 0.2, 0.3], [0.1, float("nan")], [float("inf"), 0.2], [[0.1], [0.2]]])
def test_malformed_score_vectors_are_server_errors(api, path, scores):
    client, _, state = api
    state.scores = scores
    response = client.post(path, json=payload(path, ["doc0", "doc1"]))
    assert response.status_code in (500, 503), response.text


def test_cohere_chunking_score_mismatch_is_server_error(api, monkeypatch):
    client, _, state = api
    monkeypatch.setattr(settings, "rerank_enable_chunking", True)
    monkeypatch.setattr(settings, "rerank_max_tokens_per_doc", 100)
    state.scores = [0.5]
    response = client.post("/v1/rerank", json=payload("/v1/rerank", ["a" * 5000, "short"]))
    assert response.status_code in (500, 503), response.text


@pytest.mark.parametrize("path", ENDPOINTS)
def test_default_character_guard(api, path):
    client, calls, _ = api
    assert client.post(path, json=payload(path, ["a" * 65536])).status_code == 200
    calls.clear()
    response = client.post(path, json=payload(path, ["a" * 65537]))
    assert response.status_code in (400, 422)
    assert not calls


def test_cohere_chunk_count_guard_prevents_backend_call(api, monkeypatch):
    client, calls, _ = api
    monkeypatch.setattr(settings, "rerank_enable_chunking", True)
    monkeypatch.setattr(settings, "rerank_max_tokens_per_doc", 10)
    monkeypatch.setattr(settings, "max_passages_per_rerank", 2)
    response = client.post("/v1/rerank", json=payload("/v1/rerank", ["a" * 5000]))
    assert response.status_code == 400
    assert not calls
