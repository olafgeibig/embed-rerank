import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import health_router


class FakeRerankerManager:
    def __init__(self):
        self._ready = True

    def is_ready(self):
        return self._ready

    def get_current_backend_info(self):
        # Mimic the current MLXCrossEncoderBackend manager metadata.
        return {
            "name": "MLXCrossEncoderBackend",
            "status": "ready",
            "model_name": "fake/mlx-reranker",
            "device": "mlx",
            "rerank_method": "cross-encoder",
        }


@pytest.fixture
def client_with_fake_reranker(monkeypatch):
    # Isolate the router from application startup and real model loading.
    class FakeEmbeddingManager:
        def is_ready(self):
            return True

        async def health_check(self):
            return {"backend": "FakeEmbeddingBackend", "status": "healthy", "model_loaded": True}

    monkeypatch.setattr(health_router, "_reranker_backend_manager", FakeRerankerManager())
    monkeypatch.setattr(health_router, "_embedding_service", None)
    app = FastAPI()
    app.include_router(health_router.router)
    app.dependency_overrides[health_router.get_backend_manager] = lambda: FakeEmbeddingManager()
    with TestClient(app) as client:
        yield client


def test_health_includes_reranker_block(client_with_fake_reranker):
    resp = client_with_fake_reranker.get("/health/")
    assert resp.status_code == 200
    data = resp.json()
    rk = data["reranker"]
    assert rk["name"] == "MLXCrossEncoderBackend"
    assert rk["type"] == "mlx"
    assert rk["model_name"] == "fake/mlx-reranker"
    assert rk["method"] == "cross-encoder"
    assert rk["pooling"] is None
    assert rk["score_norm"] is None
