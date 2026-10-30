from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.backends import mlx_backend, torch_backend


@pytest.mark.parametrize(
    ("configured", "config", "tokenizer_limit", "expected"),
    [
        (8192, {"max_position_embeddings": 2048}, 10**30, 2048),
        (8192, {"text_config": {"max_position_embeddings": 3072}}, 10**30, 3072),
        (8192, {}, 1024, 1024),
        (4096, {"max_position_embeddings": 32768}, 32768, 4096),
        (2048, {}, 10**30, 2048),
    ],
)
@pytest.mark.asyncio
async def test_mlx_resolves_effective_sequence_limit_from_model_metadata(
    monkeypatch, configured, config, tokenizer_limit, expected
):
    monkeypatch.setattr(mlx_backend, "MLX_AVAILABLE", True)
    backend = mlx_backend.MLXBackend("local-test-model", max_sequence_length=configured)
    tokenizer = SimpleNamespace(model_max_length=tokenizer_limit)
    monkeypatch.setattr(
        backend,
        "_load_model_sync",
        lambda: (object(), object(), config, tokenizer),
    )

    await backend.load_model()

    assert backend.configured_max_sequence_length == configured
    assert backend.max_sequence_length == expected
    info = backend.get_model_info()
    assert info["configured_max_sequence_length"] == configured
    assert info["max_sequence_length"] == expected


@pytest.mark.asyncio
async def test_mlx_embed_validation_uses_effective_limit_for_long_text(monkeypatch):
    monkeypatch.setattr(mlx_backend, "MLX_AVAILABLE", True)
    backend = mlx_backend.MLXBackend("local-test-model", max_sequence_length=2048)
    backend.max_sequence_length = 2048
    backend._is_loaded = True
    monkeypatch.setattr(backend, "_embed_sync", lambda texts, batch_size: np.ones((len(texts), 2)))

    result = await backend.embed_texts(["x" * 10241], batch_size=1)

    assert result.vectors.shape == (1, 2)


def test_torch_model_native_limit_is_preserved_and_reported(monkeypatch):
    model = Mock()
    model.max_seq_length = 384
    monkeypatch.setattr(torch_backend, "SentenceTransformer", lambda *args, **kwargs: model)
    backend = torch_backend.TorchBackend("local-test-model", device="cpu", max_sequence_length=8192)

    loaded_model = backend._load_model_sync()

    assert loaded_model is model
    assert model.max_seq_length == 384
    assert backend.configured_max_sequence_length == 8192
    assert backend.max_sequence_length == 384
    info = backend.get_model_info()
    assert info["configured_max_sequence_length"] == 8192
    assert info["max_sequence_length"] == 384


@pytest.mark.asyncio
async def test_torch_embed_validation_uses_effective_limit(monkeypatch):
    monkeypatch.setattr(torch_backend, "SentenceTransformer", lambda *args, **kwargs: Mock())
    backend = torch_backend.TorchBackend("local-test-model", device="cpu", max_sequence_length=2048)
    backend.max_sequence_length = 2048
    backend._is_loaded = True
    monkeypatch.setattr(backend, "_embed_sync", lambda texts, batch_size: np.ones((len(texts), 2)))

    result = await backend.embed_texts(["x" * 10241], batch_size=1)

    assert result.vectors.shape == (1, 2)
