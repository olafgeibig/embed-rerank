from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.models.requests import EmbedRequest
from app.services import embedding_service
from app.services.embedding_service import EmbeddingService


@pytest.fixture
def disable_model_cache_metadata(monkeypatch):
    monkeypatch.setattr(
        embedding_service.ModelMetadataExtractor,
        "get_model_cache_path",
        lambda self, model_name: None,
    )


@pytest.mark.parametrize(
    ("configured", "backend_limit", "env_override", "request_override", "expected"),
    [
        (8192, 8192, None, None, 8192),
        (4096, 4096, None, None, 4096),
        (8192, 2048, None, None, 2048),
        (8192, 8192, 4096, None, 4096),
        (8192, 8192, 4096, 1024, 1024),
        (4096, 4096, None, 8192, 4096),
    ],
)
@pytest.mark.asyncio
async def test_preprocessing_limit_is_bounded_by_config_backend_and_overrides(
    monkeypatch,
    disable_model_cache_metadata,
    configured,
    backend_limit,
    env_override,
    request_override,
    expected,
):
    backend_info = {"model_name": "test/model"}
    if backend_limit is not None:
        backend_info["max_sequence_length"] = backend_limit
    manager = SimpleNamespace(get_current_backend_info=lambda: backend_info)
    config = Settings(_env_file=None, backend="torch", max_sequence_length=configured)
    # Set explicitly so ambient legacy aliases cannot affect this case.
    config.default_max_tokens_override = env_override
    service = EmbeddingService(manager, config)
    captured = {}

    def record_processing_options(**kwargs):
        captured.update(kwargs)
        return kwargs["texts"], None

    monkeypatch.setattr(
        embedding_service.TextSummarizer,
        "process_texts_with_options",
        record_processing_options,
    )
    request = EmbedRequest(texts=["text"], max_tokens_override=request_override)

    await service._preprocess_texts(request)

    assert captured["absolute_max_tokens"] == min(
        configured, backend_limit if backend_limit is not None else configured
    )
    assert captured["max_tokens"] == expected
    assert captured["return_processing_info"] is True


@pytest.mark.parametrize("legacy_override", [0, -1])
@pytest.mark.asyncio
async def test_nonpositive_legacy_override_is_ignored(
    monkeypatch, disable_model_cache_metadata, legacy_override
):
    manager = SimpleNamespace(
        get_current_backend_info=lambda: {"model_name": "test/model", "max_sequence_length": 8192}
    )
    config = Settings(_env_file=None, backend="torch", max_sequence_length=4096)
    config.default_max_tokens_override = legacy_override
    service = EmbeddingService(manager, config)
    captured = {}

    def record_processing_options(**kwargs):
        captured.update(kwargs)
        return kwargs["texts"], None

    monkeypatch.setattr(
        embedding_service.TextSummarizer,
        "process_texts_with_options",
        record_processing_options,
    )

    await service._preprocess_texts(EmbedRequest(texts=["text"]))

    assert captured["max_tokens"] == 4096
    assert captured["absolute_max_tokens"] == 4096


@pytest.mark.asyncio
async def test_model_metadata_absolute_limit_is_respected(monkeypatch):
    metadata_path = Path("/fake/model-cache")
    monkeypatch.setattr(
        embedding_service.ModelMetadataExtractor,
        "get_model_cache_path",
        lambda self, model_name: metadata_path,
    )
    monkeypatch.setattr(
        embedding_service.ModelMetadataExtractor,
        "extract_metadata_from_path",
        lambda self, model_path: {
            "recommended_max_tokens": 3072,
            "absolute_max_tokens": 2048,
        },
    )
    manager = SimpleNamespace(
        get_current_backend_info=lambda: {"model_name": "test/model", "max_sequence_length": 8192}
    )
    config = Settings(_env_file=None, backend="torch", max_sequence_length=8192)
    config.default_max_tokens_override = None
    service = EmbeddingService(manager, config)
    captured = {}

    def record_processing_options(**kwargs):
        captured.update(kwargs)
        return kwargs["texts"], None

    monkeypatch.setattr(
        embedding_service.TextSummarizer,
        "process_texts_with_options",
        record_processing_options,
    )

    await service._preprocess_texts(EmbedRequest(texts=["text"]))

    assert captured["absolute_max_tokens"] == 2048
    assert captured["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_long_german_text_is_not_truncated_at_recommended_fallback_limit(
    disable_model_cache_metadata,
):
    manager = SimpleNamespace(
        get_current_backend_info=lambda: {"model_name": "test/model", "max_sequence_length": 8192}
    )
    config = Settings(_env_file=None, backend="torch", max_sequence_length=8192)
    config.default_max_tokens_override = None
    service = EmbeddingService(manager, config)
    text = "Abschnitt mit deutschem Inhalt. " * 400

    processed_texts, processing_info = await service._preprocess_texts(EmbedRequest(texts=[text]))

    assert processed_texts == [text]
    assert processing_info is None


@pytest.mark.asyncio
async def test_exactly_6000_char_text_with_marker_is_preserved(disable_model_cache_metadata):
    manager = SimpleNamespace(
        get_current_backend_info=lambda: {"model_name": "test/model", "max_sequence_length": 8192}
    )
    config = Settings(_env_file=None, backend="torch", max_sequence_length=8192)
    config.default_max_tokens_override = None
    service = EmbeddingService(manager, config)
    marker = "END-MARKER-6000"
    text = "x" * (6000 - len(marker)) + marker
    assert len(text) == 6000

    processed_texts, _ = await service._preprocess_texts(EmbedRequest(texts=[text]))

    assert processed_texts == [text]
    assert processed_texts[0].endswith(marker)


@pytest.mark.asyncio
async def test_truncation_warning_is_logged_without_returning_processing_info(
    caplog, disable_model_cache_metadata
):
    manager = SimpleNamespace(
        get_current_backend_info=lambda: {"model_name": "test/model", "max_sequence_length": 8192}
    )
    config = Settings(_env_file=None, backend="torch", max_sequence_length=8192)
    config.default_max_tokens_override = None
    service = EmbeddingService(manager, config)
    text = "Sensitive content. " * 1000
    caplog.set_level("WARNING", logger=embedding_service.__name__)

    processed_texts, processing_info = await service._preprocess_texts(
        EmbedRequest(texts=[text], max_tokens_override=1024)
    )

    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == embedding_service.__name__ and record.levelname == "WARNING"
    ]
    assert any("1/1 texts truncated" in message for message in warning_messages)
    assert any("max_tokens=1024" in message for message in warning_messages)
    assert all("Sensitive content" not in message for message in warning_messages)
    assert len(processed_texts[0]) < len(text)
    assert processing_info is None
