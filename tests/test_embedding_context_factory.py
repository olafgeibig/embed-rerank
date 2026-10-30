import pytest

from app.backends import factory


@pytest.mark.parametrize("backend_type", ["mlx", "torch"])
def test_factory_passes_configured_sequence_length(monkeypatch, backend_type):
    constructor_calls = []

    def fake_backend(*args, **kwargs):
        constructor_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(factory.settings, "max_sequence_length", 4096)
    monkeypatch.setattr(factory, "MLXBackend", fake_backend)
    monkeypatch.setattr(factory, "TorchBackend", fake_backend)
    monkeypatch.setattr(factory, "MLX_AVAILABLE", True)
    monkeypatch.setattr(factory.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(factory.platform, "machine", lambda: "arm64")

    factory.BackendFactory.create_backend(backend_type, "local-test-model")

    assert len(constructor_calls) == 1
    assert constructor_calls[0][1]["max_sequence_length"] == 4096


@pytest.mark.parametrize("backend_type", ["mlx", "torch"])
def test_factory_passes_explicit_sequence_length_override(monkeypatch, backend_type):
    constructor_calls = []

    def fake_backend(*args, **kwargs):
        constructor_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(factory.settings, "max_sequence_length", 4096)
    monkeypatch.setattr(factory, "MLXBackend", fake_backend)
    monkeypatch.setattr(factory, "TorchBackend", fake_backend)
    monkeypatch.setattr(factory, "MLX_AVAILABLE", True)
    monkeypatch.setattr(factory.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(factory.platform, "machine", lambda: "arm64")

    factory.BackendFactory.create_backend(
        backend_type, "local-test-model", max_sequence_length=2048
    )

    assert constructor_calls[0][1]["max_sequence_length"] == 2048