import pytest

from app.utils.embedding_limits import resolve_embedding_limit


@pytest.mark.parametrize(
    ("configured", "model_limits", "expected"),
    [
        (8192, (), 8192),
        (4096, (32768,), 4096),
        (8192, (2048, 32768), 2048),
        (16384, (32768,), 16384),
        (8192, (None, 0, -1, "unknown", 10**30), 8192),
        (8192, (True, False, 2048), 2048),
    ],
)
def test_resolve_embedding_limit(configured, model_limits, expected):
    assert resolve_embedding_limit(configured, *model_limits) == expected


@pytest.mark.parametrize("configured", [0, -1])
def test_resolve_embedding_limit_rejects_nonpositive_configured(configured):
    with pytest.raises(ValueError):
        resolve_embedding_limit(configured)
