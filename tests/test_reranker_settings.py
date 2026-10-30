import pytest
from pydantic import ValidationError
from app.config import Settings


DEFAULTS = {"rerank_max_seq_len": 8192, "rerank_max_passage_tokens": 6144, "rerank_max_passage_chars": 65536, "rerank_batch_size": 1}


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for key in DEFAULTS:
        monkeypatch.delenv(key.upper(), raising=False)
        monkeypatch.delenv(key, raising=False)


def test_operational_defaults():
    s = Settings(_env_file=None, backend="torch")
    assert {k: getattr(s, k, None) for k in DEFAULTS} == DEFAULTS
    assert s.rerank_max_tokens_per_doc == 4096


@pytest.mark.parametrize("field", list(DEFAULTS))
def test_dotenv_and_env_override(tmp_path, monkeypatch, field):
    path = tmp_path / "settings.env"
    path.write_text(f"{field.upper()}=123\n")
    assert getattr(Settings(_env_file=path, backend="torch"), field, None) == 123
    monkeypatch.setenv(field.upper(), "456")
    assert getattr(Settings(_env_file=path, backend="torch"), field, None) == 456


@pytest.mark.parametrize("field", list(DEFAULTS))
@pytest.mark.parametrize("value", ["0", "-1", "not-int"])
def test_bad_limits_rejected(monkeypatch, field, value):
    monkeypatch.setenv(field.upper(), value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, backend="torch")


@pytest.mark.parametrize("field", list(DEFAULTS))
def test_blank_uses_default(monkeypatch, field):
    monkeypatch.setenv(field.upper(), "")
    assert getattr(Settings(_env_file=None, backend="torch"), field, None) == DEFAULTS[field]
