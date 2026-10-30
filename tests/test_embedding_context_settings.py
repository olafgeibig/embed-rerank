import pytest
from pydantic import ValidationError

from app.config import Settings


@pytest.fixture(autouse=True)
def clear_max_sequence_length_environment(monkeypatch):
    monkeypatch.delenv("MAX_SEQUENCE_LENGTH", raising=False)
    monkeypatch.delenv("max_sequence_length", raising=False)


def test_max_sequence_length_defaults_to_8192():
    settings = Settings(_env_file=None, backend="torch")

    assert settings.max_sequence_length == 8192


def test_max_sequence_length_can_be_configured_in_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("MAX_SEQUENCE_LENGTH=4096\n")

    settings = Settings(_env_file=env_file, backend="torch")

    assert settings.max_sequence_length == 4096


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_max_sequence_length_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("MAX_SEQUENCE_LENGTH", value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, backend="torch")
