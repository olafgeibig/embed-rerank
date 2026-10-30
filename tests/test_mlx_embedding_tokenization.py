from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.backends import mlx_backend


class FakeTokenizer:
    def __init__(self):
        self.normal_token_ids = None

    def __call__(
        self,
        texts,
        *,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    ):
        rows = [list(range(1, int(text) + 1)) for text in texts]
        if truncation:
            rows = [row[:max_length] for row in rows]

        if padding:
            width = max(map(len, rows))
            masks = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
            rows = [row + [0] * (width - len(row)) for row in rows]
        else:
            masks = [[1] * len(row) for row in rows]

        if padding and truncation and return_tensors == "np":
            self.normal_token_ids = [row[:] for row in rows]

        if return_tensors == "np":
            return {
                "input_ids": np.asarray(rows, dtype=np.int64),
                "attention_mask": np.asarray(masks, dtype=np.int64),
            }
        return {"input_ids": rows, "attention_mask": masks}


@pytest.mark.parametrize(
    ("max_sequence_length", "long_text_tokens", "expected_tokens", "truncated_count"),
    [
        (8192, 700, 700, 0),
        (1024, 1024, 1024, 0),
        (1024, 1200, 1024, 1),
    ],
)
def test_mlx_embedding_uses_effective_token_budget_and_reports_truncation(
    monkeypatch, max_sequence_length, long_text_tokens, expected_tokens, truncated_count
):
    tokenizer = FakeTokenizer()
    observed_batches = []

    def backbone(input_ids):
        ids = np.asarray(input_ids)
        observed_batches.append(ids.copy())
        return np.stack((ids, np.ones_like(ids)), axis=-1)

    logger = Mock()
    monkeypatch.setattr(mlx_backend, "logger", logger)
    monkeypatch.setattr(mlx_backend, "_mx_array", np.asarray)
    monkeypatch.setattr(mlx_backend, "mx", SimpleNamespace(float32=np.float32, clear_cache=lambda: None))

    backend = object.__new__(mlx_backend.MLXBackend)
    backend.model = object()
    backend._hf_tokenizer = tokenizer
    backend.max_sequence_length = max_sequence_length
    backend._get_transformer_backbone = lambda: backbone

    vectors = backend._embed_sync([str(long_text_tokens), "10"], batch_size=2)

    assert tokenizer.normal_token_ids is not None
    assert len(tokenizer.normal_token_ids[0]) == expected_tokens
    assert tokenizer.normal_token_ids[0][expected_tokens - 1] == expected_tokens
    assert observed_batches[0].shape[1] == expected_tokens
    assert vectors.shape == (2, 2)
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-7)

    if truncated_count:
        logger.warning.assert_called_once_with(
            "Embedding input truncated at backend token limit",
            truncated_count=truncated_count,
            max_input_tokens=long_text_tokens,
            max_sequence_length=max_sequence_length,
        )
    else:
        logger.warning.assert_not_called()


def test_mlx_tokenizer_adapter_preserves_ragged_unpadded_token_lists():
    adapter = mlx_backend._MLXTokenizerAdapter(
        SimpleNamespace(encode=lambda text: list(range(int(text))))
    )

    encoded = adapter(
        ["700", "10"],
        padding=False,
        truncation=False,
        return_tensors=None,
    )

    assert [len(row) for row in encoded["input_ids"]] == [700, 10]
    assert [len(row) for row in encoded["attention_mask"]] == [700, 10]
    assert encoded["input_ids"][0] == list(range(700))
    assert encoded["input_ids"][1] == list(range(10))
    assert all(all(mask_value == 1 for mask_value in mask) for mask in encoded["attention_mask"])

    default_encoded = adapter("700")
    assert default_encoded["input_ids"].shape == (1, 700)
    assert default_encoded["input_ids"][0, -1] == 699
