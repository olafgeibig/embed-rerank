"""CPU-only model doubles: never load weights or execute MLX inference."""
from types import SimpleNamespace

import numpy as np
import pytest

from app.backends import mlx_reranker_backend as module


class Tokenizer:
    pad_token_id = 0
    model_max_length = 10**30
    unk_token_id = 99

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class Head:
    def __init__(self):
        self.shapes = []

    def __call__(self, hidden):
        self.shapes.append(hidden.shape)
        assert hidden.ndim == 2, "only final hidden states may reach vocabulary projection"
        return np.concatenate([hidden, -hidden], axis=-1)

    as_linear = __call__


class Model:
    def __init__(self, tied=True):
        self.inputs = []
        self.head = Head()
        self.args = SimpleNamespace(tie_word_embeddings=tied)
        self.model = self.hidden
        # A callable object with the Qwen3 .embed_tokens API.
        self.model = SimpleNamespaceModel(self)
        self.lm_head = self.head

    def hidden(self, ids):
        self.inputs.append(ids.copy())
        # Causal states depend on both prefix and absolute position. Left padding
        # or gathering the padded tail changes the score.
        return (np.cumsum(ids, axis=1) / 1000 + np.arange(ids.shape[1])[None, :] / 100)[..., None]

    def __call__(self, ids):
        raise AssertionError("full sequence vocabulary projection is forbidden")


class SimpleNamespaceModel:
    def __init__(self, parent):
        self.parent = parent
        self.embed_tokens = parent.head

    def __call__(self, ids):
        return self.parent.hidden(ids)


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(module, "MLX_AVAILABLE", True)
    monkeypatch.setattr(module, "mx", SimpleNamespace(array=np.array, float32=np.float32, clear_cache=lambda: None))
    b = module.MLXCrossEncoderBackend("Qwen3-Reranker-test", batch_size=2, max_length=8192)
    b.model = Model()
    b._hf_tokenizer = Tokenizer()
    b._prefix_tokens = [7, 8]
    b._suffix_tokens = [9, 10]
    b._token_yes_id = 0
    b._token_no_id = 1
    b.config = {"max_position_embeddings": 40960}
    b._is_loaded = True
    yield b
    b._executor.shutdown()


@pytest.mark.parametrize("tied", [True, False])
def test_right_padding_final_state_projection_and_batch_invariance(backend, tied):
    backend.model = Model(tied)
    alone = backend._rerank_sync("q", ["short"])[0]
    mixed = backend._rerank_sync("q", ["short", "a longer passage"])
    assert mixed[0] == pytest.approx(alone, abs=1e-7)
    ids = backend.model.inputs[-1]
    short_len = len(backend._prefix_tokens + backend._hf_tokenizer.encode(backend._format_pair("q", "short")) + backend._suffix_tokens)
    assert ids[0, short_len - 1] == 10
    assert np.all(ids[0, short_len:] == 0)
    assert backend.model.head.shapes[-1] == (2, 1)


@pytest.mark.parametrize("yes,no", [(None, 1), (0, 0), (-1, 1), (99, 1), (0, 100)])
def test_yes_no_ids_must_be_valid_distinct(backend, yes, no):
    with pytest.raises(RuntimeError, match="yes/no"):
        backend._validate_score_token_ids(yes, no, vocab_size=100)


def test_exact_document_cap_preserves_query_template_suffix(backend, monkeypatch):
    warnings = []
    monkeypatch.setattr(module, "logger", SimpleNamespace(warning=lambda *a, **k: warnings.append((a, k))))
    backend._passage_max_tokens = 5
    query = "PRIVATE_QUERY"
    doc = "PRIVATE_DOCUMENT"
    ids = backend._build_pair_ids(query, doc)
    expected = backend._prefix_tokens + backend._hf_tokenizer.encode(backend._format_pair(query, ""))
    assert ids == expected + backend._hf_tokenizer.encode(doc)[:5] + backend._suffix_tokens
    assert len(warnings) == 1
    assert all(isinstance(v, int) for v in warnings[0][1].values())
    assert query not in str(warnings) and doc not in str(warnings)


def test_pair_budget_boundary_and_query_rejection(backend):
    overhead = len(backend._prefix_tokens + backend._hf_tokenizer.encode(backend._format_pair("q", "")) + backend._suffix_tokens)
    backend._pair_max_len = overhead + 1
    ids = backend._build_pair_ids("q", "abcd")
    assert len(ids) == overhead + 1
    assert ids[-3:] == [ord("a"), 9, 10]
    backend._pair_max_len = overhead
    with pytest.raises(ValueError, match="no document token"):
        backend._build_pair_ids("q", "a")


@pytest.mark.parametrize("native,tokenizer,expected", [(40960, 10**30, 32768), (4096, 8192, 4096), (40960, 2048, 2048)])
def test_effective_capacity_is_conservative(backend, native, tokenizer, expected):
    backend._pair_max_len = 50000
    backend.config["max_position_embeddings"] = native
    backend._hf_tokenizer.model_max_length = tokenizer
    info = backend.get_model_info()
    assert info["configured_max_sequence_length"] == 50000
    assert info["effective_max_sequence_length"] == expected
    assert info["configured_max_passage_tokens"] == 6144
    assert info["effective_max_passage_tokens"] == min(6144, expected)


def test_no_truncation_at_exact_document_boundary(backend, monkeypatch):
    monkeypatch.setattr(module, "logger", SimpleNamespace(warning=lambda *a, **k: pytest.fail("unexpected warning")))
    backend._passage_max_tokens = 4
    ids = backend._build_pair_ids("q", "abcd")
    assert ids[-6:-2] == [ord(c) for c in "abcd"]


@pytest.mark.asyncio
async def test_query_budget_error_remains_value_error(backend):
    backend._pair_max_len = 1
    with pytest.raises(ValueError, match="no document token"):
        await backend.rerank_passages("q", ["doc"])


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("model failure"), ValueError("bad model tensor")])
async def test_model_failure_never_returns_lexical_scores(backend, monkeypatch, error):
    def broken(*args):
        raise error
    monkeypatch.setattr(backend.model.model, "__call__", broken)
    monkeypatch.setattr(backend.model, "hidden", broken)
    with pytest.raises(RuntimeError, match="MLX reranker inference failed"):
        await backend.rerank_passages("q", ["q"])


@pytest.mark.asyncio
@pytest.mark.parametrize("scores", [[], [float("nan")], [float("inf")], [0.1, 0.2]])
async def test_backend_rejects_bad_scores(backend, monkeypatch, scores):
    monkeypatch.setattr(backend, "_rerank_sync", lambda *a: scores)
    with pytest.raises(RuntimeError, match="score"):
        await backend.rerank_passages("q", ["doc"])


def test_unsupported_projection_fails_closed(backend):
    backend.model = lambda ids: np.zeros((*ids.shape, 2))
    with pytest.raises(RuntimeError, match="final-state projection"):
        backend._rerank_sync("q", ["doc"])


def test_8k_pair_only_projects_final_states(backend):
    backend._passage_max_tokens = 10000
    scores = backend._rerank_sync("q", ["short", "d" * 10000])
    assert len(scores) == 2
    assert backend.model.inputs[-1].shape == (2, 8192)
    assert backend.model.head.shapes == [(2, 1)]


def test_valid_yes_no_ids_and_normal_score_arithmetic(backend):
    backend._validate_score_token_ids(0, 1, vocab_size=2)
    scores = backend._rerank_sync("q", ["doc"])
    ids = backend.model.inputs[-1]
    final = np.cumsum(ids, axis=1)[0, -1] / 1000 + (ids.shape[1] - 1) / 100
    expected = 1 / (1 + np.exp(-2 * final))
    assert scores == pytest.approx([expected], abs=1e-7)


def test_batch_order_and_repeated_short_inputs_are_stable(backend):
    alone = backend._rerank_sync("q", ["short"])[0]
    scores = backend._rerank_sync("q", ["much longer passage", "short", "short"])
    assert scores[1:] == pytest.approx([alone, alone], abs=1e-7)


def test_native_cap_applies_to_real_encoded_pair(backend):
    overhead = len(backend._build_pair_ids("q", "a")) - 1
    backend.config["max_position_embeddings"] = overhead + 2
    assert len(backend._build_pair_ids("q", "abcdef")) == overhead + 2


def test_generic_model_has_no_invented_qwen_capacity(backend):
    backend.model_name = "other/model"
    backend._pair_max_len = 50000
    assert backend._effective_pair_limit() == 40960


def test_local_qwen3_checkpoint_still_uses_official_capacity(backend):
    backend.model_name = "/models/local-reranker"
    backend.config["model_type"] = "qwen3"
    backend._pair_max_len = 50000
    assert backend._effective_pair_limit() == 32768


@pytest.mark.parametrize("field", ["batch_size", "max_length", "max_passage_tokens"])
@pytest.mark.parametrize("value", [0, -1])
def test_constructor_rejects_nonpositive_limits(backend, field, value):
    with pytest.raises(ValueError, match="positive"):
        module.MLXCrossEncoderBackend("test", **{field: value})


def test_factory_wires_operational_settings_without_loading(backend, monkeypatch):
    from app.backends import factory
    captured = {}
    def fake_backend(model, **kwargs):
        captured.update(kwargs)
        return captured
    monkeypatch.setattr(factory, "MLX_RERANK_AVAILABLE", True)
    monkeypatch.setattr(factory.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(factory.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(factory, "MLXCrossEncoderBackend", fake_backend)
    for key, value in [("rerank_max_seq_len", 8192), ("rerank_batch_size", 1), ("rerank_max_passage_tokens", 6144)]:
        monkeypatch.setattr(factory.settings, key, value)
    factory.BackendFactory.create_reranker_backend("mlx", "test")
    assert (captured["max_length"], captured["batch_size"], captured["max_passage_tokens"]) == (8192, 1, 6144)
    factory.BackendFactory.create_reranker_backend("mlx", "test", max_length=1024, batch_size=3, max_passage_tokens=512)
    assert (captured["max_length"], captured["batch_size"], captured["max_passage_tokens"]) == (1024, 3, 512)


def test_load_configuration_from_dataclass_args_with_fake_loader_only(backend, monkeypatch):
    from dataclasses import dataclass
    import transformers
    import huggingface_hub

    @dataclass
    class Args:
        model_type: str = "qwen3"
        max_position_embeddings: int = 1024
        hidden_size: int = 4
        vocab_size: int = 100
        tie_word_embeddings: bool = True

    model = Model()
    model.args = Args()
    tokenizer = Tokenizer()
    tokenizer.convert_tokens_to_ids = lambda text: {"yes": 0, "no": 1}[text]
    monkeypatch.setattr(module, "mlx_lm", SimpleNamespace(load=lambda name: (model, tokenizer)))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    cache_lookups = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **k: cache_lookups.append(k))
    result = backend._load_sync()
    assert result[2]["max_position_embeddings"] == 1024
    assert result[2]["hidden_size"] == 4
    assert not cache_lookups
