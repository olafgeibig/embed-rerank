"""
MLX Cross-Encoder Reranker Backend

Uses mlx-lm to run full transformer forward passes for cross-encoder reranking.
Supports Qwen3-Reranker models that use the yes/no token scoring approach:
the model predicts whether a document is relevant to a query by outputting
logits for "yes" and "no" tokens, and the relevance score is derived from
the softmax probability of the "yes" logit.

Target model example: mlx-community/Qwen3-Reranker-0.6B-mxfp8
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.logger import setup_logging
from ..utils.rerank_validation import RerankQueryBudgetError, validate_rerank_scores
from .base import BaseBackend, EmbeddingResult

logger = setup_logging()

try:
    import os

    import mlx.core as mx
    import mlx_lm

    MLX_AVAILABLE = True
    logger.info("MLX reranker modules imported")
except ImportError as e:
    MLX_AVAILABLE = False
    logger.warning("MLX not available for reranker", error=str(e))
    mx = None  # type: ignore
    mlx_lm = None  # type: ignore


def _mx_array(x):
    if not MLX_AVAILABLE or mx is None:
        import numpy as _np

        return _np.array(x)
    if hasattr(mx, "array"):
        try:
            return mx.array(x)
        except Exception:
            pass
    if hasattr(mx, "asarray"):
        try:
            return mx.asarray(x)
        except Exception:
            pass
    if hasattr(mx, "numpy") and hasattr(mx.numpy, "array"):
        try:
            return mx.numpy.array(x)
        except Exception:
            pass
    import numpy as _np

    return _np.array(x)


_DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

_SYSTEM_MSG = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)


class MLXCrossEncoderBackend(BaseBackend):
    """MLX cross-encoder reranker using full transformer inference via mlx-lm."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 1,
        max_length: int = 8192,
        pooling: str = "mean",
        score_norm: str = "none",
        max_passage_tokens: int = 6144,
    ):
        if not MLX_AVAILABLE:
            raise RuntimeError(
                "MLX backend requested but MLX is not available.\n"
                "Install mlx and mlx-lm, and ensure Apple Silicon (arm64)."
            )

        super().__init__(model_name, device or "mlx")
        budgets = (batch_size, max_length, max_passage_tokens)
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in budgets):
            raise ValueError("Reranker batch size and token budgets must be positive integers")
        self._batch_size = batch_size
        self._pair_max_len = max_length
        self._passage_max_tokens = max_passage_tokens
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="MLX-Rerank")
        self.model = None
        self.tokenizer = None
        self.config = None
        self._hf_tokenizer = None
        self._token_yes_id = None
        self._token_no_id = None
        self._prefix_tokens = None
        self._suffix_tokens = None

    async def load_model(self) -> None:
        if self._is_loaded:
            return

        logger.info("Loading MLX reranker model via mlx-lm", model_name=self.model_name)
        start_time = time.time()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self._executor, self._load_sync)
            (
                self.model,
                self.tokenizer,
                self.config,
                self._hf_tokenizer,
                self._token_yes_id,
                self._token_no_id,
                self._prefix_tokens,
                self._suffix_tokens,
            ) = result

            self._load_time = time.time() - start_time
            self._is_loaded = True

            logger.info(
                "MLX reranker model loaded",
                model_name=self.model_name,
                load_time=self._load_time,
                token_yes_id=self._token_yes_id,
                token_no_id=self._token_no_id,
            )
        except Exception as e:
            logger.error("Failed to load MLX reranker model", model_name=self.model_name, error=str(e))
            raise RuntimeError(f"MLX reranker loading failed for {self.model_name}: {e}")

    def _load_sync(self):
        import json
        import os

        model, mlx_tokenizer = mlx_lm.load(self.model_name)
        logger.info("mlx-lm reranker model loaded", model_type=type(model).__name__)

        config = {}
        if hasattr(model, "args") and hasattr(model.args, "to_dict"):
            config = model.args.to_dict()
        elif hasattr(model, "args") and is_dataclass(model.args):
            config = asdict(model.args)
        elif hasattr(model, "args") and isinstance(model.args, dict):
            config = model.args
        else:
            config_path = None
            try:
                from huggingface_hub import snapshot_download

                cache_dir = snapshot_download(
                    repo_id=self.model_name,
                    allow_patterns=["config.json"],
                )
                config_path = os.path.join(cache_dir, "config.json")
            except Exception:
                pass
            if config_path and os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                except Exception:
                    config = {}

        if "hidden_size" not in config and "d_model" in config:
            config["hidden_size"] = config["d_model"]
        if "hidden_size" not in config:
            config["hidden_size"] = getattr(model, "hidden_size", 1024)

        hf_tokenizer = None
        try:
            from transformers import AutoTokenizer

            hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception:
            pass

        if hf_tokenizer is None:
            hf_tokenizer = mlx_tokenizer

        if hasattr(hf_tokenizer, "pad_token") and hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token

        token_yes_id = hf_tokenizer.convert_tokens_to_ids("yes")
        token_no_id = hf_tokenizer.convert_tokens_to_ids("no")
        self._hf_tokenizer = hf_tokenizer
        vocab_size = config.get("vocab_size") or getattr(hf_tokenizer, "vocab_size", None)
        self._validate_score_token_ids(token_yes_id, token_no_id, vocab_size)

        prefix = "<|im_start|>system\n" + _SYSTEM_MSG + "<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        prefix_tokens = hf_tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = hf_tokenizer.encode(suffix, add_special_tokens=False)

        return model, mlx_tokenizer, config, hf_tokenizer, token_yes_id, token_no_id, prefix_tokens, suffix_tokens

    def _format_pair(self, query: str, doc: str, instruction: str = _DEFAULT_INSTRUCTION) -> str:
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _effective_pair_limit(self):
        limits = [self._pair_max_len]
        config = self.config or {}
        for key in ("max_position_embeddings", "n_positions", "max_sequence_length", "max_seq_len"):
            value = config.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                limits.append(value)
        native = getattr(self._hf_tokenizer, "model_max_length", None)
        # Hugging Face uses enormous integers to mean "unknown".
        if isinstance(native, int) and 0 < native < 1_000_000_000:
            limits.append(native)
        name = self.model_name.lower()
        if ("qwen3" in name and "reranker" in name) or config.get("model_type") == "qwen3":
            limits.append(32768)  # Official 32k context, even when checkpoint config says 40960.
        return min(limits)

    def _build_pair_ids(self, query, doc):
        # Tokenize the immutable header separately: query/instruction/template
        # are never cut to make room for a document. Only document IDs truncate.
        header = self._hf_tokenizer.encode(self._format_pair(query, ""), add_special_tokens=False)
        overhead = len(self._prefix_tokens) + len(header) + len(self._suffix_tokens)
        pair_limit = self._effective_pair_limit()
        remaining = pair_limit - overhead
        if remaining <= 0:
            raise RerankQueryBudgetError(
                f"Reranker query/template leaves no document token space: {overhead} >= {pair_limit}"
            )
        doc_limit = min(self._passage_max_tokens, remaining)
        doc_ids = self._hf_tokenizer.encode(doc, add_special_tokens=False)
        if len(doc_ids) > doc_limit:
            logger.warning(
                "Reranker document truncated",
                original_tokens=len(doc_ids),
                retained_tokens=doc_limit,
                pair_limit=pair_limit,
                overhead_tokens=overhead,
            )
        return self._prefix_tokens + header + doc_ids[:doc_limit] + self._suffix_tokens

    def _validate_score_token_ids(self, yes, no, vocab_size=None):
        unk = getattr(self._hf_tokenizer, "unk_token_id", None)
        if (
            any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i == unk for i in (yes, no))
            or yes == no
            or (vocab_size is not None and any(i >= vocab_size for i in (yes, no)))
        ):
            raise RuntimeError("Reranker requires distinct valid yes/no token IDs")

    def _final_logits(self, input_ids, lengths):
        # Qwen3 .model returns normalized hidden states, not logits. Gather
        # before the vocabulary head: B x H -> B x V, never B x L x V.
        core = getattr(self.model, "model", None)
        tied = getattr(getattr(self.model, "args", None), "tie_word_embeddings", None)
        if not callable(core) or tied not in (True, False):
            raise RuntimeError("Unsupported MLX reranker: final-state projection API required")
        hidden = core(input_ids)
        final = hidden[_mx_array(np.arange(len(lengths))), _mx_array(np.array(lengths) - 1), :]
        if tied:
            head = getattr(getattr(core, "embed_tokens", None), "as_linear", None)
        else:
            head = getattr(self.model, "lm_head", None)
        if not callable(head):
            raise RuntimeError("Unsupported MLX reranker: missing vocabulary head")
        return head(final)

    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> EmbeddingResult:
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        self.validate_inputs(texts)
        vectors = np.zeros((len(texts), self.config.get("hidden_size", 1024)), dtype=np.float32)
        return EmbeddingResult(vectors=vectors, processing_time=0.0, device=self.device, model_info=self.model_name)

    async def compute_similarity(self, query_embedding: np.ndarray, passage_embeddings: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Cross-encoder backend does not support vector similarity")

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "rerank_method": "cross-encoder",
            "rerank_model_name": self.model_name,
            "backend": "mlx",
            "batch_size": self._batch_size,
            "configured_max_sequence_length": self._pair_max_len,
            "effective_max_sequence_length": self._effective_pair_limit(),
            "configured_max_passage_tokens": self._passage_max_tokens,
            # Query-dependent remaining space is applied separately per request.
            "effective_max_passage_tokens": min(self._passage_max_tokens, self._effective_pair_limit()),
            "is_loaded": self._is_loaded,
            "load_time": self._load_time,
        }

    def get_device_info(self) -> Dict[str, Any]:
        info = {
            "device": self.device or "mlx",
            "mlx_available": MLX_AVAILABLE,
        }
        if MLX_AVAILABLE and mx is not None:
            info["mlx_version"] = getattr(mx, "__version__", "unknown")
        return info

    async def rerank_passages(self, query: str, passages: List[str]) -> List[float]:
        if not self._is_loaded:
            raise RuntimeError("MLX reranker not loaded")
        if not passages:
            return []
        start_time = time.time()
        logger.info(f"MLX reranking query with {len(passages)} passages")

        try:
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(self._executor, self._rerank_sync, query, passages)
            validate_rerank_scores(scores, len(passages))

            elapsed = time.time() - start_time
            logger.info(f"MLX reranking completed with {len(scores)} scores in {elapsed:.3f}s")
            return scores
        except RerankQueryBudgetError:
            raise
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("MLX reranking failed", elapsed_seconds=elapsed, error_type=type(e).__name__)
            raise RuntimeError("MLX reranker inference failed; no fallback scores returned") from e

    def _rerank_sync(self, query: str, passages: List[str]) -> List[float]:
        all_scores: List[float] = []
        for i in range(0, len(passages), self._batch_size):
            batch_passages = passages[i : i + self._batch_size]
            batch_ids = [self._build_pair_ids(query, doc) for doc in batch_passages]

            max_len = max(len(ids) for ids in batch_ids)
            pad_id = self._hf_tokenizer.pad_token_id or 0
            padded = []
            for ids in batch_ids:
                pad_len = max_len - len(ids)
                # Causal attention cannot see future right padding. Unmasked
                # left padding changes real-token positions and attention.
                padded.append(ids + [pad_id] * pad_len)

            input_ids_np = np.array(padded, dtype=np.int32)
            input_ids = _mx_array(input_ids_np)

            last_logits_mx = self._final_logits(input_ids, [len(ids) for ids in batch_ids])
            yes_logits = last_logits_mx[:, self._token_yes_id]
            no_logits = last_logits_mx[:, self._token_no_id]

            yes_np = np.array(yes_logits.astype(mx.float32))
            no_np = np.array(no_logits.astype(mx.float32))

            del last_logits_mx, yes_logits, no_logits, input_ids

            for j in range(len(batch_passages)):
                yes_logit = float(yes_np[j])
                no_logit = float(no_np[j])
                if not np.isfinite(yes_logit) or not np.isfinite(no_logit):
                    raise RuntimeError("MLX reranker produced nonfinite yes/no logits")
                max_logit = max(yes_logit, no_logit)
                yes_exp = np.exp(yes_logit - max_logit)
                no_exp = np.exp(no_logit - max_logit)
                score = float(yes_exp / (yes_exp + no_exp + 1e-12))
                all_scores.append(score)

            del yes_np, no_np

        mx.clear_cache()
        return all_scores

    def __del__(self):
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
