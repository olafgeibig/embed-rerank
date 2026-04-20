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
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.logger import setup_logging
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
        batch_size: int = 16,
        max_length: int = 512,
        pooling: str = "mean",
        score_norm: str = "none",
    ):
        if not MLX_AVAILABLE:
            raise RuntimeError(
                "MLX backend requested but MLX is not available.\n"
                "Install mlx and mlx-lm, and ensure Apple Silicon (arm64)."
            )

        super().__init__(model_name, device or "mlx")
        self._batch_size = batch_size
        self._pair_max_len = max_length
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
        if isinstance(token_yes_id, list):
            token_yes_id = token_yes_id[0]
        if isinstance(token_no_id, list):
            token_no_id = token_no_id[0]

        prefix = "<|im_start|>system\n" + _SYSTEM_MSG + "<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        prefix_tokens = hf_tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = hf_tokenizer.encode(suffix, add_special_tokens=False)

        return model, mlx_tokenizer, config, hf_tokenizer, token_yes_id, token_no_id, prefix_tokens, suffix_tokens

    def _format_pair(self, query: str, doc: str, instruction: str = _DEFAULT_INSTRUCTION) -> str:
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

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

            elapsed = time.time() - start_time
            logger.info(f"MLX reranking completed with {len(scores)} scores in {elapsed:.3f}s")
            return scores
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"MLX reranking failed after {elapsed:.3f}s: {str(e)}")
            return await self._fallback_rerank(query, passages)

    def _rerank_sync(self, query: str, passages: List[str]) -> List[float]:
        all_scores: List[float] = []
        for i in range(0, len(passages), self._batch_size):
            batch_passages = passages[i : i + self._batch_size]
            pair_texts = [self._format_pair(query, doc) for doc in batch_passages]

            batch_ids = []
            for pair_text in pair_texts:
                pair_ids = self._hf_tokenizer.encode(pair_text, add_special_tokens=False)
                pair_ids = pair_ids[: self._pair_max_len - len(self._prefix_tokens) - len(self._suffix_tokens)]
                full_ids = self._prefix_tokens + pair_ids + self._suffix_tokens
                batch_ids.append(full_ids)

            max_len = max(len(ids) for ids in batch_ids)
            pad_id = self._hf_tokenizer.pad_token_id or 0
            padded = []
            for ids in batch_ids:
                pad_len = max_len - len(ids)
                padded.append([pad_id] * pad_len + ids)

            input_ids_np = np.array(padded, dtype=np.int32)
            input_ids = _mx_array(input_ids_np)

            logits = self.model(input_ids)
            logits_np = np.array(logits.astype(mx.float32))

            last_logits = logits_np[:, -1, :]

            for row_logits in last_logits:
                yes_logit = float(row_logits[self._token_yes_id])
                no_logit = float(row_logits[self._token_no_id])

                max_logit = max(yes_logit, no_logit)
                yes_exp = np.exp(yes_logit - max_logit)
                no_exp = np.exp(no_logit - max_logit)
                score = float(yes_exp / (yes_exp + no_exp + 1e-12))
                all_scores.append(score)

        return all_scores

    async def _fallback_rerank(self, query: str, passages: List[str]) -> List[float]:
        logger.warning("Using fallback reranking method")
        query_words = set(query.lower().split())
        scores = []
        for passage in passages:
            passage_words = set(passage.lower().split())
            overlap = len(query_words.intersection(passage_words))
            total_words = len(query_words.union(passage_words))
            score = overlap / max(total_words, 1)
            scores.append(float(score))
        return scores

    def __del__(self):
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
