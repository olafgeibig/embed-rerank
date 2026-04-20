"""
Apple MLX Backend: Embedding generation via mlx-lm transformer inference.

Uses mlx-lm to run full transformer forward passes through all layers,
producing high-quality embeddings with last-token pooling (the standard
pooling strategy for Qwen3-Embedding and similar instruction-tuned models).
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
    logger.info("MLX modules successfully imported - Apple Silicon detected")
except ImportError as e:
    MLX_AVAILABLE = False
    logger.warning("MLX not available - Apple Silicon required", error=str(e))
    mx = None  # type: ignore
    mlx_lm = None  # type: ignore


def _mx_array(x):
    """Create an MLX array in a version-compatible way."""
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


class MLXBackend(BaseBackend):
    """
    Apple MLX Backend using mlx-lm for full transformer inference.

    Loads the model via mlx-lm (which handles quantized weights, Metal
    acceleration, and Apple Silicon optimization) and generates embeddings
    by running the complete transformer forward pass with last-token pooling.
    """

    def __init__(self, model_name: str = "mlx-community/Qwen3-Embedding-4B-4bit-DWQ", model_path: Optional[str] = None):
        if not MLX_AVAILABLE:
            raise RuntimeError(
                "MLX Framework Required!\n"
                "MLX requires Apple Silicon (M1/M2/M3/M4) and macOS.\n"
                "Install with: pip install mlx>=0.4.0 mlx-lm>=0.2.0"
            )

        super().__init__(model_name, "mlx")
        self.model_path = model_path
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="MLX-Worker")
        self.model = None
        self.tokenizer = None
        self.config = None
        self._hf_tokenizer = None

        logger.info(
            "Initializing MLX Backend",
            model_name=model_name,
            model_path=model_path,
            device="apple_silicon",
        )

    async def load_model(self) -> None:
        if self._is_loaded:
            logger.info("Model already loaded", model_name=self.model_name)
            return

        logger.info("Loading MLX model via mlx-lm", model_name=self.model_name)
        start_time = time.time()

        try:
            loop = asyncio.get_event_loop()
            self.model, self.tokenizer, self.config, self._hf_tokenizer = await loop.run_in_executor(
                self._executor, self._load_model_sync
            )

            self._load_time = time.time() - start_time
            self._is_loaded = True

            logger.info(
                "MLX model loaded successfully",
                model_name=self.model_name,
                load_time=self._load_time,
                device="apple_silicon_mlx",
            )

        except Exception as e:
            logger.error("Failed to load MLX model", model_name=self.model_name, error=str(e))
            raise RuntimeError(f"MLX model loading failed for {self.model_name}: {e}")

    def _load_model_sync(self):
        import json
        import os

        model_id = self.model_name
        if self.model_path:
            path_str = str(self.model_path)
            if os.path.exists(path_str) and os.path.exists(os.path.join(path_str, "config.json")):
                model_id = path_str

        model, mlx_tokenizer = mlx_lm.load(model_id)
        logger.info("mlx-lm model loaded", model_type=type(model).__name__)

        config = {}
        if hasattr(model, "args") and hasattr(model.args, "to_dict"):
            config = model.args.to_dict()
        elif hasattr(model, "args") and isinstance(model.args, dict):
            config = model.args
        else:
            config_path = None
            if self.model_path and os.path.exists(self.model_path):
                config_path = os.path.join(self.model_path, "config.json")
            else:
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
            config["hidden_size"] = getattr(model, "hidden_size", 2560)

        hf_tokenizer = None
        try:
            from transformers import AutoTokenizer

            for source in [self.model_path, self.model_name]:
                if source:
                    try:
                        hf_tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
                        break
                    except Exception:
                        continue
        except Exception:
            pass

        if hf_tokenizer is None:
            logger.warning("Could not load HF tokenizer, using mlx tokenizer for encoding")
            hf_tokenizer = _MLXTokenizerAdapter(mlx_tokenizer)

        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token

        return model, mlx_tokenizer, config, hf_tokenizer

    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> EmbeddingResult:
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.validate_inputs(texts)
        start_time = time.time()

        logger.info("Generating embeddings with MLX", num_texts=len(texts), batch_size=batch_size, device="mlx")

        try:
            loop = asyncio.get_event_loop()
            vectors = await loop.run_in_executor(self._executor, self._embed_sync, texts, batch_size)

            processing_time = time.time() - start_time

            logger.info(
                "MLX embeddings generated",
                num_texts=len(texts),
                embedding_dim=vectors.shape[1] if vectors.ndim > 1 else len(vectors),
                processing_time=processing_time,
                device="mlx",
            )

            return EmbeddingResult(
                vectors=vectors, processing_time=processing_time, device="mlx", model_info=self.model_name
            )

        except Exception as e:
            logger.error("MLX embedding generation failed", num_texts=len(texts), error=str(e))
            raise RuntimeError(f"MLX embedding failed: {e}")

    def _get_transformer_backbone(self):
        if hasattr(self.model, "language_model"):
            return self.model.language_model.model
        if hasattr(self.model, "model"):
            return self.model.model
        raise RuntimeError("Cannot locate transformer backbone in loaded model")

    def _embed_sync(self, texts: List[str], batch_size: int) -> np.ndarray:
        try:
            if not self.model or not self._hf_tokenizer:
                raise RuntimeError("Model or tokenizer not loaded")

            backbone = self._get_transformer_backbone()
            embeddings_list = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                enc = self._hf_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="np",
                )

                input_ids_np = enc["input_ids"].astype(np.int32)
                attention_mask_np = np.array(enc["attention_mask"])

                input_ids = _mx_array(input_ids_np)

                hidden = backbone(input_ids)
                hidden_np = np.array(hidden.astype(mx.float32))

                sequence_lengths = attention_mask_np.sum(axis=1) - 1
                batch_embeddings = np.array([hidden_np[j, seq_len] for j, seq_len in enumerate(sequence_lengths)])

                norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                batch_embeddings = batch_embeddings / (norms + 1e-8)

                embeddings_list.append(batch_embeddings)

            return np.vstack(embeddings_list)

        except Exception as e:
            logger.error("MLX sync embedding failed", error=str(e))
            raise

    async def compute_similarity(self, query_embedding: np.ndarray, passage_embeddings: np.ndarray) -> np.ndarray:
        try:
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            passage_norms = passage_embeddings / np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
            similarities = np.dot(passage_norms, query_norm)
            return similarities
        except Exception as e:
            logger.error("Similarity computation failed", error=str(e))
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            passage_norms = passage_embeddings / np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
            return np.dot(passage_norms, query_norm)

    def get_model_info(self) -> Dict[str, Any]:
        info = {
            "backend": "mlx",
            "model_name": self.model_name,
            "model_path": str(self.model_path) if self.model_path else None,
            "device": "mlx",
            "is_loaded": self._is_loaded,
            "load_time": self._load_time,
            "pooling": "last_token",
        }

        if self._is_loaded:
            try:
                info.update(
                    {
                        "mlx_device": "apple_silicon",
                        "hidden_size": (self.config.get("hidden_size") if isinstance(self.config, dict) else None),
                        "max_position_embeddings": (
                            self.config.get("max_position_embeddings") if isinstance(self.config, dict) else None
                        ),
                    }
                )
            except Exception as e:
                logger.warning("Could not get MLX model info", error=str(e))

        return info

    def get_device_info(self) -> Dict[str, Any]:
        info = {
            "backend": "mlx",
            "device": "mlx",
            "apple_silicon": True,
        }

        try:
            if MLX_AVAILABLE:
                info.update(
                    {
                        "mlx_version": getattr(mx, "__version__", "unknown"),
                        "unified_memory": True,
                        "metal_support": True,
                    }
                )
            else:
                info["mlx_available"] = False
        except Exception as e:
            logger.warning("Could not get MLX device info", error=str(e))

        return info

    async def rerank_passages(self, query: str, passages: List[str]) -> List[float]:
        start_time = time.time()
        logger.info(f"MLX reranking query with {len(passages)} passages")

        try:
            query_result = await self.embed_texts([query])
            passages_result = await self.embed_texts(passages)

            query_vector = query_result.vectors[0]
            passage_vectors = passages_result.vectors

            scores = await self.compute_similarity(query_vector, passage_vectors)

            scores_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)

            processing_time = time.time() - start_time
            logger.info(f"MLX reranking completed with {len(scores_list)} scores in {processing_time:.3f}s")
            return scores_list

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"MLX reranking failed after {processing_time:.3f}s: {str(e)}")
            return await self._fallback_rerank(query, passages)

    async def _fallback_rerank(self, query: str, passages: List[str]) -> List[float]:
        logger.warning("Using fallback reranking method")
        scores = []

        query_words = set(query.lower().split())

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


class _MLXTokenizerAdapter:
    """Adapts an mlx tokenizer to expose the HF tokenizer __call__ interface."""

    def __init__(self, mlx_tokenizer):
        self._tok = mlx_tokenizer
        self.pad_token = ""
        self.eos_token = ""

    def __call__(self, texts, padding=True, truncation=True, max_length=512, return_tensors="np"):
        all_ids = []
        for t in texts if isinstance(texts, list) else [texts]:
            ids = self._tok.encode(t)
            if truncation:
                ids = ids[:max_length]
            all_ids.append(ids)

        max_len = max(len(ids) for ids in all_ids)
        padded = []
        masks = []
        for ids in all_ids:
            pad_len = max_len - len(ids)
            padded.append(ids + [0] * pad_len)
            masks.append([1] * len(ids) + [0] * pad_len)

        import numpy as _np

        return {
            "input_ids": _np.array(padded, dtype=_np.int64),
            "attention_mask": _np.array(masks, dtype=_np.int64),
        }
