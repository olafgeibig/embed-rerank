"""Shared reranking API guards, independent of tokenizer/inference budgets."""
import math
from numbers import Real

from app.config import settings


class RerankQueryBudgetError(ValueError):
    """A valid query/template consumes the entire configured pair budget."""


def validate_rerank_scores(scores, expected):
    """Fail closed on backend protocol errors before ranking/aggregation/zip."""
    try:
        if len(scores) != expected or any(not isinstance(s, Real) or not math.isfinite(s) for s in scores):
            raise RuntimeError("Invalid reranker score vector: expected one finite score per passage")
    except (TypeError, ValueError) as e:
        raise RuntimeError("Invalid reranker score vector") from e
    return scores


def validate_rerank_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty or whitespace only")
    if len(query) > 2048:
        raise ValueError(f"Query too long: {len(query)} > 2048 characters")
    return query.strip()


def validate_rerank_passages(passages):
    if not passages:
        raise ValueError("passages cannot be empty")
    count_limit = min(1000, settings.max_passages_per_rerank)
    if len(passages) > count_limit:
        raise ValueError(f"Too many passages: {len(passages)} > {count_limit}")
    limit = settings.rerank_max_passage_chars
    for i, passage in enumerate(passages):
        if not isinstance(passage, str):
            raise ValueError(f"Passage at index {i} must be a string")
        if not passage.strip():
            raise ValueError(f"Passage at index {i} cannot be empty or whitespace only")
        if len(passage) > limit:
            raise ValueError(f"Passage at index {i} too long: {len(passage)} > {limit} characters")
    # Never strip/reorder documents: callers return original text and indices.
    return passages
