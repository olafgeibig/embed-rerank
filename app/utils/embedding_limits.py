"""Helpers for applying model-specific embedding token limits."""


def resolve_embedding_limit(configured: int, *model_limits: object) -> int:
    """Return the configured limit bounded by usable model metadata limits.

    Non-positive, non-integer, boolean, and Hugging Face sentinel-sized
    metadata values are ignored. The configured value must be positive.
    """
    if configured < 1:
        raise ValueError("configured embedding limit must be at least 1")

    limits = [configured]
    limits.extend(
        limit
        for limit in model_limits
        if isinstance(limit, int) and not isinstance(limit, bool) and 0 < limit < 10**9
    )
    return min(limits)
