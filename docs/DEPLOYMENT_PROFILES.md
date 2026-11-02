# Deployment Profiles

This page details recommended ways to run embed-rerank depending on your goals.

## Profiles

### 1) Embeddings only (simple)
- No reranker model configured (CROSS_ENCODER_MODEL/RERANKER_MODEL_ID unset)
- Endpoints: `/api/v1/embed`, `/v1/embeddings`, `/embed`
- Rerank endpoint exists but falls back to embedding similarity.

Quick start:
```bash
uv sync --extra mlx
cp -n .env.example .env  # preserve an existing configuration
# Edit .env to choose BACKEND, MODEL_NAME, HOST and PORT.
uv run --extra mlx embed-rerank
```

### 2) Embeddings + Reranking (single embedding model)
- No dedicated reranker. Reranking uses embedding similarity.
- Endpoints: Native `/api/v1/rerank`, Cohere `/v1/rerank`, `/v2/rerank`.

### 3) Embeddings + Dedicated Reranker (two models)
- Configure a dedicated cross-encoder reranker.
- Torch example:
```env
RERANKER_BACKEND=torch
RERANKER_MODEL_ID=cross-encoder/ms-marco-MiniLM-L-6-v2
```
- MLX Qwen3 reranker example (Apple Silicon):
```env
RERANKER_BACKEND=mlx
RERANKER_MODEL_ID=vserifsaglam/Qwen3-Reranker-4B-4bit-MLX
# Optional pair/document budgets and conservative batch size
RERANK_MAX_SEQ_LEN=8192
RERANK_MAX_PASSAGE_TOKENS=6144
RERANK_BATCH_SIZE=1
```
- Endpoints: Native `/api/v1/rerank`, OpenAI-compatible `/v1/openai/rerank`, Cohere-compatible `/v1/rerank` and `/v2/rerank`.

The MLX Qwen3 scorer returns a yes/no probability; legacy `RERANK_POOLING` and `RERANK_SCORE_NORM` do not affect it. See the [README](../README.md#reranker-input-budgets-mlx-qwen3) for exact budget semantics.

## OpenAI-compatible /v1/openai/rerank behavior
- Prefers the dedicated reranker if configured; else falls back to embedding similarity.
- Scores are sigmoid-normalized by default, including MLX yes/no probabilities; disable with `OPENAI_RERANK_AUTO_SIGMOID=false` to preserve the original score.

## Native /api/v1/rerank OpenAI auto-sigmoid
- If the client looks like OpenAI (user-agent contains `openai`) or header `x-openai-compat: true`, scores are auto sigmoid-normalized by default.
- Disable globally with `OPENAI_RERANK_AUTO_SIGMOID=false`.

## Links
- OpenAI usage: `docs/ENHANCED_OPENAI_API.md`
- Troubleshooting: `docs/TROUBLESHOOTING.md`
- MLX/Backend specs: `docs/BACKEND_TECHNICAL_SPECS.md`
- Performance: `docs/PERFORMANCE_COMPARISON_CHARTS.md`