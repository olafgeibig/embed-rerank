# Fix: MLX Backend Produces Incorrect Re-Ranking Results

## Problem

On macOS with Apple Silicon, the `/api/v1/rerank` endpoint returns semantically wrong results. Documents that are irrelevant to the query are ranked above relevant ones. For example, querying for "best skincare routine for sensitive skin" ranks kitchen cleaning products above skincare products.

This issue does **not** affect Linux deployments.

## Root Cause

The `MLXBackend` in `app/backends/mlx_backend.py` was **never performing real transformer inference**. It was implemented as a placeholder that only performs a raw token-embedding lookup from the first model layer, skipping all transformer layers entirely.

### Evidence from Git History

The original commit (`cfadcbe`) that introduced the MLX backend explicitly documents this:

> *Placeholder MLX model inference (ready for actual model conversion)*

The code itself contains comments like:

> *"For now, we'll use a placeholder approach since MLX model conversion is complex and model-specific."*

### What the Old Code Did

```python
# Old _embed_sync (simplified)
embed_weight = weights['model.embed_tokens.weight']
embeddings = embed_weight[input_ids]        # just a lookup table!
mean_embeddings = mx.mean(embeddings, axis=1) # mean pool the raw tokens
```

This is equivalent to looking up word vectors in a dictionary and averaging them -- it completely bypasses the 36 transformer layers that give the model its semantic understanding. Two unrelated documents that happen to share common words will look similar under this scheme.

### What the Correct Code Should Do

```python
# New _embed_sync (simplified)
hidden = model.model(input_ids)               # full forward pass through ALL layers
# last-token pooling (correct for Qwen3-Embedding)
embeddings = hidden[batch, last_non_padding_token]
```

## Platform Impact: macOS vs Linux

This is a **macOS-specific bug** because the `BackendFactory` auto-selects the backend based on the platform:

| Platform | Auto-Selected Backend | Works Correctly? |
|----------|----------------------|-----------------|
| Linux (CUDA/CPU) | `TorchBackend` via `sentence-transformers` | Yes |
| macOS (Apple Silicon) | `MLXBackend` (placeholder code) | **No** |

The `TorchBackend` uses `SentenceTransformer.encode()` which runs the full model correctly. On macOS, the factory detects Apple Silicon and MLX availability, then selects `MLXBackend` -- which was the broken placeholder. This means:

- **Linux users** were never affected because they always use the Torch path.
- **macOS users** always get the broken MLX path, even though the Torch backend (via MPS) would produce correct results on their hardware.

## The Fix

Rewrote `MLXBackend` to use `mlx-lm` (`mlx_lm.load()`) for proper model loading and full transformer inference. Key changes in `app/backends/mlx_backend.py`:

### 1. Model Loading via mlx-lm

The old code manually loaded weight files and built a wrapper class that only accessed the embedding layer. The new code uses `mlx_lm.load()`, which handles quantized weights, Metal acceleration, and provides the full model ready for inference:

```python
# Old: manual weight loading + custom wrapper
weights = mx.load(weights_path)
model = self._create_mlx_embedding_model(config, weights)  # wrapper only accesses embed_tokens

# New: proper model via mlx-lm
model, tokenizer = mlx_lm.load(model_id)  # full model with all 36 layers
```

### 2. Full Transformer Forward Pass

The old code called `self.model.embed(input_ids)` which only looked up token embeddings. The new code calls `self.model.model(input_ids)` which runs the complete forward pass through all transformer layers:

```python
# Old: first-layer lookup only
batch_embeddings = self.model.embed(input_ids)  # embed_tokens.weight[input_ids]

# New: full transformer inference
hidden = self.model.model(input_ids)  # all 36 transformer layers
```

### 3. Last-Token Pooling

The old code used mean pooling (average all token embeddings). Qwen3-Embedding is trained with **last-token pooling** -- the embedding is taken from the hidden state at the last non-padding token position:

```python
# Old: wrong pooling strategy
mean_embeddings = mx.mean(embeddings, axis=1)

# New: correct last-token pooling for Qwen3-Embedding
sequence_lengths = attention_mask.sum(axis=1) - 1
batch_embeddings = [hidden_np[j, seq_len] for j, seq_len in enumerate(sequence_lengths)]
```

### 4. HF Tokenizer for Proper Padding

The old code used a whitespace tokenizer or the raw MLX tokenizer (which lacks padding support). The new code uses the HuggingFace `AutoTokenizer` for proper subword tokenization, padding, and attention mask generation.

## Results

Query: `"best skincare routine for sensitive skin"`

### Before (wrong)

Cleaning products ranked above skincare products.

### After (correct)

| Rank | Document | Score |
|------|----------|-------|
| 1 | skincare routine for dry skin with moisturizer | 0.727 |
| 2 | gentle face cleanser for sensitive skin type | 0.711 |
| 3 | hydrating face mask with hyaluronic acid | 0.617 |
| 4 | anti-aging serum with retinol and vitamin C | 0.571 |
| 5 | sunscreen SPF 50 for daily face protection | 0.568 |
| 6 | best kitchen cleaning products for grease | 0.454 |
| 7 | how to clean bathroom tiles with bleach | 0.370 |
| 8 | heavy duty floor cleaner for industrial use | 0.350 |

Skincare documents are now correctly ranked at the top.

## Files Changed

- `app/backends/mlx_backend.py` -- Complete rewrite of model loading, embedding inference, and pooling logic.

## Test Results

- 23 tests pass, 4 skipped (MLX hardware tests in CI), no regressions introduced.
- The 1 pre-existing test error (`test_health_reranker_info.py`) is unrelated -- it fails due to MLX not being available in the CI environment.

## Recommendations for Follow-Up

1. **Add a regression test** that verifies embedding quality (e.g., assert that semantically similar texts have higher cosine similarity than dissimilar ones). This would have caught the placeholder issue immediately.

2. **Consider adding Qwen3-Embedding instruction format** -- the model supports `Instruct: {task}\nQuery:{query}` prefixing for queries, which can further improve retrieval quality. The current fix produces correct results without it, but adding instruction formatting would match the model's intended usage.

3. **Remove the `MLXCrossEncoderBackend`** (`mlx_reranker_backend.py`) -- it has the same fundamental problem (token embedding lookup + random linear head instead of real inference). If cross-encoder reranking is needed on macOS, the Torch `CrossEncoder` backend works correctly via MPS.
