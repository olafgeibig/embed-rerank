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

3. ~~**Remove the `MLXCrossEncoderBackend`** (`mlx_reranker_backend.py`) -- it has the same fundamental problem (token embedding lookup + random linear head instead of real inference). If cross-encoder reranking is needed on macOS, the Torch `CrossEncoder` backend works correctly via MPS.~~ **Fixed** -- see below.

---

# Fix: MLX Cross-Encoder Reranker Backend Produces Incorrect Scores

## Problem

The dedicated MLX reranker backend (`MLXCrossEncoderBackend` in `app/backends/mlx_reranker_backend.py`) had the same fundamental flaw as the embedding backend: it never ran real transformer inference. It loaded raw token embedding weights and applied a random linear head to produce scores.

This is the follow-up fix recommended in item 3 above.

## Root Cause

The old `MLXCrossEncoderBackend._load_sync()` loaded weights manually and extracted only `model.embed_tokens.weight`. The scoring path in `_rerank_sync()` then:

1. Looked up token embeddings via `embed_weight[input_ids]` (first layer only)
2. Mean-pooled the raw token vectors
3. Applied a deterministic random linear head (`cls_head`) to produce scores

This bypassed all 28 transformer layers entirely. The "classification head" was not the model's actual head -- it was either loaded from a `cls_head.npz` file (which doesn't exist in standard model repos) or generated deterministically from a hash of the model name using `np.random`.

### What the Old Code Did

```python
# Old _pooled_embeddings (simplified)
embed_weight = weights['model.embed_tokens.weight']
emb = embed_weight[input_ids]          # just a lookup table!
pooled = mx.mean(emb, axis=1)          # mean pool raw tokens

# Old scoring
w, b = self._cls_head                  # random linear head
scores = pooled @ w + b                # meaningless projection
```

### What the Correct Code Does

```python
# New _rerank_sync (simplified)
logits = self.model(input_ids)         # full forward pass: all 28 layers + LM head
last_logits = logits[:, -1, :]         # logits at last token position
yes_logit = last_logits[:, token_yes_id]
no_logit = last_logits[:, token_no_id]
score = sigmoid(yes_logit - no_logit)  # relevance probability
```

## The Fix

Rewrote `MLXCrossEncoderBackend` to use `mlx-lm` for full model loading and proper Qwen3-Reranker cross-encoder scoring.

### 1. Model Loading via mlx-lm

```python
# Old: manual weight loading + custom wrapper
weights = mx.load(weights_path)
embed_weight = weights['model.embed_tokens.weight']
head = self._load_linear_head(...)  # random fallback head

# New: full model via mlx-lm
model, tokenizer = mlx_lm.load(model_name)  # all 28 layers + LM head
```

### 2. Qwen3-Reranker Scoring Protocol

Qwen3-Reranker is a language model fine-tuned to answer "yes" or "no" when asked whether a document is relevant to a query. The scoring works by:

1. Formatting (query, document) pairs with the chat template:
   - System: "Judge whether the Document meets the requirements..."
   - User: `<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}`
   - Suffix: `<think\>\n\n</think\>\n\n` (triggers the model to output its answer)

2. Running the full forward pass through the model to get logits at the last position

3. Extracting logits for the "yes" token (ID 9693) and "no" token (ID 2152)

4. Computing the relevance score as: `exp(yes_logit) / (exp(yes_logit) + exp(no_logit))`

This matches the official Qwen3-Reranker inference code from the model card.

### 3. Proper Token Padding

Left-padding for batch processing, matching the reference implementation. Padding tokens are added to the left so that the last token (where we extract logits) is always the suffix-ending token.

### 4. Prefix/Suffix Token Assembly

The input for each (query, doc) pair is assembled as:
```
[prefix_tokens] + [pair_tokens] + [suffix_tokens]
```
Where prefix encodes the system + user role tokens, and suffix encodes the assistant role start + think tags. The pair tokens are truncated to fit within `max_length - len(prefix) - len(suffix)`.

## Results

Query: `"Organic skincare products for sensitive skin"`

| Rank | Index | Score | Document |
|------|-------|-------|----------|
| 1 | 6 | 0.981 | Sensitive skin-friendly facial cleansers and toners |
| 2 | 3 | 0.958 | Natural organic skincare range for sensitive skin |
| 3 | 2 | 0.042 | Organic cotton baby clothes for sensitive skin |
| 4 | 7 | 0.001 | Organic food wraps and storage solutions |
| 5 | 9 | 0.001 | Yoga mats made from recycled materials |
| 6 | 1 | 0.001 | Biodegradable cleaning supplies for eco-conscious consumers |
| 7 | 0 | 0.001 | Eco-friendly kitchenware for modern homes |
| 8 | 8 | 0.000 | All-natural pet food for dogs with allergies |
| 9 | 5 | 0.000 | Sustainable gardening tools and compost solutions |
| 10 | 4 | 0.000 | Tech gadgets for smart homes: 2024 edition |

The top 3 results are `[6, 3, 2]` -- all semantically correct. Irrelevant documents score near zero.

## Files Changed

- `app/backends/mlx_reranker_backend.py` -- Complete rewrite: model loading via mlx-lm, full transformer forward pass, Qwen3-Reranker yes/no token scoring protocol.

## Lessons Learned

### 1. Placeholder code is invisible until it breaks

Both the embedding backend and the reranker backend shipped with placeholder inference code. The original commits explicitly called this out, but over time the comments were buried and the code looked "finished." A CI regression test comparing embeddings against a known-good baseline would have caught this immediately.

### 2. The same bug pattern can repeat across files

The embedding backend and the reranker backend had **identical** root causes (token embedding lookup instead of real inference), implemented independently. When fixing a fundamental issue, always audit sibling components for the same pattern.

### 3. Model architecture determines the scoring protocol

For embedding models (Qwen3-Embedding): run transformer, extract last-token hidden state, L2-normalize, compute cosine similarity.

For reranker models (Qwen3-Reranker): run the full model including the LM head, extract "yes"/"no" logits at the last position, compute softmax probability. These are fundamentally different approaches -- the reranker is a generative model that predicts token probabilities, not a representation model.

### 4. The LM head matters for rerankers

The embedding backend only needs the transformer backbone (hidden states). But the reranker needs the full model including the vocabulary projection layer (`lm_head`), because the output logits are vocabulary-sized (151,669 dimensions) and we index into them by token ID. Using `self.model(input_ids)` (the full model) instead of `self.model.model(input_ids)` (just the backbone) is critical.

### 5. Chat template tokens must be exact

The Qwen3-Reranker requires a specific prompt format with system/user/assistant role tokens and a `<think\>\n\n</think\>\n\n` suffix to trigger the answer. Getting even one token wrong in the prefix or suffix produces garbage scores. Always verify by comparing against the model card's reference implementation or by checking `tokenizer.apply_chat_template()` output character-by-character.

### 6. Beware of HTML entity encoding in source files

When writing special characters like `<` and `>` in Python source files via tools, they can get HTML-entity-encoded to `&lt;` and `&gt;`. The tokenizer would then encode the literal `&`, `l`, `t`, `;` characters instead of `<`, producing completely wrong token sequences. Always verify the raw bytes of the saved file.
