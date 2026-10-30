# 🔥 Embeddings + Reranking on your Mac (MLX‑first)

<p align="center">
  <a href="docs/ENHANCED_OPENAI_API.md">
  <a href="https://github.com/joonsoo-me/embed-rerank/blob/main/LICENSE"><img src="https://img.shields.io/github/license/joonsoo-me/embed-rerank?logo=opensource&logoColor=white" /></a>
    <img src="https://img.shields.io/badge/OpenAI%20rerank-supported-2ea44f" alt="OpenAI rerank supported (/v1/openai/rerank)" />
  </a>
  <a href="docs/DEPLOYMENT_PROFILES.md">
    <img src="https://img.shields.io/badge/auto--sigmoid-default%20on-blue" alt="auto-sigmoid default on" />
  </a><a href="https://ml-explore.github.io/mlx/"><img src="https://img.shields.io/badge/MLX-Optimized-green?logo=apple&logoColor=white" /></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" /></a>
  <a href="https://pypi.org/project/embed-rerank/">
    <img src="https://img.shields.io/pypi/v/embed-rerank?logo=pypi&logoColor=white" alt="PyPI Version" />
  </a>
</p>

Blazing‑fast local embeddings and true cross‑encoder reranking on Apple Silicon. Works with Native, OpenAI, TEI, and Cohere APIs.

This page is a beginner‑friendly quick start. Detailed guides live in docs/.

## 🌐 Four APIs, One Service

| API | Endpoint | Use Case |
|-----|----------|----------|
| **Native** | `/api/v1/embed`, `/api/v1/rerank` | New projects |
| **OpenAI** | `/v1/embeddings`, `/v1/openai/rerank` (alias: `/v1/rerank_openai`) | Existing OpenAI code |
| **TEI** | `/embed`, `/rerank`, `/info` | Hugging Face TEI replacement |
| **Cohere** | `/v1/rerank`, `/v2/rerank` | Cohere API replacement |
| | `/docs` `/health` | More info. |

## 📈 Performance Visualization

### Latency Comparison (Projected)

```
Single Text Embedding Latency (milliseconds)

Apple MLX    ████ 0.2ms
PyTorch MPS  ████████████████████████████████████████████████ 45ms  
PyTorch CPU  ████████████████████████████████████████████████████████████████████████████████████████████████████████ 120ms
CUDA (Est.)  ████████████ 12ms
Vulkan (Est.) ████████████████████████ 25ms

0ms        25ms       50ms       75ms       100ms      125ms
```

### Throughput Comparison (texts/second)

```
Maximum Throughput (texts per second)

Apple MLX     ████████████████████████████████████████████████████████████████████████████████████████████████████████ 35,000
CUDA (Est.)   ████████████████████████████████ 8,000  
PyTorch MPS   ██████ 1,500
Vulkan (Est.) ████████████ 3,000
PyTorch CPU   ██ 500

0          10k        20k        30k        40k
```

## 🚀 Start here (60 seconds)

1) Install and run (embeddings only)

```bash
pip install embed-rerank

# Minimal .env
cat > .env <<'ENV'
BACKEND=auto
MODEL_NAME=mlx-community/Qwen3-Embedding-4B-4bit-DWQ
PORT=9000
HOST=0.0.0.0
ENV

embed-rerank  # http://localhost:9000
```

Want 2560‑D vectors by default? Add this to .env and restart:

```bash
cat >> .env <<'ENV'
# Use the model hidden_size (2560 for Qwen3-Embedding-4B) as output dimension
DIMENSION_STRATEGY=hidden_size
# Or enforce a fixed size (pads/truncates as needed):
# OUTPUT_EMBEDDING_DIMENSION=2560
# DIMENSION_STRATEGY=pad_or_truncate
ENV

# Verify
curl -s http://localhost:9000/api/v1/embed/ \
  -H 'Content-Type: application/json' \
  -d '{"texts":["hello"],"normalize":true}' | jq '.vectors[0] | length'
```

2) Try it (embeddings + simple rerank)

```bash
# Embeddings (Native)
curl -s http://localhost:9000/api/v1/embed/ \
  -H 'Content-Type: application/json' \
  -d '{"texts":["Hello MLX","Apple Silicon rocks"]}' | jq '.embeddings | length'

# Rerank fallback (no dedicated reranker yet)
curl -s http://localhost:9000/api/v1/rerank/ \
  -H 'Content-Type: application/json' \
  -d '{"query":"capital of france","documents":["Paris is the capital of France","Berlin is in Germany"],"top_n":2}' | jq '.results[0]'
```

3) Add a dedicated reranker (better quality)

```bash
cat >> .env <<'ENV'
# Torch cross-encoder (stable)
RERANKER_BACKEND=auto
RERANKER_MODEL_ID=cross-encoder/ms-marco-MiniLM-L-6-v2

# MLX reranker (Apple Silicon). Required for MLX models:
# RERANKER_BACKEND=mlx
# RERANKER_MODEL_NAME=vserifsaglam/Qwen3-Reranker-4B-4bit-MLX
ENV

# Optional: pre-download the MLX reranker to the default HF cache
# ./download-rerank-model.sh

# Restart server, then call Native or OpenAI-compatible rerank
curl -s http://localhost:9000/api/v1/rerank/ \
  -H 'Content-Type: application/json' \
  -d '{"query":"capital of france","documents":["Paris is the capital of France","Berlin is in Germany"],"top_n":2}' | jq '.results[0]'
```

4) (Optional) Run as a macOS service

```bash
# Uses your .env to generate a LaunchAgent and start the service
./tools/setup-macos-service.sh

# Check status and health
launchctl list | grep com.embed-rerank.server
open http://localhost:9000/health/
```

Notes
- OpenAI drop-in supported for both embeddings and rerank (/v1/embeddings, /v1/rerank). See docs for a tiny SDK example.
- Scores may be auto‑sigmoid‑normalized for OpenAI clients by default (disable via `OPENAI_RERANK_AUTO_SIGMOID=false`).
- The root endpoint `/` shows both `embedding_dimension` (served) and `hidden_size` (model config) for clarity.
- Reranker backend `auto` defaults to Torch for compatibility; MLX reranker models require `RERANKER_BACKEND=mlx`.

Run the full validation suite
```bash
./tools/server-tests.sh --full
```

## 🧭 Pick your path

- Deployment profiles (Embeddings‑only, Fallback rerank, Dedicated reranker): docs/DEPLOYMENT_PROFILES.md
- OpenAI usage (tiny example + options): docs/ENHANCED_OPENAI_API.md
- Quality benchmarks (JSONL/CSV judgments): docs/QUALITY_BENCHMARKS.md
- Troubleshooting: docs/TROUBLESHOOTING.md
- Backend specs and performance: docs/BACKEND_TECHNICAL_SPECS.md, docs/PERFORMANCE_COMPARISON_CHARTS.md

### Try it with OpenAI SDK (tiny)

```python
import openai

client = openai.OpenAI(base_url="http://localhost:9000/v1", api_key="dummy")

# Embeddings
res = client.embeddings.create(model="text-embedding-ada-002", input=["hello world"]) 
print(len(res.data[0].embedding))

# Rerank (OpenAI-compatible)
rr = client._request(
  "post",
  "/v1/openai/rerank",
  json={
    "query": "capital of france",
    "documents": [
      {"id": "a", "text": "Paris is the capital of France"},
      {"id": "b", "text": "Berlin is in Germany"},
    ],
    "top_n": 2,
  },
)
print(rr.get("results", rr))
```

## Tested Frameworks
| | Framework | Tests |
|---|---|---|
| ✅ | [**Open WebUI**](https://github.com/open-webui/open-webui) | `Embed` |
| ✅ | [**LightRAG**](https://github.com/HKUDS/LightRAG) | `Embed` `Rerank` |
| ✅ | [**continue.dev**](https://www.continue.dev) | `Embed` `Rerank` |
| ✅ | [**Kilo Code**](https://kilocode.ai/) | `Embed` |
| ✅ | [**LiteLLM**](https://www.litellm.ai/) | `Embed` `Rerank` |
| ✅ | [**RisuAI**](https://risuai.net/) | `Embed` |
###### We are waiting for your reports!

### LightRAG Option Parity
- `EMBEDDING_SEND_DIM`: gate honoring OpenAI-compatible `dimensions` in `/v1/embeddings`
- `EMBEDDING_TOKEN_LIMIT`: alias for `DEFAULT_MAX_TOKENS_OVERRIDE` (default per-text token budget)
- `RERANK_ENABLE_CHUNKING`, `RERANK_MAX_TOKENS_PER_DOC`: Cohere-compatible rerank document chunking
- These compatibility options retain their existing defaults when unset; embedding input context is configured separately below.

## Embedding input context

`MAX_SEQUENCE_LENGTH` must be a positive integer (`>=1`); it sets the embedding server's token cap and defaults to `8192`. The effective service cap is bounded by the server/backend limit and discovered model or tokenizer metadata; if metadata is unavailable, use the conservative `8192` cap. MLX uses the lower of the server setting and any known model/tokenizer limit. Torch does not expand a loaded model's native `max_seq_length`. Preprocessing uses the effective server budget by default. A request-level preprocessing override takes precedence over the legacy `EMBEDDING_TOKEN_LIMIT` alias for `DEFAULT_MAX_TOKENS_OVERRIDE`, but neither can exceed the server/model cap. The per-request schema remains capped at `8192`, even when `MAX_SEQUENCE_LENGTH` is configured higher.

Preprocessing estimates tokens as `len(text) // 4`; this is approximate. MLX uses its actual tokenizer for final truncation. Numeric warnings and `processing_info` are estimates, not exact backend token accounting, and warnings do not expose input text or token IDs. These limits do not guarantee unlimited or lossless long-text handling. For the safest long-input test across API routes, send one text per request. `BATCH_SIZE=1` alone does not guarantee per-request batching: set `batch_size: 1` in each OpenAI-compatible or native embedding request (the OpenAI-compatible route also accepts `X-MLX-Batch-Size: 1`). TEI requests should contain one text each. This embedding setting does not change `RERANK_MAX_SEQ_LEN`.

Changing the cap does not update embeddings already stored in an index or recover content truncated upstream. After a model switch, rebuild all vectors consistently in each affected index from the original full texts; do not mix vectors from different models. Restart only through the managed service owner. Do not start a second server. Treat restart and reindexing as separate actions requiring separate approval.

## Reranker input budgets (MLX Qwen3)

Operational defaults favor a 16 GiB host, not the model's maximum context:

| Setting | Default | Meaning |
| --- | --- | --- |
| `RERANK_MAX_SEQ_LEN` | `8192` | Total pair tokens, including instruction, query, document, chat wrappers and suffix |
| `RERANK_MAX_PASSAGE_TOKENS` | `6144` | Exact MLX document-token cap, further reduced by remaining pair space |
| `RERANK_MAX_PASSAGE_CHARS` | `65536` | API safety guard on each original document, including Cohere requests before chunking |
| `RERANK_BATCH_SIZE` | `1` | Conservative reranker batch size; larger batches and longer pairs increase memory use |

All four settings must be positive integers (`>=1`); environment and dotenv overrides are supported. Blank values use these defaults. The [official Qwen3-Reranker-0.6B model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) advertises **32k context** and uses `max_length=8192` in its reference example. Effective pair capacity is the minimum of the configured budget, known native model/tokenizer limits and the conservative official `32768` Qwen3-Reranker ceiling. A converted checkpoint reporting `max_position_embeddings=40960` is metadata, not proof of tested 40,960-token serving. Raising a limit requires separate memory/quality validation.

The exact MLX allowance is:

```text
overhead = tokenized instruction + query + document label + system/chat wrappers + scoring suffix
document allowance = min(RERANK_MAX_PASSAGE_TOKENS, effective pair cap - overhead)
```

The header and document are tokenized separately, then concatenated with the wrappers and suffix. Instruction/query/template tokens are preserved; only document token IDs are truncated. Truncation warnings contain numeric counts/limits, not query/document text. Queries leaving no document-token space fail with **HTTP 400**. API guards reject overlong or blank documents and blank/overlong queries (generic query limit: 2048 characters); character counts are not token estimates. Original response text and indices are preserved. Native, Cohere, OpenAI and TEI conversions use the same document guards, including the optional Cohere chunking path. Schema/conversion validation errors remain HTTP 400/422 depending on the route; inference failures or incomplete/nonfinite score vectors are server errors (500/503), never silent lexical/Jaccard scores.

Separate header/document encoding preserves the decoded formatted text but can produce different token IDs at their boundary than the official whole-pair tokenizer example. Exact reference-score parity is therefore not promised; validate relevance and batch invariance for the selected checkpoint rather than assuming identical upstream scores.

MLX batches use causal **right padding** and gather each pair's real final suffix state before the vocabulary head. This avoids projecting a full `batch × sequence × vocabulary` logit tensor; it does not eliminate transformer attention/activation memory costs. Backend model info exposes configured/effective sequence and passage caps; the passage allowance additionally depends on each query's actual overhead. Torch CrossEncoder retains its native model/tokenizer budgeting.

`RERANK_MAX_TOKENS_PER_DOC=4096` remains an **approximate Cohere chunking target** (`1 token ≈ 4 characters`), enabled only by `RERANK_ENABLE_CHUNKING=true`. It is not the exact MLX document cap, nor the API character guard. `MAX_SEQUENCE_LENGTH` independently controls embeddings. There is **no universal 8k chunk target**: choose retrieval chunks for document semantics, recall and latency, then fit them into the reranker's remaining pair allowance. Future OpenViking/client chunk-budget harmonization can use these distinct limits, but this change does not alter OpenViking, upstream chunking, existing indexes or stored embeddings.

## 📄 License

MIT License – build amazing things locally.
 
