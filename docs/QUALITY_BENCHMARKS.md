# Rerank Quality Benchmarks

Use these lightweight tools to compare reranker configurations (for example Torch vs MLX) on fixed inputs and relevance judgments.

## Quick check (tiny dataset)
```bash
./tools/server-tests.sh --quality-rerank-only
```

## Custom judgments
The script accepts JSONL or CSV via `--judgments`.

- JSONL: each line is an object with fields
```json
{"query":"...","passages":["...","..."],"judgments":[1,0,1]}
```
  - Aliases: `documents` for `passages`, `relevances` for `judgments`.

- CSV: columns `query`, `passages`, `judgments`
  - `passages` and `judgments` are JSON arrays.

Run:
```bash
python tools/tests/quality-rerank-only.py --host http://localhost:9000 --k 3 --judgments /path/to/data.jsonl
```

### Metrics
- MRR@k – reciprocal rank of the first relevant result
- nDCG@k – graded relevance quality of the top-k ranking

## Tips
- The MLX Qwen3 scorer already returns a yes/no probability. `RERANK_POOLING` and `RERANK_SCORE_NORM` do not change this scorer. OpenAI auto-sigmoid applies an additional sigmoid; disable it with `OPENAI_RERANK_AUTO_SIGMOID=false` to preserve raw probabilities.
- Keep queries, document pools, judgments and input budgets fixed when comparing models. Fixed-pool ranking quality does not measure whether retrieval found the right documents in the first place.
