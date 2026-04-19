# Evaluation Results

Numeric results from the 2026-04-18 ablation run. All tables reproduce
directly from the CSVs in `evaluation/results/`.

- Dataset: 12 reviewed QA pairs across 2 fact-dense videos.
- Retrieval: full 48-config grid (576 rows).
- Generation + faithfulness: 6-config subset (72 + 72 rows).

For methodology and caveats see `01_process.md`; for interpretation see
`03_observations.md`.

## Retrieval — 48-config ablation

### Overall means (all 48 configs × 12 QA)

| Metric        | Value |
|---------------|-------|
| Precision@1   | 0.490 |
| Precision@3   | 0.214 |
| Precision@5   | 0.164 |
| Recall@5      | 0.681 |
| Hit Rate@5    | 0.681 |
| MRR           | 0.576 |

Note: each QA pair has exactly one gold chunk, so the ceiling for P@5 is
`1/5 = 0.20`. Recall@5 and Hit Rate@5 are the more meaningful "coverage"
numbers for this dataset.

### By chunking strategy (most impactful axis)

| Chunking       | P@1   | P@3   | P@5   | Recall@5 | Hit@5 | MRR   |
|----------------|-------|-------|-------|----------|-------|-------|
| fixed-200      | 0.000 | 0.023 | 0.044 | 0.222    | 0.222 | 0.123 |
| fixed-1000     | 0.167 | 0.167 | 0.167 | 0.500    | 0.500 | 0.299 |
| **fixed-500**      | **0.917** | **0.333** | **0.233** | **1.000**    | **1.000** | **0.954** |
| **sentence-500**   | **0.875** | **0.333** | **0.213** | **1.000**    | **1.000** | **0.928** |

### By embedding model

| Embedding | P@1   | P@5   | Recall@5 | MRR   |
|-----------|-------|-------|----------|-------|
| minilm    | 0.469 | 0.166 | 0.688    | 0.566 |
| mpnet     | 0.500 | 0.166 | 0.688    | 0.579 |
| e5        | 0.500 | 0.161 | 0.667    | 0.582 |

### Reranker on vs. off

| Reranker  | P@1   | P@5   | Recall@5 | MRR   |
|-----------|-------|-------|----------|-------|
| off       | 0.438 | 0.167 | 0.694    | 0.536 |
| on        | 0.542 | 0.161 | 0.667    | 0.616 |

### Top 5 configurations (ranked by P@5)

| emb     | chunk     | rerank | P@1   | P@5   | Recall@5 | MRR   |
|---------|-----------|--------|-------|-------|----------|-------|
| minilm  | fixed-500 | off    | 0.833 | 0.233 | 1.000    | 0.917 |
| minilm  | fixed-500 | on     | 1.000 | 0.233 | 1.000    | 1.000 |
| mpnet   | fixed-500 | on     | 1.000 | 0.233 | 1.000    | 1.000 |
| e5      | fixed-500 | off    | 0.833 | 0.233 | 1.000    | 0.917 |
| e5      | fixed-500 | on     | 1.000 | 0.233 | 1.000    | 1.000 |

### By video / domain

| domain         | video         | P@1   | P@5   | Recall@5 | MRR   |
|----------------|---------------|-------|-------|----------|-------|
| cs_lectures    | RRKwmeyIc24   | 0.444 | 0.162 | 0.583    | 0.511 |
| news_analysis  | VSFuqMh4hus   | 0.505 | 0.165 | 0.713    | 0.597 |

Source CSV: `evaluation/results/full_retrieval_retrieval.csv` (576 rows).

## Generation — 6-config subset

### By config

| cfg_name             | BLEU  | ROUGE-1 | ROUGE-2 | ROUGE-L | BERT-P | BERT-R | BERT-F1 |
|----------------------|-------|---------|---------|---------|--------|--------|---------|
| no-rerank-mistral    | 0.240 | 0.498   | 0.379   | 0.441   | 0.177  | 0.672  | 0.382   |
| sentence-mistral     | 0.268 | 0.540   | 0.414   | 0.469   | 0.240  | 0.667  | 0.424   |
| **top-e5-mistral**       | **0.285** | **0.554**   | **0.431**   | **0.493**   | 0.229  | **0.724**  | **0.437**   |
| top-minilm-llama2    | 0.160 | 0.375   | 0.255   | 0.317   | 0.200  | 0.547  | 0.350   |
| top-minilm-mistral   | 0.261 | 0.523   | 0.400   | 0.460   | 0.192  | 0.679  | 0.396   |
| top-mpnet-mistral    | 0.261 | 0.537   | 0.404   | 0.465   | 0.227  | 0.703  | 0.429   |

### By domain

| Domain         | BLEU  | ROUGE-L | BERT-F1 |
|----------------|-------|---------|---------|
| cs_lectures    | 0.250 | 0.467   | 0.379   |
| news_analysis  | 0.245 | 0.432   | 0.411   |

Source CSV: `evaluation/results/subset_generation.csv` (72 rows).

## Faithfulness — atomic fact precision

"Num facts" is the count of atomic claims extracted from the generated
answer; "supported" is the count whose non-stopword tokens overlap ≥50%
with the retrieved context; "fact precision" = supported / num_facts.

### By config

| cfg_name             | num_facts | supported | fact_precision | unsupported_rate |
|----------------------|-----------|-----------|----------------|------------------|
| no-rerank-mistral    | 3.58      | 2.67      | 0.668          | 33.2%            |
| sentence-mistral     | 3.17      | 2.25      | 0.653          | 34.7%            |
| top-e5-mistral       | 3.33      | 2.25      | 0.661          | 33.9%            |
| **top-minilm-llama2**    | 4.50      | 3.50      | **0.824**          | **17.6%**            |
| top-minilm-mistral   | 4.25      | 2.75      | 0.610          | 39.0%            |
| top-mpnet-mistral    | 3.67      | 2.42      | 0.617          | 38.3%            |

### By LLM

| LLM      | num_facts | supported | fact_precision |
|----------|-----------|-----------|----------------|
| mistral  | 3.60      | 2.47      | 0.642          |
| llama2   | 4.50      | 3.50      | 0.824          |

Source CSV: `evaluation/results/subset_faithfulness.csv` (72 rows).

## Plots

All under `evaluation/results/`:

- `plot_precision_by_chunking.png`
- `plot_mrr_by_embedding.png`
- `plot_rerank_effect.png`
- `plot_bertscore_by_config.png`
- `plot_faithfulness_by_llm.png`

## Best overall configuration

Two "winners" depending on what you optimise for.

**Fluency / reference match:**
`e5 + fixed-500 + rerank + mistral` —
BLEU 0.285, ROUGE-L 0.493, BERT-F1 0.437; retrieval MRR 1.0.

**Grounding / low hallucination:**
`minilm + fixed-500 + rerank + llama2` —
fact precision 0.824 (vs ~0.62 for Mistral equivalents), at the cost of
lower text-overlap scores.

## Driver script (generation subset)

This is the exact script used to produce `subset_generation.csv` and
`subset_faithfulness.csv`. Keep for reproducibility.

```python
import time, pandas as pd
from config import (
    PipelineConfig, ChunkingConfig, RetrievalConfig,
    GenerationConfig, RESULTS_DIR,
)
from src.pipeline import build_query_pipeline
from evaluation.retrieval_eval import load_qa_pairs
from evaluation.generation_eval import evaluate_generation
from evaluation.faithfulness_eval import compute_fact_precision

CONFIGS = {
    'top-minilm-mistral': PipelineConfig(
        embedding_model='minilm',
        chunking=ChunkingConfig(strategy='fixed', chunk_size=500),
        retrieval=RetrievalConfig(use_reranker=True),
        generation=GenerationConfig(model_name='mistral')),
    'top-minilm-llama2':  PipelineConfig(
        embedding_model='minilm',
        chunking=ChunkingConfig(strategy='fixed', chunk_size=500),
        retrieval=RetrievalConfig(use_reranker=True),
        generation=GenerationConfig(model_name='llama2')),
    'top-mpnet-mistral':  PipelineConfig(
        embedding_model='mpnet',
        chunking=ChunkingConfig(strategy='fixed', chunk_size=500),
        retrieval=RetrievalConfig(use_reranker=True),
        generation=GenerationConfig(model_name='mistral')),
    'top-e5-mistral':     PipelineConfig(
        embedding_model='e5',
        chunking=ChunkingConfig(strategy='fixed', chunk_size=500),
        retrieval=RetrievalConfig(use_reranker=True),
        generation=GenerationConfig(model_name='mistral')),
    'sentence-mistral':   PipelineConfig(
        embedding_model='minilm',
        chunking=ChunkingConfig(strategy='sentence', chunk_size=500),
        retrieval=RetrievalConfig(use_reranker=True),
        generation=GenerationConfig(model_name='mistral')),
    'no-rerank-mistral':  PipelineConfig(
        embedding_model='minilm',
        chunking=ChunkingConfig(strategy='fixed', chunk_size=500),
        retrieval=RetrievalConfig(use_reranker=False),
        generation=GenerationConfig(model_name='mistral')),
}

qa_pairs = load_qa_pairs()
all_gen, all_faith = [], []

for cfg_name, cfg in CONFIGS.items():
    for video_id in ['VSFuqMh4hus', 'RRKwmeyIc24']:
        video_qa = [q for q in qa_pairs if q.get('video_id') == video_id]
        if not video_qa:
            continue

        qp = build_query_pipeline(video_id, cfg, skip_llm_health_check=True)

        generated, contexts = [], []
        for qa in video_qa:
            qp.reset()
            chunks = qp.retrieve_only(qa['question'])
            contexts.append([c['text'] for c in chunks])
            generated.append(qp.ask(qa['question'])['answer'])

        gen_df = evaluate_generation(cfg, video_qa, generated)
        gen_df['cfg_name'] = cfg_name
        all_gen.append(gen_df)

        for qa, ans, ctx in zip(video_qa, generated, contexts):
            fp = compute_fact_precision(ans, ' '.join(ctx))
            all_faith.append({
                'cfg_name': cfg_name,
                'video_id': video_id,
                'domain': qa.get('domain'),
                'question': qa['question'],
                'num_facts': fp['num_facts'],
                'supported_facts': fp['supported_facts'],
                'fact_precision': fp['fact_precision'],
                'llm': cfg.generation.model_name,
            })

pd.concat(all_gen, ignore_index=True).to_csv(
    RESULTS_DIR / 'subset_generation.csv', index=False)
pd.DataFrame(all_faith).to_csv(
    RESULTS_DIR / 'subset_faithfulness.csv', index=False)
```
