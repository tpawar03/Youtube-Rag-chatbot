# Observations and Interpretation

Interpretation of the numbers reported in `02_results.md`, organised around
the four research questions from the project README, plus notes on
limitations and where to push next.

## RQ1 — Pipeline accuracy across domains

| Domain         | P@5   | Recall@5 | MRR   | BERT-F1 |
|----------------|-------|----------|-------|---------|
| news_analysis  | 0.165 | 0.713    | 0.597 | 0.411   |
| cs_lectures    | 0.162 | 0.583    | 0.511 | 0.379   |

Both domains are fact-dense. News edges CS lectures on every metric, most
likely because:

- The news video's 5 chunks are individually shorter and topically distinct
  ("vector database", "RAG", "MCP", "MoE", "AGI/ASI"), so a question about
  any single topic has only one strong attractor chunk.
- The CS-lectures video's 3 chunks overlap more heavily on recurring terms
  ("AI system", "orchestration", "model"), which gives the retriever more
  distractors for the same-sized gold set.

**Cross-domain coverage is incomplete for RQ1.** Both videos are fact-dense;
the non-fact-dense comparison (cooking, travel) never ran because those
videos could not be ingested after the YouTube IP-block. Treat the RQ1
numbers as a within-fact-dense slice, not the full domain sweep.

## RQ2 — Chunking strategy comparison (strongest signal in the study)

Chunking accounts for the single largest effect observed.

- **`fixed-200` is unusable.** P@1 = 0.00, MRR = 0.12. Chunks are too small
  to contain the full statement that answers a factual question — e.g. the
  fact "MoE was published in 1991" gets split across a boundary, and neither
  resulting chunk is a better retrieval hit than generic adjacent content.
- **`fixed-1000` dilutes.** P@1 = 0.17. A chunk that merges multiple facts
  forces the embedding to average across unrelated topics, losing the
  specificity a precise question needs.
- **`fixed-500` and `sentence-500` tie.** Both hit P@1 ≈ 0.9 and MRR ≈ 0.94.
  Around 500 tokens is the sweet spot for transcript content: big enough to
  contain a whole claim plus its connective context, small enough that the
  embedding still represents one idea.

Practical implication: when someone adds a new chunking strategy, the
benchmark to beat is `sentence-500` / `fixed-500` — anything meaningfully
worse than MRR 0.93 in this harness is a regression.

## RQ3 — Embedding model impact (near-zero on this corpus)

P@1 spread across MiniLM / MPNet / E5 is within 3 percentage points
(0.469 / 0.500 / 0.500). MRR spread is within 1.6 points. For a fact-dense
corpus this small, any decent sentence encoder saturates.

Reasons this is expected, not surprising:

1. **Short corpus.** Each video produces 3–5 chunks. With so few candidates,
   any reasonable similarity function separates the gold chunk from the rest.
2. **Distinct vocabulary.** Questions re-use content words from the transcript
   ("MCP", "AGI", "orchestration"). Even 384-dim MiniLM recovers the right
   chunk from token overlap alone.

To actually discriminate between embeddings, retest on videos with:

- More chunks per video (30+, not 3–5), so ranking matters.
- Paraphrased questions that avoid content-word overlap.
- Multi-gold QA pairs, to exercise Recall@K differences.

Until then, **pick MiniLM for compute**: same retrieval quality, ~2× faster
and half the index size of MPNet.

## RQ4 — Hallucination on fact-dense domains

This is the most actionable finding.

| LLM     | Fact precision | Unsupported-fact rate |
|---------|----------------|-----------------------|
| mistral | 0.642          | 35.8%                 |
| llama2  | 0.824          | 17.6%                 |

**Llama-2 hallucinates about half as often as Mistral** on the same retrieved
context. Qualitatively, Mistral is more willing to paraphrase or extrapolate
beyond the transcript (adding framing, drawing connections), while Llama-2
sticks more literally to what the chunks say.

There is a tension with RQ3-style metrics: Llama-2's conservatism also gives
it the **worst** BLEU / ROUGE / BERTScore numbers in the generation table,
because its answers are more verbose and less phrased-like the reference.
This is the classic faithfulness/fluency trade-off made visible:

- Rank by text-overlap: `top-e5-mistral` wins (BERT-F1 0.437, fact-P 0.661).
- Rank by grounding: `top-minilm-llama2` wins (fact-P 0.824, BERT-F1 0.350).

If the downstream use case allows any hallucination (e.g. casual Q&A), prefer
Mistral. If hallucinations are costly (facts, figures, citations), prefer
Llama-2 and accept the hit on fluency metrics.

## Secondary observation — reranker

| Reranker | P@1   | P@5   | Recall@5 | MRR   |
|----------|-------|-------|----------|-------|
| off      | 0.438 | 0.167 | 0.694    | 0.536 |
| on       | 0.542 | 0.161 | 0.667    | 0.616 |

+10pp on P@1 and +8pp on MRR; -3pp on Recall@5. This is the expected
precision/recall exchange: the cross-encoder re-scores candidates with
joint (query, chunk) attention, pushing the single best answer to rank 1,
but occasionally demoting a still-relevant secondary chunk out of the top-k.

For this dataset it is a pure win on the "first answer is right" metric.
For tasks where you need to show the user multiple supporting passages (think
multi-source citation), the recall cost matters more and the verdict flips.

## Best overall configuration — final picks

- **Balanced default:**
  `e5 + fixed-500 + rerank + mistral`.
  Best text-overlap metrics in the subset (BLEU 0.285, ROUGE-L 0.493,
  BERT-F1 0.437) and MRR = 1.0 on retrieval.
- **High-stakes / grounded default:**
  `minilm + fixed-500 + rerank + llama2`.
  Fact precision 0.824 vs ~0.62 for all Mistral configs. Swap MiniLM for E5
  for a small boost in retrieval quality with no faithfulness cost.

## Limitations (how far you can trust these numbers)

1. **n = 2 videos, 12 QA pairs.** Directional, not statistically powered.
   Differences under ~5 points on any metric are inside noise. Any finding
   rated "near-zero" (embedding impact) would need a larger corpus before
   being used to make a final decision.
2. **Both videos fact-dense.** RQ1 across 5 domains is unverified — the
   cooking / travel slice of the intended grid never ran.
3. **Fact-precision is a heuristic, not an NLI judge.** The 50%-token-overlap
   rule over-counts support when an LLM paraphrases correctly using synonyms,
   and under-counts when it quotes verbatim but adds one unsupported clause.
   RAGAS Faithfulness (LLM-judged) would give a stronger number but was
   skipped in this run for time.
4. **QA pairs generated by Mistral.** Five of six subset configs also use
   Mistral for answering. That self-consistency bias inflates Mistral
   numbers relative to Llama-2 — the actual Mistral-vs-Llama-2 gap may be
   larger than reported once a neutral question generator is used.
5. **Only `fixed-500` chunking was used for QA-pair generation.** Evaluating
   a `sentence-500` or `fixed-200` config against those QA pairs imports a
   slight structural bias — the `relevant_chunk_ids` labels are attached to
   a 500-token chunk layout, not the layout under test. The effect should
   be small for `sentence-500` (similar granularity) but nontrivial for
   `fixed-200`, contributing to its extreme-bad scores.

## Where to push next (highest-value work)

1. **Wait for the YouTube IP-block to clear, re-run against all 20 videos.**
   Uses the existing cache-first fetch; no further code changes needed. This
   alone upgrades every metric from "directional" to "presentable".
2. **Re-generate QA pairs with a neutral model** (e.g. a larger LLM or a
   hand-written seed per video) to remove the Mistral self-consistency bias.
3. **Add RAGAS Faithfulness** alongside the heuristic precision. Two
   complementary numbers are much more defensible than one.
4. **Evaluate retrieval layouts against layout-matched QA labels.** Either
   re-label `relevant_chunk_ids` under each chunking strategy, or score at
   the *time-range* level (does the retrieved chunk overlap the gold time
   span?) rather than the chunk-index level.
5. **Pick one of the two best-config families as the shipping default** and
   make it the `PipelineConfig()` no-argument default in `config.py`. That
   locks in the "what should production look like" answer this study
   produced.
