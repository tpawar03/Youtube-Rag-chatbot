# Observations and Interpretation

Interpretation of the numbers reported in `02_results.md`, organised around
the four research questions from the project README, plus notes on
limitations and where to push next. Reflects the 2026-04-21 re-run.

## RQ1 — Pipeline accuracy across domains

| Domain         | P@5   | Recall@5 | MRR   | BERT-F1 |
|----------------|-------|----------|-------|---------|
| news_analysis  | 0.165 | 0.713    | 0.597 | 0.271   |
| cs_lectures    | 0.162 | 0.583    | 0.511 | 0.357   |

Both domains are fact-dense. Retrieval numbers (P@5, Recall@5, MRR) are
deterministic and reproduce the previous run: news edges CS lectures on
every retrieval metric, most likely because:

- The news video's 5 chunks are individually shorter and topically distinct
  ("vector database", "RAG", "MCP", "MoE", "AGI/ASI"), so a question about
  any single topic has only one strong attractor chunk.
- The CS-lectures video's 3 chunks overlap more heavily on recurring terms
  ("AI system", "orchestration", "model"), which gives the retriever more
  distractors for the same-sized gold set.

**Generation rank flipped this run.** CS-lectures BERT-F1 is now higher
than news (0.357 vs. 0.271), reversing the 2026-04-18 result. Retrieval
is identical, so this is purely LLM-sampling variance on a very small
slice (9 QA for news, 3 QA for CS-lectures). With n=3 on the CS side,
a single unusually good Mistral completion is enough to move the group
mean by ~0.03–0.05 BERT-F1. **The cross-domain generation rank is not
stable** on this dataset size — do not quote it as a finding.

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

The headline from the 2026-04-18 run ("Llama-2 hallucinates half as often
as Mistral") **did not reproduce**. Fresh numbers from 2026-04-21:

| LLM     | Fact precision | Unsupported-fact rate |
|---------|----------------|-----------------------|
| mistral | 0.492          | 50.8%                 |
| llama2  | 0.407          | 59.3%                 |

| Run        | Mistral fact-P | Llama-2 fact-P | Delta |
|------------|----------------|----------------|-------|
| 2026-04-18 | 0.642          | 0.824          | +0.18 (Llama-2) |
| 2026-04-21 | 0.492          | 0.407          | −0.09 (Mistral) |

Same pipeline, same QA pairs, same config subset — only the LLM samples
differ (temperature 0.4, no seed). That the relative ranking flipped means
the signal is inside the noise floor of this harness. Best explanation:

1. **Metric is token-overlap, not semantic.** `check_fact_in_context` in
   `faithfulness_eval.py` marks a claim "supported" iff ≥50% of its
   non-stopword tokens appear in the concatenated context. Under stochastic
   sampling a single synonym substitution ("employs" for "uses", "across"
   for "throughout") can tip a sentence from supported to unsupported.
   Llama-2 paraphrases more aggressively than Mistral on the iPhone of this
   dataset, and that paraphrasing costs it precision on this particular
   metric — not on actual grounding.
2. **n is too small.** Each LLM has 12 QA pairs × 1 answer = 12 observations
   per run; the config-level cells have only 12 observations each. A ±0.1
   swing is well within the ±σ band you'd expect from re-sampling.
3. **Config-level noise dominates LLM-level noise.** On 2026-04-21 the
   *Mistral* configs spread from 0.321 (top-minilm-mistral) to 0.721
   (top-e5-mistral) — a 0.40 range. That intra-LLM spread is ~5× the
   inter-LLM gap, which is another way of saying "we can't see the LLM
   signal through the retrieval/config signal."

**Fluency-vs-grounding trade-off also flattened this run.** On
2026-04-21 `top-e5-mistral` is simultaneously the highest BERT-F1
(0.411) *and* the highest fact precision (0.721) — there is no
trade-off to observe in the 6-config subset.

**What's still trustworthy in RQ4:**

- Retrieval layout (embedding × chunking × rerank) has a **bigger effect
  on grounding than LLM choice does** on this corpus. `top-e5-mistral`
  at 0.721 vs. `top-minilm-mistral` at 0.321 is the same LLM, different
  retriever. Fix the retriever first.
- The fact-overlap metric is not fit-for-purpose for cross-LLM comparisons
  at this scale. Any future RQ4-style claim should wait for an NLI-based
  or LLM-judged faithfulness score.

**What's no longer trustworthy:** the "pick Llama-2 for grounded answers"
recommendation from the previous doc revision. Do not use it.

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

Re-run produced a single dominant config rather than the Pareto-pair from
the prior revision:

- **Shipping default:** `e5 + fixed-500 + rerank + mistral`.
  - Retrieval: MRR 1.00, Recall@5 1.00.
  - Fluency: BLEU 0.292, ROUGE-L 0.487, BERT-F1 0.411 (best in subset).
  - Grounding: fact precision 0.721 (best in subset, +0.10 over nearest
    Mistral config, +0.31 over the nearest Llama-2 config).
- **If you need a cheaper embedder,** fall back to `mpnet + fixed-500 +
  rerank + mistral` — retrieval is identical, generation degrades
  gracefully (BERT-F1 0.352, fact-P 0.618).

The "high-stakes / grounded default" slot previously occupied by
`minilm + fixed-500 + rerank + llama2` no longer has evidence backing it
(see RQ4 above). Revisit once a more robust faithfulness metric is in
place.

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
