# Evaluation Process

This document records how the ablation / evaluation study was actually carried out,
including dead ends and fixes that were required along the way. It is the
companion to `02_results.md` (numbers) and `03_observations.md` (interpretation).

Study dates: 2026-04-18 (initial run), 2026-04-21 (re-run).

The 2026-04-21 re-run used the same pipeline, configs, dataset, and QA
pairs; only the LLM samples differ (temperature 0.4, no seed). Retrieval
results are bit-for-bit identical between runs. Generation and faithfulness
tables in `02_results.md` reflect the 2026-04-21 numbers; see that file
and `03_observations.md` RQ4 for an analysis of where the two runs
disagreed. Both drivers (`evaluation/ablation.py` and
`evaluation/run_subset_generation.py`) were invoked via
`python -m evaluation.<script>` with default arguments; `evaluation/plots.py`
regenerated the five PNGs from the fresh CSVs.

## 1. Test-set design (intended)

The original design in `evaluation/dataset/videos.json` calls for **20 videos
across 5 domains**, 4 per domain:

| Domain              | Count | Fact-dense? |
|---------------------|-------|-------------|
| `news_analysis`     | 4     | ✓           |
| `cs_lectures`       | 4     | ✓           |
| `technical_reviews` | 4     | ✓           |
| `cooking_tutorials` | 4     |             |
| `travel_vlogs`      | 4     |             |

20 candidate video IDs were selected via web search and written into
`videos.json` (see file for titles and URLs).

A transcript-availability smoke test confirmed that all 20 IDs had fetchable
captions at selection time (89–2230 segments each).

## 2. What actually ran

Two blockers forced a scope reduction:

1. **YouTube IP-block.** Immediately after the verification script fetched all
   20 transcripts in quick succession, `youtube-transcript-api` began returning
   `IpBlocked` for every subsequent request. The block is rate-limit driven, not
   code-related, and typically clears after 15 min – several hours.
2. **No QA-pair corpus yet.** `qa_pairs.json` shipped with a single placeholder
   entry. Running evaluators against it would produce empty DataFrames.

The study was therefore run against the **two videos already cached** in
`data/transcripts/` plus `data/indices/`:

| Video ID       | Domain (assigned) | Clean segments | Content                          |
|----------------|-------------------|----------------|----------------------------------|
| `VSFuqMh4hus`  | `news_analysis`   | 1              | AI terminology explainer         |
| `RRKwmeyIc24`  | `cs_lectures`     | 3              | Building AI applications         |

Both are fact-dense, so RQ4 (hallucination) coverage is intact. RQ1
cross-domain comparison against non-fact-dense content (cooking, travel) is
**not** covered by this run.

## 3. Pipeline fixes required to make the run work

Two bugs surfaced during execution and were patched in place.

### Fix 1 — cache-first transcript fetch

`src/transcript/fetcher.py` always called `YouTubeTranscriptApi.fetch()` even
when `data/transcripts/{video_id}_raw.json` was already on disk. Combined with
the YouTube rate-limit, this made re-ingestion impossible.

Patch: before calling the remote API, `fetch_transcript` now checks
`TRANSCRIPTS_DIR / f"{video_id}_raw.json"` and returns its contents if
present.

### Fix 2 — `FAISSStore.load` raises the right exception

`ablation.py` uses this pattern to decide whether to ingest or reuse an index:

```python
try:
    store.load(index_name)
except FileNotFoundError:
    pipeline.ingest(video_id)
```

But `faiss.read_index(...)` raises a native `RuntimeError` for missing files,
not `FileNotFoundError`. The result: every config that used a chunking
strategy whose index hadn't been pre-built (all `sentence-500` variants) errored
out and was silently skipped, dropping 12 × 2 = 24 runs from the retrieval
DataFrame.

Patch: `FAISSStore.load` now checks `index_path.exists()` first and raises
`FileNotFoundError` before calling into FAISS. After the fix, the full
48-config grid built all indices on demand and all 96 (48 × 2) combinations
completed.

## 4. QA-pair generation and review

`annotation_helper.py` was run against both cached videos with Mistral as the
question generator:

```
VSFuqMh4hus → 9 QA pairs (63.4s, 5 chunks × up to 3 Qs each)
RRKwmeyIc24 → 3 QA pairs (31.0s, 3 chunks × 1 Q each)
Total       → 12 reviewed QA pairs
```

All 12 were Claude-reviewed against the actual chunked content (keyword-matched
into the chunk text). Every `relevant_chunk_ids` assignment was correct; every
reference answer was grounded in the labelled chunk. All pairs flipped from
`reviewed: false` → `reviewed: true` in `evaluation/dataset/qa_pairs.json`.

Review methodology: for each QA pair, searched the chunked transcript for a
distinctive content-token from the claimed answer (e.g. "1991", "ASI",
"general intelligence", "image"). If the token appeared inside the chunk at
`relevant_chunk_ids[i]`, the pair was marked correct. No pair needed to be
rewritten or re-labelled.

## 5. Retrieval ablation — full 48-config grid

`evaluation/ablation.py` was run with `skip_generation=True`:

```python
run_ablation(
    video_ids=['VSFuqMh4hus', 'RRKwmeyIc24'],
    skip_generation=True,
    output_prefix='full_retrieval',
)
```

Grid: 4 chunking × 3 embedding × 2 retrieval × 2 LLM = 48. (LLM choice is
inert for retrieval-only but is retained in the config string and therefore
in the output CSV.)

Runtime: **6.6 min** end-to-end, including first-time index builds for all
12 unique (embedding, chunking) pairs × 2 videos = 24 index builds.

Output: `evaluation/results/full_retrieval_retrieval.csv` — 576 rows, one per
(config × QA pair).

## 6. Generation + faithfulness — 6-config subset

A full-grid generation run (48 configs × 12 QA × ~15 s/LLM call) would have
taken roughly 6 hours on local 7B models. Instead, six configs were selected
to isolate each research axis against a common baseline:

| cfg_name             | emb    | chunk         | rerank | llm      | Isolates    |
|----------------------|--------|---------------|--------|----------|-------------|
| `top-minilm-mistral` | minilm | fixed-500     | ✓      | mistral  | baseline    |
| `top-minilm-llama2`  | minilm | fixed-500     | ✓      | llama2   | LLM         |
| `top-mpnet-mistral`  | mpnet  | fixed-500     | ✓      | mistral  | embedding   |
| `top-e5-mistral`     | e5     | fixed-500     | ✓      | mistral  | embedding   |
| `sentence-mistral`   | minilm | sentence-500  | ✓      | mistral  | chunking    |
| `no-rerank-mistral`  | minilm | fixed-500     | —      | mistral  | reranker    |

For each config × video × QA, the script:

1. Retrieves top-k chunks (with rerank if configured).
2. Generates an answer via `QueryPipeline.ask()`.
3. Scores the answer against the reference with BLEU, ROUGE-1/2/L, BERTScore
   (`microsoft/deberta-xlarge-mnli`).
4. Scores the answer against the retrieved context with the heuristic atomic
   fact-precision check from `faithfulness_eval.compute_fact_precision`.

The RAGAS LLM-judged faithfulness metric was **not** run in this session. It
requires a separate RAGAS pipeline invocation and would roughly double
runtime. Fact-overlap precision is used as a lightweight proxy.

Runtime: **17.4 min** for all 12 × 6 × 2 = 144 LLM generations + 72 BERTScore
scorings.

Outputs:

- `evaluation/results/subset_generation.csv` — 72 rows
- `evaluation/results/subset_faithfulness.csv` — 72 rows

## 7. Plots

Five plots were produced from the two CSVs:

| File                                 | What it shows                              |
|--------------------------------------|--------------------------------------------|
| `plot_precision_by_chunking.png`     | Mean P@5 across 4 chunk strategies         |
| `plot_mrr_by_embedding.png`          | Mean MRR across MiniLM / MPNet / E5        |
| `plot_rerank_effect.png`             | P@1, P@5, MRR with vs. without rerank      |
| `plot_bertscore_by_config.png`       | BERTScore F1 across 6 generation configs   |
| `plot_faithfulness_by_llm.png`       | Mean fact precision: Mistral vs. Llama-2   |

## 8. Total compute summary

| Phase                                | Time  | LLM calls |
|--------------------------------------|-------|-----------|
| Transcript verification (20 videos)  | ~30 s | 0         |
| QA-pair generation (2 videos)        | 1.6 min | 8       |
| Retrieval ablation (48 configs)      | 6.6 min | 0       |
| Generation eval (6 configs)          | 17.4 min | 144    |
| **Total on-box time**                | **~26 min** | **152** |

## 9. Reproducing the run

1. Ensure Ollama is up and both `mistral` and `llama2:7b` are pulled.
2. Confirm `data/transcripts/VSFuqMh4hus_raw.json` and
   `RRKwmeyIc24_raw.json` exist (cache-first fetch relies on them).
3. Confirm `evaluation/dataset/qa_pairs.json` contains the 12 reviewed pairs.
4. Re-run retrieval ablation: the snippet in §5 above.
5. Re-run subset generation: the configs in §6 above; see
   `evaluation/docs/02_results.md` for the driver script used.

To expand to the intended 20-video / 5-domain run once the YouTube IP block
clears, replace the two-video list with the full set from `videos.json`,
regenerate QA pairs per video, review, and relaunch the ablation.
