"""
Generation quality evaluation.

Computes BLEU, ROUGE-1/2/L, and BERTScore by comparing
generated answers against reference answers from the
annotated QA dataset.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import BERTScorer

from config import PipelineConfig, EVAL_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)


class GenerationEvaluator:
    """
    Evaluates generated answers against reference answers
    using BLEU, ROUGE, and BERTScore.
    """

    def __init__(self):
        self._rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self._bert_scorer = None  # Lazy-loaded
        self._smooth = SmoothingFunction().method1

    def _get_bert_scorer(self) -> BERTScorer:
        """Lazy-load BERTScorer (heavy model)."""
        if self._bert_scorer is None:
            self._bert_scorer = BERTScorer(
                model_type="microsoft/deberta-xlarge-mnli",
                lang="en",
                rescale_with_baseline=True,
            )
        return self._bert_scorer

    def compute_bleu(self, prediction: str, reference: str) -> float:
        """
        Compute sentence-level BLEU score.

        Args:
            prediction: Generated answer.
            reference: Ground-truth answer.

        Returns:
            BLEU score (0.0 to 1.0).
        """
        ref_tokens = reference.lower().split()
        pred_tokens = prediction.lower().split()

        if not ref_tokens or not pred_tokens:
            return 0.0

        return sentence_bleu(
            [ref_tokens], pred_tokens, smoothing_function=self._smooth
        )

    def compute_rouge(self, prediction: str, reference: str) -> dict[str, float]:
        """
        Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

        Returns:
            Dict with keys: rouge1, rouge2, rougeL.
        """
        scores = self._rouge_scorer.score(reference, prediction)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }

    def compute_bertscore(
        self, predictions: list[str], references: list[str]
    ) -> list[dict[str, float]]:
        """
        Compute BERTScore for a batch of prediction-reference pairs.

        Returns:
            List of dicts with keys: precision, recall, f1.
        """
        scorer = self._get_bert_scorer()
        P, R, F1 = scorer.score(predictions, references)

        results = []
        for p, r, f in zip(P.tolist(), R.tolist(), F1.tolist()):
            results.append({
                "bert_precision": p,
                "bert_recall": r,
                "bert_f1": f,
            })
        return results

    def evaluate_batch(
        self,
        predictions: list[str],
        references: list[str],
        metadata: Optional[list[dict]] = None,
    ) -> pd.DataFrame:
        """
        Evaluate a batch of predictions against references.

        Args:
            predictions: List of generated answers.
            references: List of ground-truth answers.
            metadata: Optional list of dicts with extra columns
                      (question, domain, etc.).

        Returns:
            DataFrame with per-sample scores.
        """
        assert len(predictions) == len(references), (
            f"Length mismatch: {len(predictions)} predictions "
            f"vs {len(references)} references"
        )

        results = []

        # Compute BLEU and ROUGE per sample
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            row = {}

            # Add metadata if provided
            if metadata and i < len(metadata):
                row.update(metadata[i])

            row["bleu"] = self.compute_bleu(pred, ref)
            row.update(self.compute_rouge(pred, ref))

            results.append(row)

        # Compute BERTScore in batch (much faster)
        bert_scores = self.compute_bertscore(predictions, references)

        for i, bs in enumerate(bert_scores):
            results[i].update(bs)

        return pd.DataFrame(results)


def evaluate_generation(
    config: PipelineConfig,
    qa_pairs: list[dict],
    generated_answers: list[str],
) -> pd.DataFrame:
    """
    Convenience function: evaluate generated answers against QA pairs.

    Args:
        config: Pipeline config (for labeling).
        qa_pairs: List of QA dicts with "answer" as reference.
        generated_answers: List of generated answer strings.

    Returns:
        DataFrame with all metrics.
    """
    evaluator = GenerationEvaluator()

    references = [qa["answer"] for qa in qa_pairs]
    metadata = [
        {
            "question": qa.get("question", ""),
            "domain": qa.get("domain", "unknown"),
            "video_id": qa.get("video_id", ""),
            "config": str(config),
        }
        for qa in qa_pairs
    ]

    return evaluator.evaluate_batch(generated_answers, references, metadata)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate generation quality")
    parser.add_argument("--predictions", required=True, help="JSON file with generated answers")
    parser.add_argument("--qa-pairs", default=None, help="QA pairs JSON path")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    qa_path = args.qa_pairs or str(EVAL_DIR / "dataset" / "qa_pairs.json")
    with open(qa_path, "r") as f:
        qa_pairs = json.load(f)

    with open(args.predictions, "r") as f:
        predictions = json.load(f)

    cfg = PipelineConfig()  # Default config for labeling
    results = evaluate_generation(cfg, qa_pairs, predictions)

    print("\n=== Generation Evaluation Results ===")
    metric_cols = ["bleu", "rouge1", "rouge2", "rougeL", "bert_f1"]
    print(results[metric_cols].describe().to_string())

    output_path = args.output or str(RESULTS_DIR / "generation_eval.csv")
    results.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
