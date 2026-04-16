"""
Faithfulness evaluation for fact-dense domains.

Measures:
    1. RAGAS Faithfulness — LLM-judged factual consistency between
       generated answer and retrieved context.
    2. Fact-level precision — extracts atomic facts from the generated
       answer and checks each against the context.

Specifically targets fact-dense domains (news analysis, CS lectures,
technical reviews) where hallucinated details are most consequential.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from config import PipelineConfig, EVAL_DIR, RESULTS_DIR, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

# Domains considered "fact-dense" where hallucination is most harmful
FACT_DENSE_DOMAINS = {"news_analysis", "cs_lectures", "technical_reviews"}


def extract_atomic_facts(text: str) -> list[str]:
    """
    Extract atomic factual claims from a text.

    Uses simple heuristics: splits on sentence boundaries,
    then further splits compound sentences on conjunctions.

    For a more rigorous approach, use an LLM to decompose.
    """
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    sentences = nltk.sent_tokenize(text)

    facts = []
    for sent in sentences:
        # Skip meta-statements
        if any(phrase in sent.lower() for phrase in [
            "i don't have", "based on the context",
            "the video doesn't", "i cannot",
        ]):
            continue

        # Split compound sentences on "and", "but", "also"
        sub_claims = re.split(r"\band\b|\bbut\b|\balso\b", sent)
        for claim in sub_claims:
            claim = claim.strip()
            if len(claim.split()) >= 4:  # Skip very short fragments
                facts.append(claim)

    return facts


def check_fact_in_context(fact: str, context: str) -> bool:
    """
    Heuristic check: is the fact supported by the context?

    Uses token overlap as a proxy. A more rigorous implementation
    would use an NLI model or LLM judge.
    """
    fact_tokens = set(fact.lower().split())
    context_tokens = set(context.lower().split())

    # Remove stopwords for better precision
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after",
        "that", "this", "it", "they", "them", "their", "its",
        "and", "or", "but", "not", "no", "so", "if", "then",
    }

    fact_meaningful = fact_tokens - stopwords
    context_meaningful = context_tokens - stopwords

    if not fact_meaningful:
        return True  # Trivial statement

    overlap = len(fact_meaningful & context_meaningful) / len(fact_meaningful)
    return overlap >= 0.5  # At least 50% token overlap


def compute_fact_precision(
    answer: str,
    context: str,
) -> dict:
    """
    Compute fact-level precision.

    Args:
        answer: Generated answer text.
        context: Concatenated retrieved context chunks.

    Returns:
        Dict with:
            - num_facts: Total atomic facts extracted.
            - supported_facts: Facts found in context.
            - fact_precision: Fraction supported.
            - unsupported: List of unsupported fact strings.
    """
    facts = extract_atomic_facts(answer)

    if not facts:
        return {
            "num_facts": 0,
            "supported_facts": 0,
            "fact_precision": 1.0,  # No claims = no hallucination
            "unsupported": [],
        }

    supported = []
    unsupported = []

    for fact in facts:
        if check_fact_in_context(fact, context):
            supported.append(fact)
        else:
            unsupported.append(fact)

    return {
        "num_facts": len(facts),
        "supported_facts": len(supported),
        "fact_precision": len(supported) / len(facts),
        "unsupported": unsupported,
    }


def evaluate_faithfulness_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    use_local_llm: bool = True,
) -> pd.DataFrame:
    """
    Evaluate faithfulness using the RAGAS framework.

    Args:
        questions: List of questions.
        answers: List of generated answers.
        contexts: List of context lists (one per question).
        use_local_llm: If True, use Ollama as RAGAS judge.

    Returns:
        DataFrame with Faithfulness scores per question.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import Faithfulness
        from datasets import Dataset
    except ImportError:
        logger.error(
            "RAGAS not installed. Run: pip install ragas datasets"
        )
        return pd.DataFrame()

    # Build RAGAS dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    dataset = Dataset.from_dict(data)

    # Configure LLM for evaluation
    eval_kwargs = {}

    if use_local_llm:
        try:
            from langchain_ollama.llms import OllamaLLM
            from ragas.llms import LangchainLLMWrapper

            judge_llm = OllamaLLM(
                model="mistral",
                base_url=OLLAMA_BASE_URL,
                temperature=0.0,
            )
            eval_kwargs["llm"] = LangchainLLMWrapper(judge_llm)
        except Exception as e:
            logger.warning(
                f"Could not configure local LLM for RAGAS: {e}. "
                "Falling back to default (requires OPENAI_API_KEY)."
            )

    # Run RAGAS evaluation
    try:
        result = ragas_evaluate(
            dataset,
            metrics=[Faithfulness()],
            **eval_kwargs,
        )
        return result.to_pandas()
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return pd.DataFrame()


def evaluate_faithfulness(
    qa_pairs: list[dict],
    generated_answers: list[str],
    retrieved_contexts: list[list[str]],
    config: PipelineConfig,
    fact_dense_only: bool = True,
) -> pd.DataFrame:
    """
    Full faithfulness evaluation pipeline.

    Combines RAGAS Faithfulness with fact-level precision,
    optionally filtering to fact-dense domains only.

    Args:
        qa_pairs: Annotated QA pairs.
        generated_answers: LLM-generated answers.
        retrieved_contexts: Context chunks used for each answer.
        config: Pipeline config for labeling.
        fact_dense_only: If True, only evaluate fact-dense domains.

    Returns:
        DataFrame with faithfulness metrics per question.
    """
    results = []

    for i, (qa, answer, contexts) in enumerate(
        zip(qa_pairs, generated_answers, retrieved_contexts)
    ):
        domain = qa.get("domain", "unknown")

        # Filter to fact-dense domains if requested
        if fact_dense_only and domain not in FACT_DENSE_DOMAINS:
            continue

        # Compute fact-level precision
        context_text = " ".join(contexts)
        fact_result = compute_fact_precision(answer, context_text)

        results.append({
            "question": qa.get("question", ""),
            "domain": domain,
            "video_id": qa.get("video_id", ""),
            "config": str(config),
            "llm": config.generation.model_name,
            "num_facts": fact_result["num_facts"],
            "supported_facts": fact_result["supported_facts"],
            "fact_precision": fact_result["fact_precision"],
            "unsupported_examples": json.dumps(fact_result["unsupported"][:3]),
        })

    df = pd.DataFrame(results)

    if not df.empty:
        # Also run RAGAS if available
        if fact_dense_only:
            filtered_qa = [
                qa for qa in qa_pairs
                if qa.get("domain") in FACT_DENSE_DOMAINS
            ]
            filtered_answers = [
                a for a, qa in zip(generated_answers, qa_pairs)
                if qa.get("domain") in FACT_DENSE_DOMAINS
            ]
            filtered_contexts = [
                c for c, qa in zip(retrieved_contexts, qa_pairs)
                if qa.get("domain") in FACT_DENSE_DOMAINS
            ]
        else:
            filtered_qa = qa_pairs
            filtered_answers = generated_answers
            filtered_contexts = retrieved_contexts

        ragas_df = evaluate_faithfulness_ragas(
            questions=[qa["question"] for qa in filtered_qa],
            answers=filtered_answers,
            contexts=filtered_contexts,
        )

        if not ragas_df.empty and "faithfulness" in ragas_df.columns:
            df["ragas_faithfulness"] = ragas_df["faithfulness"].values

    return df


if __name__ == "__main__":
    # Simple test
    answer = "The video discusses machine learning. The accuracy was 95%. They used Python 3.10."
    context = "In this lecture we cover machine learning basics. We achieved 95% accuracy on the test set."

    result = compute_fact_precision(answer, context)
    print(f"Facts extracted: {result['num_facts']}")
    print(f"Supported: {result['supported_facts']}")
    print(f"Precision: {result['fact_precision']:.2f}")
    print(f"Unsupported: {result['unsupported']}")
