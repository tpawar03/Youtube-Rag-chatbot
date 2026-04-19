"""
LangChain RAG chain with conversational memory.

Uses LangChain Expression Language (LCEL) to build a modern RAG chain
that supports multi-turn dialogue with context-aware follow-ups.
"""

from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.language_models import BaseLanguageModel

from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.generation.prompts import (
    CONVERSATIONAL_PROMPT,
    CONDENSE_PROMPT,
    SIMPLE_QA_PROMPT,
    OVERVIEW_PROMPT,
    STYLE_DIRECTIVES,
    OFF_TOPIC_MESSAGE,
    format_context,
    parse_confidence,
    parse_overview,
)
from src.generation.grounding import answer_is_grounded


def is_off_topic(candidates: list[dict], threshold: float) -> bool:
    """
    Decide whether the retrieval result is weak enough that the question
    should be considered unrelated to the video. The store uses FAISS
    L2-distance scores (lower = better match), so a top score AT OR ABOVE
    the threshold means nothing in the index is close enough.
    """
    if not candidates:
        return True
    top_score = candidates[0].get("score")
    if top_score is None:
        return False
    return float(top_score) >= threshold
from config import PipelineConfig


class RAGChain:
    """
    Conversational RAG chain that retrieves context from the FAISS
    index and generates grounded answers with conversation memory.

    Uses LCEL instead of deprecated ConversationalRetrievalChain.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        retriever: Retriever,
        reranker: Optional[Reranker] = None,
        config: Optional[PipelineConfig] = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.config = config or PipelineConfig()

        # Conversation history as list of messages
        self._chat_history: list[HumanMessage | AIMessage] = []

        # Build chains
        self._condense_chain = (
            CONDENSE_PROMPT | self.llm | StrOutputParser()
        )
        self._qa_chain = (
            CONVERSATIONAL_PROMPT | self.llm | StrOutputParser()
        )

    def _condense_question(self, question: str) -> str:
        """
        If there's chat history, rephrase the follow-up question
        as a standalone question. Otherwise, return as-is.
        """
        if not self._chat_history:
            return question

        return self._condense_chain.invoke({
            "chat_history": self._chat_history,
            "question": question,
        })

    def _prepare_context(
        self, question: str
    ) -> tuple[str, list[dict], list[dict], bool]:
        """
        Run the retrieval-side of the pipeline: condense → retrieve → check
        off-topic gate on RAW FAISS scores → rerank → format_context.

        Returns:
            (context_str, candidate_chunks, source_dicts, off_topic)
        """
        standalone_question = self._condense_question(question)
        raw_candidates = self.retriever.retrieve(standalone_question)

        # Off-topic gate on RAW FAISS L2 scores — must happen BEFORE the
        # reranker rewrites candidates[i]["score"] with cross-encoder logits
        # that aren't comparable to the threshold.
        off_topic = is_off_topic(
            raw_candidates, self.config.retrieval.off_topic_threshold
        )
        if off_topic:
            return "", [], [], True

        if self.reranker and self.config.retrieval.use_reranker:
            candidates = self.reranker.rerank(
                standalone_question,
                raw_candidates,
                top_k=self.config.retrieval.top_k,
            )
        else:
            candidates = raw_candidates[:self.config.retrieval.top_k]

        context = format_context(candidates)

        sources = []
        for chunk in candidates:
            text = chunk["text"]
            sources.append({
                "text": text[:200] + "..." if len(text) > 200 else text,
                "start_time": chunk.get("start_time", 0),
                "end_time": chunk.get("end_time", 0),
                "video_id": chunk.get("video_id", ""),
                "score": chunk.get("score", 0),
            })

        return context, candidates, sources, False

    def _generate_with_context(
        self,
        question: str,
        context: str,
        prompt_style: str,
    ) -> str:
        """Invoke the QA chain with the prepared context and style directive."""
        return self._qa_chain.invoke({
            "context": context,
            "chat_history": self._chat_history,
            "question": question,
            "style_directive": STYLE_DIRECTIVES.get(prompt_style, ""),
        })

    def ask(self, question: str, prompt_style: str = "default") -> dict:
        """
        Ask a question and get a grounded answer.

        Args:
            question: User's question.
            prompt_style: One of STYLE_DIRECTIVES keys ("default"/"concise"/"detailed").

        Returns:
            Dict with keys:
                - answer (str): Generated response (Confidence line stripped).
                - sources (list[dict]): Source chunks with timestamps.
                - candidates (list[dict]): Raw candidate chunks (for grounding).
                - confidence (int | None): 1-5 self-rating, or None if absent.
        """
        context, candidates, sources, off_topic = self._prepare_context(question)

        # Layer 1: retrieval-side gate (raw FAISS scores, pre-rerank).
        if off_topic:
            self._chat_history.append(HumanMessage(content=question))
            self._chat_history.append(AIMessage(content=OFF_TOPIC_MESSAGE))
            return {
                "answer": OFF_TOPIC_MESSAGE,
                "sources": [],
                "candidates": [],
                "confidence": None,
                "off_topic": True,
            }

        raw_answer = self._generate_with_context(question, context, prompt_style)
        answer, confidence = parse_confidence(raw_answer)

        # Layer 2: post-generation safety net. If retrieval borderline-passed
        # but the LLM produced an answer with zero grounded sentences, treat
        # it as off-topic — the LLM is almost certainly using outside
        # knowledge or confabulating.
        if answer and not answer_is_grounded(answer, candidates):
            self._chat_history.append(HumanMessage(content=question))
            self._chat_history.append(AIMessage(content=OFF_TOPIC_MESSAGE))
            return {
                "answer": OFF_TOPIC_MESSAGE,
                "sources": [],
                "candidates": [],
                "confidence": None,
                "off_topic": True,
            }

        # Store the cleaned answer in history so follow-up condensing
        # doesn't have to reason about the trailing confidence line.
        self._chat_history.append(HumanMessage(content=question))
        self._chat_history.append(AIMessage(content=answer))

        return {
            "answer": answer,
            "sources": sources,
            "candidates": candidates,
            "confidence": confidence,
            "off_topic": False,
        }

    def generate_overview(self, transcript_text: str) -> dict:
        """
        Produce a summary, key topics, and suggested questions from the
        full transcript text. Does not use retrieval — the LLM reads the
        whole (possibly truncated) transcript in a single pass.

        Long transcripts are truncated head+tail to fit in the LLM's
        context: middle content is dropped, which preserves intro/outro
        where video framing typically lives.

        Returns:
            {"summary": str, "topics": list[str], "suggested_questions": list[str]}
        """
        MAX_CHARS = 12000
        text = transcript_text
        if len(text) > MAX_CHARS:
            half = MAX_CHARS // 2
            text = text[:half] + "\n\n[...transcript truncated...]\n\n" + text[-half:]

        overview_chain = OVERVIEW_PROMPT | self.llm | StrOutputParser()
        raw = overview_chain.invoke({"transcript": text})
        return parse_overview(raw)

    def ask_simple(self, question: str, context_chunks: list[dict]) -> str:
        """
        Simple single-turn QA without memory (for evaluation).

        Args:
            question: The question.
            context_chunks: Pre-retrieved chunks.

        Returns:
            Generated answer string.
        """
        context = format_context(context_chunks)
        simple_chain = SIMPLE_QA_PROMPT | self.llm | StrOutputParser()
        return simple_chain.invoke({
            "context": context,
            "question": question,
        })

    def reset_memory(self):
        """Clear conversation history."""
        self._chat_history.clear()

    @property
    def chat_history(self) -> list:
        """Return current chat history."""
        return list(self._chat_history)
