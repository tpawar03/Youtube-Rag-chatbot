"""
LangChain RAG chain with conversational memory.

Uses LangChain Expression Language (LCEL) to build a modern RAG chain
that supports multi-turn dialogue with context-aware follow-ups.
"""

import re
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
    get_conversational_prompt,
    format_context,
)
from config import PipelineConfig

_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*([1-5])\s*/\s*5", re.IGNORECASE)


def _parse_confidence(raw: str) -> tuple[str, int | None]:
    """Strip the CONFIDENCE line from the answer and return (clean_answer, confidence)."""
    match = _CONFIDENCE_RE.search(raw)
    if match:
        confidence = int(match.group(1))
        clean = _CONFIDENCE_RE.sub("", raw).rstrip()
        return clean, confidence
    return raw, None


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

    def ask(self, question: str, prompt_style: str = "default") -> dict:
        """
        Ask a question and get a grounded answer.

        Args:
            question: User's question.
            prompt_style: One of "default", "concise", "detailed", "eli5".

        Returns:
            Dict with keys:
                - answer (str): Generated response (confidence line stripped).
                - sources (list[dict]): Source chunks with timestamps.
                - confidence (int | None): Self-rated confidence 1-5.
        """
        standalone_question = self._condense_question(question)

        candidates = self.retriever.retrieve(standalone_question)
        if self.reranker and self.config.retrieval.use_reranker:
            candidates = self.reranker.rerank(
                standalone_question,
                candidates,
                top_k=self.config.retrieval.top_k,
            )
        else:
            candidates = candidates[:self.config.retrieval.top_k]

        context = format_context(candidates)

        qa_chain = get_conversational_prompt(prompt_style) | self.llm | StrOutputParser()
        raw = qa_chain.invoke({
            "context": context,
            "chat_history": self._chat_history,
            "question": question,
        })
        answer, confidence = _parse_confidence(raw)

        self._chat_history.append(HumanMessage(content=question))
        self._chat_history.append(AIMessage(content=answer))

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

        return {"answer": answer, "sources": sources, "confidence": confidence}

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

    def generate_from_context(
        self, question: str, context: str, candidates: list[dict],
        prompt_style: str = "default",
    ) -> dict:
        """
        Generate an answer from pre-retrieved context (no retrieval step).
        Used for dual-model comparison where retrieval is shared.
        """
        qa_chain = get_conversational_prompt(prompt_style) | self.llm | StrOutputParser()
        raw = qa_chain.invoke({
            "context": context,
            "chat_history": self._chat_history,
            "question": question,
        })
        answer, confidence = _parse_confidence(raw)

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

        return {"answer": answer, "sources": sources, "confidence": confidence}

    def update_memory(self, question: str, answer: str):
        """Append a Q/A turn to chat history (used after dual-model selection)."""
        self._chat_history.append(HumanMessage(content=question))
        self._chat_history.append(AIMessage(content=answer))

    def reset_memory(self):
        """Clear conversation history."""
        self._chat_history.clear()

    @property
    def chat_history(self) -> list:
        """Return current chat history."""
        return list(self._chat_history)
