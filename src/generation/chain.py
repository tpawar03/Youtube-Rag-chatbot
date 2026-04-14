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
    format_context,
)
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

    def ask(self, question: str) -> dict:
        """
        Ask a question and get a grounded answer.

        Args:
            question: User's question.

        Returns:
            Dict with keys:
                - answer (str): Generated response.
                - sources (list[dict]): Source chunks with timestamps.
        """
        # Step 1: Condense question if follow-up
        standalone_question = self._condense_question(question)

        # Step 2: Retrieve candidates
        candidates = self.retriever.retrieve(standalone_question)

        # Step 3: Optionally rerank
        if self.reranker and self.config.retrieval.use_reranker:
            candidates = self.reranker.rerank(
                standalone_question,
                candidates,
                top_k=self.config.retrieval.top_k,
            )
        else:
            candidates = candidates[:self.config.retrieval.top_k]

        # Step 4: Format context
        context = format_context(candidates)

        # Step 5: Generate answer
        answer = self._qa_chain.invoke({
            "context": context,
            "chat_history": self._chat_history,
            "question": question,
        })

        # Step 6: Update chat history
        self._chat_history.append(HumanMessage(content=question))
        self._chat_history.append(AIMessage(content=answer))

        # Step 7: Build source info
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

        return {
            "answer": answer,
            "sources": sources,
        }

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
