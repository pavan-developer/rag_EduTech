"""
@module    BaseRAG
@description Abstract base class for all 16 RAG pipeline implementations.
             Follows Strategy Pattern — each RAG type is a concrete strategy
             that inherits this interface. Core pipeline never changes when
             new RAG types are added (Open/Closed Principle).
@author    EduMind AI Engineering
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class RAGQuery:
    """Represents a user query entering any RAG pipeline."""
    query_text: str
    top_k: int = 5
    filters: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None


@dataclass
class RAGResult:
    """Represents the output from any RAG pipeline."""
    answer: str
    source_chunks: List[str]
    rag_type: str
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRAG(ABC):
    """
    Abstract base class that every RAG implementation must inherit.

    Contract:
      - retrieve()  -> fetch relevant document chunks from vector store
      - augment()   -> combine query + chunks into a prompt
      - generate()  -> call LLM and return final answer
      - run()       -> full pipeline: retrieve -> augment -> generate
    """

    def __init__(self, rag_type: str):
        self.rag_type = rag_type

    @abstractmethod
    def retrieve(self, query: RAGQuery) -> List[str]:
        """Retrieve relevant chunks from the vector store."""
        ...

    @abstractmethod
    def augment(self, query: RAGQuery, chunks: List[str]) -> str:
        """Build the final prompt by combining query and retrieved chunks."""
        ...

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send prompt to LLM and return the generated answer."""
        ...

    def run(self, query: RAGQuery) -> RAGResult:
        """
        Full RAG pipeline: retrieve -> augment -> generate.
        This method is FINAL — subclasses override the 3 steps above, not this.
        """
        chunks = self.retrieve(query)
        prompt = self.augment(query, chunks)
        answer = self.generate(prompt)

        return RAGResult(
            answer=answer,
            source_chunks=chunks,
            rag_type=self.rag_type,
            metadata={"query": query.query_text, "top_k": query.top_k}
        )