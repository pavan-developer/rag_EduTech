"""
@module    AdaptiveRAG
@description Adaptive RAG — intelligently routes each query to the most
             suitable RAG pipeline based on query analysis.
             Simple questions → NaiveRAG
             Filtered queries → SelfQueryRAG
             Follow-up questions → ContextualRAG
             Complex queries → EnsembleRAG
             Acts as a smart dispatcher — one RAG to rule them all.
@author    EduMind AI Engineering
"""

import os
from groq import Groq
from typing import List

from .base_rag import BaseRAG, RAGQuery, RAGResult
from .naive_rag import NaiveRAG
from .contextual_rag import ContextualRAG
from .self_query_rag import SelfQueryRAG
from .ensemble_rag import EnsembleRAG


# Query type constants — no magic strings
QUERY_TYPE_SIMPLE = "simple"
QUERY_TYPE_FILTERED = "filtered"
QUERY_TYPE_FOLLOWUP = "followup"
QUERY_TYPE_COMPLEX = "complex"


class AdaptiveRAG(BaseRAG):
    """
    Adaptive RAG — intelligent query routing pipeline.

    Key difference from all previous RAGs:
      - Analyzes the query FIRST to classify its type
      - Routes to the best RAG pipeline for that type
      - Result = right tool for right job every time

    Routing logic:
      simple   → NaiveRAG   (plain factual questions)
      filtered → SelfQueryRAG (queries with grade/subject/topic)
      followup → ContextualRAG (pronouns or references to history)
      complex  → EnsembleRAG  (multi-part or broad questions)

    Flow:
      1. classify_query() → LLM classifies query type
      2. route()          → pick the right RAG pipeline
      3. retrieve()       → delegate to chosen RAG
      4. augment()        → delegate to chosen RAG
      5. generate()       → delegate to chosen RAG
    """

    def __init__(self):
        super().__init__(rag_type="adaptive")

        # Initialize all member RAG pipelines
        self._naive = NaiveRAG()
        self._contextual = ContextualRAG()
        self._self_query = SelfQueryRAG()
        self._ensemble = EnsembleRAG()

        # Connect to Groq for query classification
        self._groq_client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        self._model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

        # Track which RAG was selected for last query
        self._selected_rag_type: str = QUERY_TYPE_SIMPLE

    def _classify_query(self, query_text: str, has_history: bool) -> str:
        """
        Uses LLM to classify the query into one of 4 types.
        Returns one of: simple, filtered, followup, complex

        Classification rules:
          - followup: contains pronouns (it, this, that, they) OR has_history=True
          - filtered: contains grade/subject/topic indicators
          - complex:  multi-part question OR asks for comparison/explanation
          - simple:   everything else
        """
        classify_prompt = f"""Classify this student query into exactly one category.

Categories:
- simple:   plain factual question (e.g. "What is photosynthesis?")
- filtered: contains subject/grade/topic filters (e.g. "Class 10 Biology about cells")
- followup: contains pronouns like it/this/that OR is a follow-up to previous question
- complex:  multi-part, comparison, or broad explanation request

Has conversation history: {has_history}
Query: {query_text}

Reply with ONLY one word — the category name:"""

        try:
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": classify_prompt}],
                temperature=0.0,
                max_tokens=10
            )
            raw = response.choices[0].message.content.strip().lower()

            # Validate — only accept known types
            valid_types = [
                QUERY_TYPE_SIMPLE,
                QUERY_TYPE_FILTERED,
                QUERY_TYPE_FOLLOWUP,
                QUERY_TYPE_COMPLEX
            ]
            return raw if raw in valid_types else QUERY_TYPE_SIMPLE

        except Exception:
            # Default to simple if classification fails
            return QUERY_TYPE_SIMPLE

    def _route(self, query_type: str) -> BaseRAG:
        """
        Maps query type to the best RAG pipeline.
        Returns the RAG instance to use for this query.
        """
        routing_map = {
            QUERY_TYPE_SIMPLE:   self._naive,
            QUERY_TYPE_FILTERED: self._self_query,
            QUERY_TYPE_FOLLOWUP: self._contextual,
            QUERY_TYPE_COMPLEX:  self._ensemble,
        }
        return routing_map.get(query_type, self._naive)

    def retrieve(self, query: RAGQuery) -> List[str]:
        """
        Step 1: Classify query → route to best RAG → delegate retrieve().
        """
        has_history = hasattr(self._contextual, '_history') and \
                      len(self._contextual._history) > 0

        query_type = self._classify_query(query.query_text, has_history)
        self._selected_rag_type = query_type

        selected_rag = self._route(query_type)
        return selected_rag.retrieve(query)

    def augment(self, query: RAGQuery, chunks: List[str]) -> str:
        """
        Step 2: Delegate augment() to the selected RAG pipeline.
        """
        selected_rag = self._route(self._selected_rag_type)
        return selected_rag.augment(query, chunks)

    def generate(self, prompt: str) -> str:
        """
        Step 3: Delegate generate() to the selected RAG pipeline.
        """
        selected_rag = self._route(self._selected_rag_type)
        return selected_rag.generate(prompt)

    def run(self, query: RAGQuery) -> RAGResult:
        """
        Full adaptive pipeline — classify → route → run selected RAG.
        Includes selected_rag_type in metadata for transparency.
        """
        chunks = self.retrieve(query)
        prompt = self.augment(query, chunks)
        answer = self.generate(prompt)

        return RAGResult(
            answer=answer,
            source_chunks=chunks,
            rag_type=self.rag_type,
            metadata={
                "query": query.query_text,
                "top_k": query.top_k,
                "routed_to": self._selected_rag_type
            }
        )