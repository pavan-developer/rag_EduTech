"""
@module    EnsembleRAG
@description Ensemble RAG — runs multiple RAG pipelines in parallel and
             combines their results using a voting/scoring mechanism.
             Chunks found by more RAGs score higher — consensus = quality.
             Most powerful retrieval strategy for maximum recall.
@author    EduMind AI Engineering
"""

import os
import chromadb
from groq import Groq
from typing import List, Dict
from collections import Counter

from .base_rag import BaseRAG, RAGQuery, RAGResult
from .naive_rag import NaiveRAG
from .hybrid_rag import HybridRAG


class EnsembleRAG(BaseRAG):
    """
    Ensemble RAG — multi-pipeline consensus retrieval.

    Key difference from all previous RAGs:
      - Runs NaiveRAG + HybridRAG retrievers in parallel
      - Scores each chunk by how many retrievers found it
      - Chunks agreed upon by multiple RAGs rank higher
      - Result = highest quality, highest recall retrieval

    Flow:
      1. retrieve()  → run all RAGs → score chunks by frequency
      2. augment()   → build prompt with consensus-ranked chunks
      3. generate()  → call Groq with best chunks, get answer
    """

    def __init__(self):
        super().__init__(rag_type="ensemble")

        # Initialize member RAG pipelines
        # These are the "experts" we ask in parallel
        self._retrievers = [
            NaiveRAG(),
            HybridRAG(),
        ]

        # Connect to Groq LLM for generation
        self._groq_client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        self._model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

    def _score_chunks(self, all_chunks: List[List[str]]) -> List[str]:
        """
        Scores chunks by how many retrievers found them.
        Chunks found by multiple retrievers rank higher.

        Example:
          NaiveRAG  found: [chunk_A, chunk_B, chunk_C]
          HybridRAG found: [chunk_B, chunk_D, chunk_E]

          Scores:
            chunk_A → 1 (found by 1 retriever)
            chunk_B → 2 (found by 2 retrievers) ← ranked highest
            chunk_C → 1
            chunk_D → 1
            chunk_E → 1

          Sorted result: [chunk_B, chunk_A, chunk_C, chunk_D, chunk_E]
        """
        # Flatten all chunks into one list
        flat_chunks = [
            chunk
            for retriever_chunks in all_chunks
            for chunk in retriever_chunks
        ]

        if not flat_chunks:
            return []

        # Count how many times each chunk appears
        chunk_counts = Counter(flat_chunks)

        # Sort by count (descending) — most agreed-upon chunks first
        ranked_chunks = sorted(
            chunk_counts.keys(),
            key=lambda chunk: chunk_counts[chunk],
            reverse=True
        )

        return ranked_chunks

    def retrieve(self, query: RAGQuery) -> List[str]:
        """
        Step 1: Run all member retrievers in parallel.
                Score chunks by consensus. Return top_k ranked chunks.
        """
        all_chunks: List[List[str]] = []

        # Run each retriever and collect their chunks
        for retriever in self._retrievers:
            try:
                chunks = retriever.retrieve(query)
                all_chunks.append(chunks)
            except Exception:
                # If one retriever fails, continue with others
                all_chunks.append([])

        # Score and rank chunks by consensus
        ranked_chunks = self._score_chunks(all_chunks)

        return ranked_chunks[:query.top_k]

    def augment(self, query: RAGQuery, chunks: List[str]) -> str:
        """
        Step 2: Build prompt with consensus-ranked chunks.
        """
        if not chunks:
            context = "No relevant documents found."
        else:
            context = "\n\n".join(
                [f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(chunks)]
            )

        prompt = f"""You are EduMind AI — a helpful education assistant.
Use ONLY the context below to answer the question.
If the context does not contain the answer, say "I don't have enough information."

Context:
{context}

Question: {query.query_text}

Answer:"""

        return prompt

    def generate(self, prompt: str) -> str:
        """
        Step 3: Send consensus-ranked prompt to Groq and return answer.
        """
        try:
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024
            )
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"[EnsembleRAG] generate() failed: {str(e)}")