"""
@module    ReRankRAG
@description Re-Rank RAG — two-stage retrieval pipeline.
             Stage 1: Fast broad retrieval (ChromaDB) — get top-N candidates
             Stage 2: Accurate deep reranking (LLM) — score and reorder top-K
             Same pattern used in Google Search, Cohere Rerank, and
             every enterprise production RAG system at scale.
@author    EduMind AI Engineering
"""

import os
import chromadb
from groq import Groq
from typing import List, Tuple

from .base_rag import BaseRAG, RAGQuery, RAGResult


class ReRankRAG(BaseRAG):
    """
    ReRank RAG — two-stage retrieve-then-rerank pipeline.

    Key difference from all previous RAGs:
      - Stage 1: Retrieve broad candidates (top_k * 3) from ChromaDB
      - Stage 2: LLM scores each candidate for exact relevance
      - Final: Return top_k highest-scored chunks
      - Result = fast retrieval + accurate ranking

    Flow:
      1. retrieve()    → broad ChromaDB search (3x candidates)
      2. _rerank()     → LLM scores each chunk 0-10
      3. augment()     → build prompt with reranked top chunks
      4. generate()    → call Groq with best-ranked context
    """

    def __init__(self, candidates_multiplier: int = 3):
        super().__init__(rag_type="rerank")

        # Fetch this many candidates before reranking
        self._candidates_multiplier = int(
            os.getenv("RERANK_CANDIDATES_MULTIPLIER", str(candidates_multiplier))
        )

        self._chroma_client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_DB_PATH", "./chroma_db")
        )
        self._collection_name = os.getenv("CHROMA_COLLECTION", "edumind_docs")

        self._groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

    def _score_chunk(self, query_text: str, chunk: str) -> float:
        """
        Uses LLM to score a single chunk's relevance to the query.
        Returns a float score from 0.0 to 10.0.

        Unlike CorrectiveRAG which gives binary relevant/irrelevant,
        ReRankRAG gives a fine-grained numeric score for precise ordering.

        Example:
          Query: "What is Newton's second law?"
          Chunk: "F=ma is Newton's second law..."     → score: 9.5
          Chunk: "Newton discovered gravity in 1687"  → score: 4.0
          Chunk: "Crispy fried chicken recipe..."     → score: 0.5
        """
        score_prompt = f"""You are a relevance scoring system.
Score how relevant this document chunk is to the student's question.

Scoring scale:
  0-2:  completely irrelevant
  3-4:  loosely related
  5-6:  somewhat relevant
  7-8:  relevant and helpful
  9-10: perfectly answers the question

Rules:
- Return ONLY a number between 0 and 10
- One decimal place allowed (e.g. 7.5)
- No explanation, no preamble

Question: {query_text}
Chunk: {chunk[:300]}

Score:"""

        try:
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": score_prompt}],
                temperature=0.0,
                max_tokens=10
            )
            raw = response.choices[0].message.content.strip()
            score = float(raw)
            # Clamp between 0 and 10
            return max(0.0, min(10.0, score))

        except Exception:
            # Default middle score if scoring fails
            return 5.0

    def _rerank(
        self,
        query_text: str,
        chunks: List[str],
        top_k: int
    ) -> List[str]:
        """
        Scores all candidate chunks and returns top_k by score.

        Example:
          candidates = [chunk_A, chunk_B, chunk_C, chunk_D, chunk_E]
          scores     = [9.5,     4.0,     7.5,     2.0,     8.0   ]
          ranked     = [chunk_A, chunk_E, chunk_C, chunk_B, chunk_D]
          top_3      = [chunk_A, chunk_E, chunk_C]
        """
        if not chunks:
            return []

        # Score each chunk
        scored: List[Tuple[str, float]] = []
        for chunk in chunks:
            score = self._score_chunk(query_text, chunk)
            scored.append((chunk, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top_k chunks only
        return [chunk for chunk, score in scored[:top_k]]

    def retrieve(self, query: RAGQuery) -> List[str]:
        """
        Step 1: Broad retrieval (top_k * multiplier candidates).
                Deep reranking (LLM scores each candidate).
                Return top_k highest scored chunks.
        """
        try:
            collection = self._chroma_client.get_or_create_collection(
                name=self._collection_name
            )

            # Stage 1 — broad retrieval (more candidates than needed)
            num_candidates = query.top_k * self._candidates_multiplier
            results = collection.query(
                query_texts=[query.query_text],
                n_results=num_candidates
            )

            if not results["documents"] or not results["documents"][0]:
                return []

            candidates = results["documents"][0]

            # Stage 2 — deep reranking
            reranked = self._rerank(query.query_text, candidates, query.top_k)

            return reranked

        except Exception as e:
            raise RuntimeError(f"[ReRankRAG] retrieve() failed: {str(e)}")

    def augment(self, query: RAGQuery, chunks: List[str]) -> str:
        if not chunks:
            context = "No relevant documents found."
        else:
            context = "\n\n".join(
                [f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(chunks)]
            )

        prompt = f"""You are EduMind AI — a helpful education assistant.
The context below has been carefully ranked by relevance to your question.
Use ONLY the context below to answer the question.
If the context does not contain the answer, say "I don't have enough information."

Context:
{context}

Question: {query.query_text}

Answer:"""
        return prompt

    def generate(self, prompt: str) -> str:
        try:
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"[ReRankRAG] generate() failed: {str(e)}")