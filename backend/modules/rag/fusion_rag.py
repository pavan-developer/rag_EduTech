"""
@module    FusionRAG
@description Fusion RAG — uses Reciprocal Rank Fusion (RRF) to combine
             results from multiple searches with position-aware scoring.
             RRF formula: score = 1/(rank + 60)
             Same algorithm used in Azure AI Search, Elasticsearch 8.x,
             and LangChain EnsembleRetriever.
@author    EduMind AI Engineering
"""

import os
import chromadb
from groq import Groq
from typing import List, Dict
from .base_rag import BaseRAG, RAGQuery, RAGResult


class FusionRAG(BaseRAG):
    """
    Fusion RAG — Reciprocal Rank Fusion retrieval pipeline.

    Key difference from EnsembleRAG:
      - EnsembleRAG scores by COUNT (found by N retrievers)
      - FusionRAG scores by POSITION (rank in each result list)
      - RRF: score = 1/(rank + 60) — higher rank = higher score
      - Position-aware = more accurate than simple vote counting

    Flow:
      1. _multi_search()  → run N searches with query variations
      2. _rrf_score()     → score chunks by position using RRF
      3. retrieve()       → return top_k RRF-ranked chunks
      4. augment()        → build prompt with fusion-ranked chunks
      5. generate()       → call Groq, get answer
    """

    # RRF smoothing constant — from original 2009 research paper
    RRF_K = 60

    def __init__(self, num_searches: int = 3):
        super().__init__(rag_type="fusion")

        self._num_searches = int(os.getenv("FUSION_SEARCH_COUNT", str(num_searches)))

        self._chroma_client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_DB_PATH", "./chroma_db")
        )
        self._collection_name = os.getenv("CHROMA_COLLECTION", "edumind_docs")

        self._groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

    def _generate_query_variations(self, query_text: str) -> List[str]:
        """
        Generates N variations of the query for multi-search fusion.
        Similar to MultiQueryRAG but used specifically for RRF scoring.
        """
        variation_prompt = f"""Generate {self._num_searches - 1} alternative ways to search for the same information.
Each variation should use different keywords but seek the same answer.
Return ONLY the questions, one per line, no numbering.

Original: {query_text}
Variations:"""

        try:
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": variation_prompt}],
                temperature=0.7,
                max_tokens=200
            )
            raw = response.choices[0].message.content.strip()
            variations = [
                line.strip()
                for line in raw.split("\n")
                if line.strip() and len(line.strip()) > 10
            ]
            return [query_text] + variations[:self._num_searches - 1]
        except Exception:
            return [query_text]

    def _multi_search(
        self,
        queries: List[str],
        top_k: int
    ) -> List[List[str]]:
        """
        Runs ChromaDB search for each query variation.
        Returns list of ranked result lists — one list per query.

        Example:
          queries = ["Newton's laws", "force and motion", "F=ma"]
          returns = [
            [chunk_A, chunk_B, chunk_C],  ← ranked results for query 1
            [chunk_B, chunk_D, chunk_E],  ← ranked results for query 2
            [chunk_A, chunk_C, chunk_F],  ← ranked results for query 3
          ]
        """
        collection = self._chroma_client.get_or_create_collection(
            name=self._collection_name
        )

        all_ranked_lists: List[List[str]] = []

        for query in queries:
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=top_k
                )
                if results["documents"] and results["documents"][0]:
                    all_ranked_lists.append(results["documents"][0])
                else:
                    all_ranked_lists.append([])
            except Exception:
                all_ranked_lists.append([])

        return all_ranked_lists

    def _rrf_score(
        self,
        ranked_lists: List[List[str]]
    ) -> List[str]:
        """
        Applies Reciprocal Rank Fusion formula to combine ranked lists.

        RRF formula: score(chunk) = Σ 1/(rank + K)
          where K=60 (smoothing constant from original paper)
          rank is 1-indexed position in each result list

        Example:
          ranked_lists = [
            [chunk_A, chunk_B, chunk_C],
            [chunk_B, chunk_C, chunk_A],
          ]

          chunk_A: 1/(1+60) + 1/(3+60) = 0.0164 + 0.0159 = 0.0323
          chunk_B: 1/(2+60) + 1/(1+60) = 0.0161 + 0.0164 = 0.0325 ← winner
          chunk_C: 1/(3+60) + 1/(2+60) = 0.0159 + 0.0161 = 0.0320
        """
        rrf_scores: Dict[str, float] = {}

        for ranked_list in ranked_lists:
            for rank, chunk in enumerate(ranked_list, start=1):
                # RRF formula — rank is 1-indexed
                score = 1.0 / (rank + self.RRF_K)
                rrf_scores[chunk] = rrf_scores.get(chunk, 0.0) + score

        # Sort by RRF score descending
        ranked_chunks = sorted(
            rrf_scores.keys(),
            key=lambda c: rrf_scores[c],
            reverse=True
        )

        return ranked_chunks

    def retrieve(self, query: RAGQuery) -> List[str]:
        """
        Step 1: Generate query variations → multi-search → RRF rank.
        """
        try:
            queries = self._generate_query_variations(query.query_text)
            ranked_lists = self._multi_search(queries, query.top_k)
            rrf_ranked = self._rrf_score(ranked_lists)
            return rrf_ranked[:query.top_k]

        except Exception as e:
            raise RuntimeError(f"[FusionRAG] retrieve() failed: {str(e)}")

    def augment(self, query: RAGQuery, chunks: List[str]) -> str:
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
        try:
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"[FusionRAG] generate() failed: {str(e)}")