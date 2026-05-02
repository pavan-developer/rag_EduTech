"""
@module    HybridRAG
@description Hybrid RAG — combines semantic search (ChromaDB) with
             keyword search (BM25) for higher retrieval accuracy.
             Semantic search finds meaning-similar chunks.
             BM25 finds exact keyword matches.
             Final result = union of both, deduplicated.
@author    EduMind AI Engineering
"""

import os
import chromadb
from groq import Groq
from typing import List
from rank_bm25 import BM25Okapi

from .base_rag import BaseRAG, RAGQuery, RAGResult


class HybridRAG(BaseRAG):
    """
    Hybrid RAG — two-way retrieval pipeline.

    Flow:
      1. retrieve()  → semantic search (ChromaDB) + keyword search (BM25)
                       combine and deduplicate results
      2. augment()   → combine query + hybrid chunks into prompt
      3. generate()  → send prompt to Groq Llama-3, get answer
    """

    def __init__(self):
        super().__init__(rag_type="hybrid")

        # Connect to ChromaDB for semantic search
        self._chroma_client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_DB_PATH", "./chroma_db")
        )
        self._collection_name = os.getenv("CHROMA_COLLECTION", "edumind_docs")

        # Connect to Groq LLM
        self._groq_client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        self._model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

    def _semantic_search(self, query_text: str, top_k: int) -> List[str]:
        """
        Semantic search using ChromaDB.
        Finds chunks that are similar in MEANING to the query.
        """
        try:
            collection = self._chroma_client.get_or_create_collection(
                name=self._collection_name
            )
            results = collection.query(
                query_texts=[query_text],
                n_results=top_k
            )
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            raise RuntimeError(f"[HybridRAG] semantic_search() failed: {str(e)}")

    def _keyword_search(self, query_text: str, chunks: List[str], top_k: int) -> List[str]:
        """
        Keyword search using BM25.
        Finds chunks that contain the exact WORDS from the query.
        BM25 is the same algorithm used by Elasticsearch and Solr.
        """
        if not chunks:
            return []

        try:
            # Tokenize each chunk into words
            tokenized_chunks = [chunk.lower().split() for chunk in chunks]

            # Build BM25 index from all chunks
            bm25 = BM25Okapi(tokenized_chunks)

            # Score each chunk against the query
            tokenized_query = query_text.lower().split()
            scores = bm25.get_scores(tokenized_query)

            # Return top_k highest scoring chunks
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]

            return [chunks[i] for i in top_indices]

        except Exception as e:
            raise RuntimeError(f"[HybridRAG] keyword_search() failed: {str(e)}")

    def retrieve(self, query: RAGQuery) -> List[str]:
        """
        Step 1: Run BOTH semantic and keyword search.
        Combine results and remove duplicates.
        """
        # Semantic search — meaning based
        semantic_chunks = self._semantic_search(query.query_text, query.top_k)

        # Keyword search — word match based (searches within semantic results)
        keyword_chunks = self._keyword_search(
            query.query_text, semantic_chunks, query.top_k
        )

        # Combine both — semantic first, then keyword additions
        # dict.fromkeys preserves order and removes duplicates
        combined = list(dict.fromkeys(semantic_chunks + keyword_chunks))

        return combined[:query.top_k]

    def augment(self, query: RAGQuery, chunks: List[str]) -> str:
        """
        Step 2: Build prompt with hybrid retrieved chunks.
        Labels each source so LLM knows context came from documents.
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
        Step 3: Send prompt to Groq Llama-3 and return answer.
        """
        try:
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1024
            )
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"[HybridRAG] generate() failed: {str(e)}")