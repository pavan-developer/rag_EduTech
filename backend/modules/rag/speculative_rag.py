"""
@module    SpeculativeRAG (HyDE)
@description Hypothetical Document Embedding RAG — generates a hypothetical
             answer first, then searches ChromaDB with that answer instead
             of the original question. Answer-shaped queries find better
             answer-shaped chunks. Also known as HyDE (Hypothetical Document
             Embeddings) in research literature.
@author    EduMind AI Engineering
"""

import os
import chromadb
from groq import Groq
from typing import List

from .base_rag import BaseRAG, RAGQuery, RAGResult


class SpeculativeRAG(BaseRAG):
    """
    Speculative RAG (HyDE) — hypothetical answer retrieval pipeline.

    Key difference from NaiveRAG:
      - Generates a hypothetical answer BEFORE searching
      - Searches ChromaDB with the hypothetical answer
      - Answer-shaped query → finds answer-shaped chunks
      - Result = much better semantic match in vector space

    Flow:
      1. _generate_hypothesis() → LLM writes hypothetical answer
      2. retrieve()             → search ChromaDB with hypothesis
      3. augment()              → build prompt with matched chunks
      4. generate()             → call Groq with real context
    """

    def __init__(self):
        super().__init__(rag_type="speculative")

        self._chroma_client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_DB_PATH", "./chroma_db")
        )
        self._collection_name = os.getenv("CHROMA_COLLECTION", "edumind_docs")

        self._groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

    def _generate_hypothesis(self, query_text: str) -> str:
        """
        Generates a hypothetical answer to the query.
        This hypothetical is used as the search query — not shown to student.

        Example:
          Query:      "What causes thunder?"
          Hypothesis: "Thunder is caused by the rapid expansion of air
                       superheated by a lightning bolt. The sound wave
                       produced travels slower than light, which is why
                       we see lightning before hearing thunder."
        """
        hypothesis_prompt = f"""Write a short, factual paragraph that directly answers
this question as if you were writing a textbook explanation.
Keep it under 80 words. Return ONLY the paragraph — no preamble.

Question: {query_text}
Answer:"""

        try:
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": hypothesis_prompt}],
                temperature=0.4,
                max_tokens=150
            )
            hypothesis = response.choices[0].message.content.strip()
            return hypothesis if len(hypothesis) > 20 else query_text

        except Exception:
            return query_text

    def retrieve(self, query: RAGQuery) -> List[str]:
        """
        Step 1: Generate hypothetical answer.
                Search ChromaDB with hypothesis instead of original query.
                Return best matching chunks.
        """
        try:
            # Generate hypothetical answer
            hypothesis = self._generate_hypothesis(query.query_text)

            collection = self._chroma_client.get_or_create_collection(
                name=self._collection_name
            )

            # Search with hypothesis — not original query
            results = collection.query(
                query_texts=[hypothesis],
                n_results=query.top_k
            )

            return results["documents"][0] if results["documents"] else []

        except Exception as e:
            raise RuntimeError(f"[SpeculativeRAG] retrieve() failed: {str(e)}")

    def augment(self, query: RAGQuery, chunks: List[str]) -> str:
        """
        Step 2: Build prompt with hypothesis-matched chunks.
        Note: We answer the ORIGINAL question — not the hypothesis.
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
        """Step 3: Send prompt to Groq and return answer."""
        try:
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"[SpeculativeRAG] generate() failed: {str(e)}")