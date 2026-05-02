"""
@module    StepBackRAG
@description Step-Back RAG — abstracts specific queries into general ones
             before retrieval to find broader relevant context.
             Specific question → step back → general question → search
             → use general context to answer specific question.
             Solves the "too specific to find" problem in RAG systems.
@author    EduMind AI Engineering
"""

import os
import chromadb
from groq import Groq
from typing import List, Tuple

from .base_rag import BaseRAG, RAGQuery, RAGResult


class StepBackRAG(BaseRAG):
    """
    Step-Back RAG — abstraction-first retrieval pipeline.

    Key difference from NaiveRAG:
      - Generates a broader "step-back" version of the query
      - Searches with BOTH specific + general queries
      - Combines results for complete context
      - LLM answers specific question using general context

    Flow:
      1. _generate_stepback()  → LLM abstracts query to general form
      2. retrieve()            → search with both queries, combine
      3. augment()             → build prompt with both contexts
      4. generate()            → answer specific question with broad context
    """

    def __init__(self):
        super().__init__(rag_type="step_back")

        # Connect to ChromaDB
        self._chroma_client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_DB_PATH", "./chroma_db")
        )
        self._collection_name = os.getenv("CHROMA_COLLECTION", "edumind_docs")

        # Connect to Groq LLM
        self._groq_client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        self._model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

    def _generate_stepback(self, query_text: str) -> str:
        """
        Uses LLM to generate a more general "step-back" version of the query.

        Examples:
          Specific: "What is the wavelength of red light in nm?"
          General:  "What is the electromagnetic spectrum?"

          Specific: "What year did Fleming discover penicillin?"
          General:  "What are the major discoveries in medical history?"

          Specific: "What is Newton's 2nd law formula?"
          General:  "What are the fundamental laws of physics?"
        """
        stepback_prompt = f"""You are an expert at abstracting specific questions
into broader, more general questions for better document retrieval.

Given a specific question, generate ONE broader question that:
- Covers the general topic area of the specific question
- Would help find background context to answer the specific question
- Is more general but still relevant

Rules:
- Return ONLY the general question
- No explanation, no preamble
- One question only

Specific question: {query_text}
General question:"""

        try:
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": stepback_prompt}],
                temperature=0.3,
                max_tokens=64
            )
            stepback = response.choices[0].message.content.strip()
            return stepback if len(stepback) > 10 else query_text

        except Exception:
            # Fallback to original query if step-back fails
            return query_text

    def _search_collection(
        self,
        query_text: str,
        top_k: int
    ) -> List[str]:
        """
        Helper: searches ChromaDB with a single query.
        Returns list of chunks or empty list on failure.
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
        except Exception:
            return []

    def retrieve(self, query: RAGQuery) -> List[str]:
        """
        Step 1: Generate step-back query.
                Search with BOTH specific + general queries.
                Combine and deduplicate results.
        """
        try:
            # Generate general step-back query
            stepback_query = self._generate_stepback(query.query_text)

            # Search with specific query (original)
            specific_chunks = self._search_collection(
                query.query_text, query.top_k
            )

            # Search with general step-back query
            general_chunks = self._search_collection(
                stepback_query, query.top_k
            )

            # Combine: specific first, then general additions
            # Specific chunks are more relevant so they go first
            combined = list(dict.fromkeys(specific_chunks + general_chunks))

            return combined[:query.top_k]

        except Exception as e:
            raise RuntimeError(f"[StepBackRAG] retrieve() failed: {str(e)}")

    def augment(self, query: RAGQuery, chunks: List[str]) -> str:
        """
        Step 2: Build prompt that instructs LLM to use
                general context to answer specific question.
        """
        if not chunks:
            context = "No relevant documents found."
        else:
            context = "\n\n".join(
                [f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(chunks)]
            )

        prompt = f"""You are EduMind AI — a helpful education assistant.
The context below contains both specific and general information
that will help you answer the student's specific question accurately.
Use ONLY the context below to answer the question.
If the context does not contain the answer, say "I don't have enough information."

Context:
{context}

Specific Question: {query.query_text}

Answer:"""

        return prompt

    def generate(self, prompt: str) -> str:
        """
        Step 3: Send step-back enriched prompt to Groq and return answer.
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
            raise RuntimeError(f"[StepBackRAG] generate() failed: {str(e)}")