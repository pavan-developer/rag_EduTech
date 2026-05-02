"""
@module    SelfQueryRAG
@description Self-Query RAG — uses LLM to extract structured metadata filters
             from natural language queries before retrieval.
             Example: "Class 10 Biology chapters about cells"
             → filters: {subject: "Biology", grade: "Class 10", topic: "cells"}
             Then searches ChromaDB with those filters for precise results.
@author    EduMind AI Engineering
"""

import os
import json
import chromadb
from groq import Groq
from typing import List, Dict, Any, Optional

from .base_rag import BaseRAG, RAGQuery, RAGResult


class SelfQueryRAG(BaseRAG):
    """
    Self-Query RAG — structured filter extraction pipeline.

    Key difference from NaiveRAG:
      - Before retrieval, LLM parses query into structured filters
      - ChromaDB search uses both semantic similarity AND metadata filters
      - Result = highly precise retrieval for structured queries

    Flow:
      1. extract_filters() → LLM parses query → returns JSON filters
      2. retrieve()        → ChromaDB search with filters applied
      3. augment()         → build prompt with filtered chunks
      4. generate()        → call Groq, get precise answer
    """

    def __init__(self):
        super().__init__(rag_type="self_query")

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

    def _extract_filters(self, query_text: str) -> Dict[str, Any]:
        """
        Uses LLM to extract structured metadata filters from natural language.

        Example inputs and outputs:
          "Class 10 Biology chapters about cells"
          → {"subject": "Biology", "grade": "Class 10", "topic": "cells"}

          "Show me Physics problems about Newton's laws"
          → {"subject": "Physics", "topic": "Newton's laws"}

          "What is photosynthesis?"
          → {}  (no filters — plain question)
        """
        extract_prompt = f"""You are a metadata extractor for an education platform.
Extract structured filters from the student's query.

Possible filter fields:
- subject: (Biology, Physics, Chemistry, Math, History, Geography, English)
- grade: (Class 1 through Class 12)
- topic: (specific topic name)
- difficulty: (easy, medium, hard)
- doc_type: (textbook, notes, exercises, examples)

Rules:
- Return ONLY a valid JSON object
- Only include fields that are clearly present in the query
- If no filters found, return empty object: {{}}
- No explanation, no preamble, no markdown — just raw JSON

Query: {query_text}
JSON filters:"""

        try:
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": extract_prompt}],
                temperature=0.0,
                max_tokens=128
            )
            raw = response.choices[0].message.content.strip()

            # Clean any accidental markdown fences
            raw = raw.replace("```json", "").replace("```", "").strip()

            filters = json.loads(raw)
            return filters if isinstance(filters, dict) else {}

        except Exception:
            # If extraction fails, return empty filters — fallback to plain search
            return {}

    def _build_chroma_where(self, filters: Dict[str, Any]) -> Optional[Dict]:
        """
        Converts extracted filters into ChromaDB where clause format.

        ChromaDB where clause format:
          {"subject": {"$eq": "Biology"}}
          {"$and": [{"subject": {"$eq": "Biology"}}, {"grade": {"$eq": "Class 10"}}]}
        """
        if not filters:
            return None

        conditions = [
            {key: {"$eq": value}}
            for key, value in filters.items()
        ]

        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}

        return None

    def retrieve(self, query: RAGQuery) -> List[str]:
        """
        Step 1: Extract filters from query, then search ChromaDB with filters.
        Falls back to plain semantic search if no filters found.
        """
        try:
            # Extract structured filters from the query
            filters = self._extract_filters(query.query_text)

            # Build ChromaDB where clause
            where_clause = self._build_chroma_where(filters)

            collection = self._chroma_client.get_or_create_collection(
                name=self._collection_name
            )

            # Search with filters if available, plain search if not
            if where_clause:
                results = collection.query(
                    query_texts=[query.query_text],
                    n_results=query.top_k,
                    where=where_clause
                )
            else:
                results = collection.query(
                    query_texts=[query.query_text],
                    n_results=query.top_k
                )

            return results["documents"][0] if results["documents"] else []

        except Exception as e:
            raise RuntimeError(f"[SelfQueryRAG] retrieve() failed: {str(e)}")

    def augment(self, query: RAGQuery, chunks: List[str]) -> str:
        """
        Step 2: Build prompt with filtered chunks.
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
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024
            )
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"[SelfQueryRAG] generate() failed: {str(e)}")