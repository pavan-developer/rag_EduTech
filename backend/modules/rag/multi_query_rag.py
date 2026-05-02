"""
@module    MultiQueryRAG
@description Multi-Query RAG — generates multiple rephrased versions of
             the original query and searches with all of them.
             Different phrasings find different relevant chunks.
             Combined results = broader, more complete retrieval coverage.
@author    EduMind AI Engineering
"""

import os
import chromadb
from groq import Groq
from typing import List

from .base_rag import BaseRAG, RAGQuery, RAGResult


class MultiQueryRAG(BaseRAG):
    """
    Multi-Query RAG — multi-perspective retrieval pipeline.

    Key difference from NaiveRAG:
      - Generates N rephrased versions of the original query
      - Searches ChromaDB with EACH version separately
      - Combines and deduplicates all results
      - Result = much broader retrieval coverage

    Flow:
      1. _generate_queries() → LLM generates N query variations
      2. retrieve()          → search ChromaDB with each query
      3. augment()           → build prompt with combined chunks
      4. generate()          → call Groq, get comprehensive answer
    """

    def __init__(self, num_queries: int = 3):
        super().__init__(rag_type="multi_query")

        # Number of query variations to generate
        self._num_queries = int(os.getenv("MULTI_QUERY_COUNT", str(num_queries)))

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

    def _generate_queries(self, query_text: str) -> List[str]:
        """
        Uses LLM to generate N different versions of the original query.
        Each version approaches the same question from a different angle.

        Example:
          Original: "How do plants make food?"
          Generated:
            1. "What is the process of photosynthesis?"
            2. "How do plants convert sunlight to energy?"
            3. "What happens inside chloroplasts during food production?"
        """
        generate_prompt = f"""You are a query expansion expert for an education search system.
Generate {self._num_queries} different ways to ask the same question.
Each version should use different words but mean the same thing.
This helps find more relevant documents in a search system.

Rules:
- Return ONLY the questions, one per line
- No numbering, no bullets, no explanation
- Each question on its own line
- Do not repeat the original question

Original question: {query_text}
Rephrased versions:"""

        try:
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": generate_prompt}],
                temperature=0.7,
                max_tokens=256
            )
            raw = response.choices[0].message.content.strip()

            # Parse one query per line
            queries = [
                line.strip()
                for line in raw.split("\n")
                if line.strip() and len(line.strip()) > 10
            ]

            # Always include original query
            all_queries = [query_text] + queries[:self._num_queries]
            return all_queries

        except Exception:
            # Fallback — just use original query
            return [query_text]

    def retrieve(self, query: RAGQuery) -> List[str]:
        """
        Step 1: Generate multiple query versions.
                Search ChromaDB with each version.
                Combine and deduplicate all results.
        """
        try:
            collection = self._chroma_client.get_or_create_collection(
                name=self._collection_name
            )

            # Generate multiple query versions
            all_queries = self._generate_queries(query.query_text)

            # Search with each query and collect all chunks
            all_chunks: List[str] = []

            for q in all_queries:
                try:
                    results = collection.query(
                        query_texts=[q],
                        n_results=query.top_k
                    )
                    if results["documents"] and results["documents"][0]:
                        all_chunks.extend(results["documents"][0])
                except Exception:
                    continue

            # Deduplicate while preserving order
            unique_chunks = list(dict.fromkeys(all_chunks))

            return unique_chunks[:query.top_k]

        except Exception as e:
            raise RuntimeError(f"[MultiQueryRAG] retrieve() failed: {str(e)}")

    def augment(self, query: RAGQuery, chunks: List[str]) -> str:
        """
        Step 2: Build prompt with multi-query retrieved chunks.
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
            raise RuntimeError(f"[MultiQueryRAG] generate() failed: {str(e)}")