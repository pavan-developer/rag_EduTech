"""
@module    NaiveRAG
@description Simplest RAG implementation — direct vector search + LLM generation.
             Inherits BaseRAG and implements all 3 abstract methods.
             This is the foundation pattern all other RAG types build upon.
@author    EduMind AI Engineering
"""

import os
import chromadb
from groq import Groq
from typing import List

from .base_rag import BaseRAG, RAGQuery, RAGResult


class NaiveRAG(BaseRAG):
    """
    Naive RAG — simplest possible RAG pipeline.

    Flow:
      1. retrieve()  → semantic search in ChromaDB
      2. augment()   → combine query + chunks into prompt
      3. generate()  → send prompt to Groq Llama-3, get answer
    """

    def __init__(self):
        super().__init__(rag_type="naive")

        # Connect to ChromaDB (local, free, no server needed)
        self._chroma_client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_DB_PATH", "./chroma_db")
        )
        self._collection_name = os.getenv("CHROMA_COLLECTION", "edumind_docs")

        # Connect to Groq LLM (free tier — 14k requests/day)
        self._groq_client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        self._model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

    def retrieve(self, query: RAGQuery) -> List[str]:
        """
        Step 1: Search ChromaDB for top-k most relevant document chunks.
        Returns a list of text strings — the raw chunks.
        """
        try:
            collection = self._chroma_client.get_or_create_collection(
                name=self._collection_name
            )

            results = collection.query(
                query_texts=[query.query_text],
                n_results=query.top_k
            )

            # Extract just the text from results
            chunks = results["documents"][0] if results["documents"] else []
            return chunks

        except Exception as e:
            raise RuntimeError(f"[NaiveRAG] retrieve() failed: {str(e)}")

    def augment(self, query: RAGQuery, chunks: List[str]) -> str:
        """
        Step 2: Combine the user query + retrieved chunks into one prompt.
        This is what gets sent to the LLM.
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
        Step 3: Send the prompt to Groq Llama-3 and return the answer.
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
            raise RuntimeError(f"[NaiveRAG] generate() failed: {str(e)}")