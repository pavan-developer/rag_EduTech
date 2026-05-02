"""
@module    ParentDocumentRAG
@description Parent Document RAG — stores small chunks for precise retrieval
             but returns larger parent chunks for full context when answering.
             Solves the context-loss problem of small chunk retrieval.
             Small chunks = better search. Large chunks = better answers.
@author    EduMind AI Engineering
"""

import os
import chromadb
from groq import Groq
from typing import List, Dict, Optional

from .base_rag import BaseRAG, RAGQuery, RAGResult


class ParentDocumentRAG(BaseRAG):
    """
    Parent Document RAG — two-level chunk storage pipeline.

    Key difference from NaiveRAG:
      - Documents stored at TWO levels:
          1. Small chunks (100-200 chars) → used for retrieval
          2. Large parent chunks (500-1000 chars) → used for answering
      - Retrieval finds small chunks → looks up their parent IDs
      - LLM receives full parent chunks → better context → better answers

    Flow:
      1. retrieve()  → find small chunks → fetch their parent chunks
      2. augment()   → build prompt with large parent chunks
      3. generate()  → call Groq with rich context, get answer
    """

    def __init__(self):
        super().__init__(rag_type="parent_document")

        # Two separate ChromaDB collections:
        # 1. child_collection  → small chunks for searching
        # 2. parent_collection → large chunks for answering
        self._chroma_client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_DB_PATH", "./chroma_db")
        )

        # Child collection — small chunks (retrieved by search)
        self._child_collection_name = os.getenv(
            "CHROMA_CHILD_COLLECTION", "edumind_child_chunks"
        )

        # Parent collection — large chunks (returned for answering)
        self._parent_collection_name = os.getenv(
            "CHROMA_PARENT_COLLECTION", "edumind_parent_chunks"
        )

        # Connect to Groq LLM
        self._groq_client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        self._model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

    def _get_parent_chunks(
        self,
        parent_ids: List[str],
        parent_collection
    ) -> List[str]:
        """
        Fetches large parent chunks from parent collection
        using the parent_ids found in child chunk metadata.
        """
        if not parent_ids:
            return []

        try:
            results = parent_collection.get(ids=parent_ids)
            return results["documents"] if results["documents"] else []
        except Exception:
            return []

    def retrieve(self, query: RAGQuery) -> List[str]:
        """
        Step 1: Search small child chunks for precise match.
                Then fetch their large parent chunks for full context.
        """
        try:
            # Get child collection (small chunks for search)
            child_collection = self._chroma_client.get_or_create_collection(
                name=self._child_collection_name
            )

            # Get parent collection (large chunks for answering)
            parent_collection = self._chroma_client.get_or_create_collection(
                name=self._parent_collection_name
            )

            # Search small child chunks
            child_results = child_collection.query(
                query_texts=[query.query_text],
                n_results=query.top_k,
                include=["documents", "metadatas"]
            )

            if not child_results["documents"] or not child_results["documents"][0]:
                return []

            # Extract parent_ids from child chunk metadata
            # Each child chunk stores its parent's ID in metadata
            parent_ids = []
            for metadata in child_results["metadatas"][0]:
                parent_id = metadata.get("parent_id")
                if parent_id and parent_id not in parent_ids:
                    parent_ids.append(parent_id)

            # Fetch large parent chunks using parent_ids
            parent_chunks = self._get_parent_chunks(parent_ids, parent_collection)

            # Fall back to child chunks if no parent chunks found
            if not parent_chunks:
                return child_results["documents"][0]

            return parent_chunks

        except Exception as e:
            raise RuntimeError(f"[ParentDocumentRAG] retrieve() failed: {str(e)}")

    def augment(self, query: RAGQuery, chunks: List[str]) -> str:
        """
        Step 2: Build prompt with large parent chunks.
        More context = better answers from LLM.
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
            raise RuntimeError(f"[ParentDocumentRAG] generate() failed: {str(e)}")