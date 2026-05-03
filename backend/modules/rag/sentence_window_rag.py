"""
@module    SentenceWindowRAG
@description Sentence Window RAG — retrieves small sentence-level chunks
             but expands to surrounding sentence window for full context.
             Small sentences = precise retrieval.
             Window expansion = rich context for LLM.
             Similar to ParentDocumentRAG but uses sentence boundaries
             instead of arbitrary chunk sizes.
@author    EduMind AI Engineering
"""

import os
import re
import chromadb
from groq import Groq
from typing import List

from .base_rag import BaseRAG, RAGQuery, RAGResult


class SentenceWindowRAG(BaseRAG):
    """
    Sentence Window RAG — sentence-level retrieval with window expansion.

    Key difference from ParentDocumentRAG:
      - ParentDocumentRAG: splits by character count
      - SentenceWindowRAG: splits by sentence boundaries (. ! ?)
      - Retrieves single sentences for precise matching
      - Expands to N sentences before + after for context

    Flow:
      1. retrieve()        → find matching sentences from ChromaDB
      2. _expand_window()  → expand each sentence to surrounding window
      3. augment()         → build prompt with expanded windows
      4. generate()        → call Groq with sentence-context
    """

    def __init__(self, window_size: int = 2):
        super().__init__(rag_type="sentence_window")

        # How many sentences before + after to include
        self._window_size = int(os.getenv("SENTENCE_WINDOW_SIZE", str(window_size)))

        self._chroma_client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_DB_PATH", "./chroma_db")
        )
        self._collection_name = os.getenv("CHROMA_COLLECTION", "edumind_docs")

        self._groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Splits text into individual sentences using regex.
        Handles common sentence endings: . ! ?
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _expand_window(
        self,
        matched_sentence: str,
        all_sentences: List[str]
    ) -> str:
        """
        Finds the matched sentence in the full sentence list
        and expands to window_size sentences before and after.

        Example (window_size=2):
          all_sentences = [s1, s2, s3, s4, s5, s6, s7]
          matched = s4
          window  = [s2, s3, s4, s5, s6]  ← 2 before + matched + 2 after
        """
        try:
            idx = all_sentences.index(matched_sentence)
        except ValueError:
            # Sentence not found exactly — return as-is
            return matched_sentence

        start = max(0, idx - self._window_size)
        end   = min(len(all_sentences), idx + self._window_size + 1)

        window_sentences = all_sentences[start:end]
        return " ".join(window_sentences)

    def retrieve(self, query: RAGQuery) -> List[str]:
        """
        Step 1: Search ChromaDB for matching sentences.
                Fetch full document text from metadata.
                Expand each match to surrounding sentence window.
        """
        try:
            collection = self._chroma_client.get_or_create_collection(
                name=self._collection_name
            )

            results = collection.query(
                query_texts=[query.query_text],
                n_results=query.top_k,
                include=["documents", "metadatas"]
            )

            if not results["documents"] or not results["documents"][0]:
                return []

            matched_sentences = results["documents"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else []

            expanded_chunks: List[str] = []

            for i, sentence in enumerate(matched_sentences):
                # Get full document text from metadata if available
                full_text = ""
                if i < len(metadatas) and metadatas[i]:
                    full_text = metadatas[i].get("full_text", "")

                if full_text:
                    # Split full text into sentences and expand window
                    all_sentences = self._split_into_sentences(full_text)
                    expanded = self._expand_window(sentence, all_sentences)
                else:
                    # No full text available — use sentence as-is
                    expanded = sentence

                if expanded not in expanded_chunks:
                    expanded_chunks.append(expanded)

            return expanded_chunks[:query.top_k]

        except Exception as e:
            raise RuntimeError(f"[SentenceWindowRAG] retrieve() failed: {str(e)}")

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
            raise RuntimeError(f"[SentenceWindowRAG] generate() failed: {str(e)}")