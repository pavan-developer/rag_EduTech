"""
@module    ContextualRAG
@description Contextual RAG — maintains conversation history to resolve
             ambiguous follow-up questions. Rewrites the current query
             by combining it with chat history before retrieval.
             Example: "How does it work?" + history → "How does photosynthesis work?"
@author    EduMind AI Engineering
"""

import os
import chromadb
from groq import Groq
from typing import List, Dict

from .base_rag import BaseRAG, RAGQuery, RAGResult


# Represents a single turn in conversation history
ConversationTurn = Dict[str, str]  # {"role": "user/assistant", "content": "..."}


class ContextualRAG(BaseRAG):
    """
    Contextual RAG — conversation-aware retrieval pipeline.

    Key difference from NaiveRAG:
      - Maintains a conversation history (list of turns)
      - Before retrieval, rewrites the query using history
      - Resolves pronouns like "it", "this", "that" to actual topics
      - Result = more accurate retrieval for follow-up questions

    Flow:
      1. rewrite_query() → use LLM to combine history + current query
      2. retrieve()      → search ChromaDB with rewritten query
      3. augment()       → build prompt with history + chunks
      4. generate()      → call Groq, get answer, save to history
    """

    def __init__(self):
        super().__init__(rag_type="contextual")

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

        # Conversation history — list of {"role": ..., "content": ...}
        self._history: List[ConversationTurn] = []

        # Max history turns to keep (prevents context window overflow)
        self._max_history = int(os.getenv("MAX_HISTORY_TURNS", "6"))

    def _rewrite_query(self, query_text: str) -> str:
        """
        Uses LLM to rewrite the current query using conversation history.
        Resolves pronouns and ambiguous references.

        Example:
          History: "What is photosynthesis? → Photosynthesis is..."
          Query:   "How does it happen?"
          Rewrite: "How does photosynthesis happen?"
        """
        if not self._history:
            # No history yet — return query as-is
            return query_text

        # Build history summary for the rewrite prompt
        history_text = "\n".join([
            f"{turn['role'].upper()}: {turn['content']}"
            for turn in self._history[-self._max_history:]
        ])

        rewrite_prompt = f"""Given this conversation history:
{history_text}

Rewrite this follow-up question as a standalone question that includes all necessary context.
If the question is already standalone, return it unchanged.
Return ONLY the rewritten question — no explanation, no preamble.

Follow-up question: {query_text}
Standalone question:"""

        try:
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": rewrite_prompt}],
                temperature=0.0,
                max_tokens=128
            )
            return response.choices[0].message.content.strip()
        except Exception:
            # If rewrite fails, fall back to original query
            return query_text

    def retrieve(self, query: RAGQuery) -> List[str]:
        """
        Step 1: Rewrite query with history context, then search ChromaDB.
        """
        try:
            # Rewrite the query using conversation history
            rewritten_query = self._rewrite_query(query.query_text)

            collection = self._chroma_client.get_or_create_collection(
                name=self._collection_name
            )
            results = collection.query(
                query_texts=[rewritten_query],
                n_results=query.top_k
            )
            return results["documents"][0] if results["documents"] else []

        except Exception as e:
            raise RuntimeError(f"[ContextualRAG] retrieve() failed: {str(e)}")

    def augment(self, query: RAGQuery, chunks: List[str]) -> str:
        """
        Step 2: Build prompt with conversation history + retrieved chunks.
        History gives the LLM full conversation context.
        """
        if not chunks:
            context = "No relevant documents found."
        else:
            context = "\n\n".join(
                [f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(chunks)]
            )

        # Include last N turns of history in prompt
        history_text = ""
        if self._history:
            history_text = "\n\nConversation History:\n" + "\n".join([
                f"{turn['role'].upper()}: {turn['content']}"
                for turn in self._history[-self._max_history:]
            ])

        prompt = f"""You are EduMind AI — a helpful education assistant.
Use ONLY the context below to answer the question.
If the context does not contain the answer, say "I don't have enough information."{history_text}

Context:
{context}

Current Question: {query.query_text}

Answer:"""

        return prompt

    def generate(self, prompt: str) -> str:
        """
        Step 3: Send prompt to Groq, get answer, save to history.
        """
        try:
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024
            )
            answer = response.choices[0].message.content
            return answer

        except Exception as e:
            raise RuntimeError(f"[ContextualRAG] generate() failed: {str(e)}")

    def run(self, query: RAGQuery) -> RAGResult:
        """
        Override run() to save conversation turns to history after each call.
        """
        chunks = self.retrieve(query)
        prompt = self.augment(query, chunks)
        answer = self.generate(prompt)

        # Save this turn to history for future queries
        self._history.append({"role": "user", "content": query.query_text})
        self._history.append({"role": "assistant", "content": answer})

        # Trim history if it exceeds max turns
        if len(self._history) > self._max_history * 2:
            self._history = self._history[-self._max_history * 2:]

        return RAGResult(
            answer=answer,
            source_chunks=chunks,
            rag_type=self.rag_type,
            metadata={
                "query": query.query_text,
                "top_k": query.top_k,
                "history_turns": len(self._history) // 2
            }
        )

    def clear_history(self) -> None:
        """Clear conversation history — call this when a new session starts."""
        self._history = []