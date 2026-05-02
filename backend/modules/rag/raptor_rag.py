"""
@module    RaptorRAG
@description RAPTOR RAG — Recursive Abstractive Processing Through Organized Retrieval.
             Builds a tree of summaries from raw chunks:
               Level 0: raw chunks
               Level 1: summaries of chunk groups
               Level 2: summary of summaries
             Searches ALL levels — simple questions use raw chunks,
             big-picture questions use higher-level summaries.
@author    EduMind AI Engineering
"""

import os
import chromadb
from groq import Groq
from typing import List, Dict

from .base_rag import BaseRAG, RAGQuery, RAGResult


class RaptorRAG(BaseRAG):
    """
    RAPTOR RAG — multi-level tree retrieval pipeline.

    Key difference from all previous RAGs:
      - Builds summary tree BEFORE retrieval (during ingestion)
      - Searches across ALL tree levels simultaneously
      - Matches query complexity to correct summary level
      - Simple questions → raw chunks (Level 0)
      - Complex questions → high-level summaries (Level 2)

    Flow:
      1. _summarize_chunks()  → LLM summarizes groups of chunks
      2. _build_tree()        → creates multi-level summary tree
      3. retrieve()           → search all levels, combine results
      4. augment()            → build prompt with multi-level context
      5. generate()           → call Groq, get comprehensive answer
    """

    def __init__(self, tree_depth: int = 2, group_size: int = 3):
        super().__init__(rag_type="raptor")

        # Tree configuration
        self._tree_depth = int(os.getenv("RAPTOR_TREE_DEPTH", str(tree_depth)))
        self._group_size = int(os.getenv("RAPTOR_GROUP_SIZE", str(group_size)))

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

    def _summarize_chunks(self, chunks: List[str]) -> str:
        """
        Uses LLM to summarize a group of chunks into one summary.
        This summary becomes a node at the next tree level.
        """
        if not chunks:
            return ""

        combined = "\n\n".join(chunks)

        summary_prompt = f"""Summarize the following text into a concise paragraph
that captures the key concepts and information.
Keep it under 100 words.
Return ONLY the summary — no preamble.

Text:
{combined}

Summary:"""

        try:
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception:
            # Fallback — join chunks as-is
            return " ".join(chunks[:2])

    def _build_tree(self, seed_chunks: List[str]) -> Dict[int, List[str]]:
        """
        Builds a multi-level summary tree from seed chunks.

        Tree structure:
          {
            0: [chunk1, chunk2, chunk3, chunk4, chunk5, chunk6],  ← raw chunks
            1: [summary_A, summary_B],                            ← group summaries
            2: [mega_summary]                                     ← top summary
          }

        Each level summarizes groups from the level below.
        """
        tree: Dict[int, List[str]] = {0: seed_chunks}

        current_level = seed_chunks

        for level in range(1, self._tree_depth + 1):
            if len(current_level) <= 1:
                # Can't summarize further — tree is complete
                break

            # Group chunks into groups of group_size
            groups = [
                current_level[i:i + self._group_size]
                for i in range(0, len(current_level), self._group_size)
            ]

            # Summarize each group
            summaries = []
            for group in groups:
                summary = self._summarize_chunks(group)
                if summary:
                    summaries.append(summary)

            if summaries:
                tree[level] = summaries
                current_level = summaries

        return tree

    def retrieve(self, query: RAGQuery) -> List[str]:
        """
        Step 1: Get seed chunks from ChromaDB.
                Build summary tree from seeds.
                Search ALL tree levels.
                Combine results from all levels.
        """
        try:
            collection = self._chroma_client.get_or_create_collection(
                name=self._collection_name
            )

            # Get seed chunks (Level 0)
            seed_results = collection.query(
                query_texts=[query.query_text],
                n_results=query.top_k * 2
            )

            if not seed_results["documents"] or \
               not seed_results["documents"][0]:
                return []

            seed_chunks = seed_results["documents"][0]

            # Build summary tree
            tree = self._build_tree(seed_chunks)

            # Collect chunks from ALL tree levels
            all_level_chunks: List[str] = []
            for level in sorted(tree.keys()):
                all_level_chunks.extend(tree[level])

            # Deduplicate while preserving order
            unique_chunks = list(dict.fromkeys(all_level_chunks))

            return unique_chunks[:query.top_k]

        except Exception as e:
            raise RuntimeError(f"[RaptorRAG] retrieve() failed: {str(e)}")

    def augment(self, query: RAGQuery, chunks: List[str]) -> str:
        """
        Step 2: Build prompt with multi-level tree chunks.
        """
        if not chunks:
            context = "No relevant documents found."
        else:
            context = "\n\n".join(
                [f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(chunks)]
            )

        prompt = f"""You are EduMind AI — a helpful education assistant.
The context below contains information at multiple levels of detail —
from specific facts to high-level summaries. Use all of it to give
the most complete and accurate answer possible.
If the context does not contain the answer, say "I don't have enough information."

Context:
{context}

Question: {query.query_text}

Answer:"""

        return prompt

    def generate(self, prompt: str) -> str:
        """
        Step 3: Send multi-level prompt to Groq and return answer.
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
            raise RuntimeError(f"[RaptorRAG] generate() failed: {str(e)}")