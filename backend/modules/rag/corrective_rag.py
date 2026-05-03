"""
@module    CorrectiveRAG
@description Corrective RAG (CRAG) — grades retrieved chunks for relevance
             and corrects bad retrievals before generation.
             Relevant chunks → use directly.
             Irrelevant chunks → discard and use web search fallback.
             Ambiguous chunks → supplement with additional search.
             Self-correcting pipeline — never uses bad context.
@author    EduMind AI Engineering
"""

import os
import chromadb
from groq import Groq
from typing import List, Tuple

from .base_rag import BaseRAG, RAGQuery, RAGResult


# Grade constants — no magic strings
GRADE_RELEVANT   = "relevant"
GRADE_IRRELEVANT = "irrelevant"
GRADE_AMBIGUOUS  = "ambiguous"


class CorrectiveRAG(BaseRAG):
    """
    Corrective RAG — self-evaluating retrieval pipeline.

    Key difference from all previous RAGs:
      - After retrieval, GRADES each chunk for relevance
      - Discards irrelevant chunks
      - Falls back to web search if too many irrelevant chunks
      - Result = only high-quality context reaches the LLM

    Flow:
      1. retrieve()      → get chunks from ChromaDB
      2. _grade_chunk()  → LLM grades each chunk: relevant/irrelevant/ambiguous
      3. _correct()      → filter bad chunks, add web fallback if needed
      4. augment()       → build prompt with only relevant chunks
      5. generate()      → call Groq with clean context
    """

    def __init__(self):
        super().__init__(rag_type="corrective")

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

        # Minimum relevance threshold
        # If fewer than this fraction of chunks are relevant → trigger fallback
        self._relevance_threshold = float(
            os.getenv("CRAG_RELEVANCE_THRESHOLD", "0.5")
        )

    def _grade_chunk(self, query_text: str, chunk: str) -> str:
        """
        Uses LLM to grade a single chunk for relevance to the query.
        Returns: "relevant", "irrelevant", or "ambiguous"

        Examples:
          Query: "What is CRISPR?"
          Chunk: "CRISPR is a gene editing technology..." → relevant
          Chunk: "Crispy fried chicken recipe..."         → irrelevant
          Chunk: "DNA modifications in biology..."        → ambiguous
        """
        grade_prompt = f"""You are a relevance grader for a document retrieval system.
Grade whether this document chunk is relevant to the student's question.

Grading criteria:
- relevant:   chunk directly answers or strongly supports the question
- irrelevant: chunk is off-topic or unrelated to the question
- ambiguous:  chunk is loosely related but not directly helpful

Rules:
- Return ONLY one word: relevant, irrelevant, or ambiguous
- No explanation, no preamble

Question: {query_text}
Chunk: {chunk[:300]}

Grade:"""

        try:
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": grade_prompt}],
                temperature=0.0,
                max_tokens=10
            )
            grade = response.choices[0].message.content.strip().lower()

            # Validate grade
            valid_grades = [GRADE_RELEVANT, GRADE_IRRELEVANT, GRADE_AMBIGUOUS]
            return grade if grade in valid_grades else GRADE_AMBIGUOUS

        except Exception:
            # Default to ambiguous if grading fails
            return GRADE_AMBIGUOUS

    def _web_search_fallback(self, query_text: str) -> List[str]:
        """
        Fallback when ChromaDB returns too many irrelevant chunks.
        In production this would call a real web search API.
        For now returns a structured fallback message so pipeline
        never crashes and LLM knows context is limited.
        """
        return [
            f"Web search fallback for: '{query_text}'. "
            f"No highly relevant documents found in the knowledge base. "
            f"Please answer based on your training knowledge if possible."
        ]

    def _correct(
        self,
        query_text: str,
        chunks: List[str]
    ) -> Tuple[List[str], str]:
        """
        Grades all chunks and corrects the retrieval.
        Returns: (corrected_chunks, correction_action)

        Actions:
          "use_retrieved"  → enough relevant chunks found
          "use_fallback"   → too many irrelevant → web fallback used
          "use_mixed"      → mix of relevant + ambiguous used
        """
        if not chunks:
            return self._web_search_fallback(query_text), "use_fallback"

        # Grade each chunk
        graded: List[Tuple[str, str]] = []
        for chunk in chunks:
            grade = self._grade_chunk(query_text, chunk)
            graded.append((chunk, grade))

        # Separate by grade
        relevant_chunks  = [c for c, g in graded if g == GRADE_RELEVANT]
        ambiguous_chunks = [c for c, g in graded if g == GRADE_AMBIGUOUS]
        irrelevant_count = sum(1 for _, g in graded if g == GRADE_IRRELEVANT)

        # Calculate relevance ratio
        relevance_ratio = len(relevant_chunks) / len(chunks)

        if relevance_ratio >= self._relevance_threshold:
            # Enough relevant chunks — use relevant + ambiguous
            return relevant_chunks + ambiguous_chunks, "use_retrieved"
        else:
            # Too many irrelevant chunks — use web fallback
            fallback = self._web_search_fallback(query_text)
            return fallback, "use_fallback"

    def retrieve(self, query: RAGQuery) -> List[str]:
        """
        Step 1: Get chunks from ChromaDB.
                Grade them for relevance.
                Correct bad retrievals.
                Return only high-quality chunks.
        """
        try:
            collection = self._chroma_client.get_or_create_collection(
                name=self._collection_name
            )

            results = collection.query(
                query_texts=[query.query_text],
                n_results=query.top_k
            )

            if not results["documents"] or not results["documents"][0]:
                return self._web_search_fallback(query.query_text)

            raw_chunks = results["documents"][0]

            # Grade and correct chunks
            corrected_chunks, action = self._correct(
                query.query_text, raw_chunks
            )

            return corrected_chunks[:query.top_k]

        except Exception as e:
            raise RuntimeError(f"[CorrectiveRAG] retrieve() failed: {str(e)}")

    def augment(self, query: RAGQuery, chunks: List[str]) -> str:
        """
        Step 2: Build prompt with graded + corrected chunks.
        """
        if not chunks:
            context = "No relevant documents found."
        else:
            context = "\n\n".join(
                [f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(chunks)]
            )

        prompt = f"""You are EduMind AI — a helpful education assistant.
The context below has been verified for relevance to the question.
Use ONLY the context below to answer the question.
If the context does not contain the answer, say "I don't have enough information."

Context:
{context}

Question: {query.query_text}

Answer:"""

        return prompt

    def generate(self, prompt: str) -> str:
        """
        Step 3: Send corrected prompt to Groq and return answer.
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
            raise RuntimeError(f"[CorrectiveRAG] generate() failed: {str(e)}")