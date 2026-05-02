"""
@module    test_rag
@description Integration tests for all RAG pipeline implementations.
             Tests verify: correct output type, non-empty answer,
             correct rag_type label, and source chunks returned.
@author    EduMind AI Engineering
"""

import pytest
from unittest.mock import MagicMock, patch
from backend.modules.rag import NaiveRAG, RAGQuery, RAGResult


class TestNaiveRAG:
    """Tests for NaiveRAG pipeline."""

    @patch("backend.modules.rag.naive_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.naive_rag.Groq")
    def test_naive_rag_returns_rag_result(self, mock_groq, mock_chroma):
        """NaiveRAG.run() must return a RAGResult object."""

        # Mock ChromaDB response
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Photosynthesis is the process of converting light into energy."]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Groq response
        mock_choice = MagicMock()
        mock_choice.message.content = "Photosynthesis converts sunlight into glucose."
        mock_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        # Run the pipeline
        rag = NaiveRAG()
        query = RAGQuery(query_text="What is photosynthesis?", top_k=1)
        result = rag.run(query)

        # Assertions
        assert isinstance(result, RAGResult)
        assert result.rag_type == "naive"
        assert len(result.answer) > 0
        assert len(result.source_chunks) > 0

    @patch("backend.modules.rag.naive_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.naive_rag.Groq")
    def test_naive_rag_handles_empty_chunks(self, mock_groq, mock_chroma):
        """NaiveRAG must handle empty ChromaDB results gracefully."""

        # Mock empty ChromaDB response
        mock_collection = MagicMock()
        mock_collection.query.return_value = {"documents": [[]]}
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Groq response
        mock_choice = MagicMock()
        mock_choice.message.content = "I don't have enough information."
        mock_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        rag = NaiveRAG()
        query = RAGQuery(query_text="Random unknown topic", top_k=5)
        result = rag.run(query)

        assert isinstance(result, RAGResult)
        assert result.rag_type == "naive"
        assert result.source_chunks == []

    @patch("backend.modules.rag.naive_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.naive_rag.Groq")
    def test_rag_query_default_top_k(self, mock_groq, mock_chroma):
        """RAGQuery default top_k must be 5."""
        query = RAGQuery(query_text="Test question")
        assert query.top_k == 5