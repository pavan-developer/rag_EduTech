"""
@module    test_rag
@description Integration tests for all RAG pipeline implementations.
             Tests verify: correct output type, non-empty answer,
             correct rag_type label, and source chunks returned.
@author    EduMind AI Engineering
"""

import pytest
from unittest.mock import MagicMock, patch
from backend.modules.rag import NaiveRAG, HybridRAG, ContextualRAG, RAGQuery, RAGResult

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

#==============================================================

class TestHybridRAG:
    """Tests for HybridRAG pipeline."""

    @patch("backend.modules.rag.hybrid_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.hybrid_rag.Groq")
    def test_hybrid_rag_returns_rag_result(self, mock_groq, mock_chroma):
        """HybridRAG.run() must return a RAGResult object."""

        # Mock ChromaDB response
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[
                "Newton's second law states Force equals mass times acceleration.",
                "F=ma is the formula for Newton's second law.",
                "Mass and acceleration determine the net force on an object."
            ]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Groq response
        mock_choice = MagicMock()
        mock_choice.message.content = "Newton's second law: F = ma."
        mock_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        # Run the pipeline
        rag = HybridRAG()
        query = RAGQuery(query_text="What is Newton's second law?", top_k=3)
        result = rag.run(query)

        # Assertions
        assert isinstance(result, RAGResult)
        assert result.rag_type == "hybrid"
        assert len(result.answer) > 0
        assert len(result.source_chunks) > 0

    @patch("backend.modules.rag.hybrid_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.hybrid_rag.Groq")
    def test_hybrid_rag_handles_empty_chunks(self, mock_groq, mock_chroma):
        """HybridRAG must handle empty ChromaDB results gracefully."""

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

        rag = HybridRAG()
        query = RAGQuery(query_text="Unknown topic", top_k=5)
        result = rag.run(query)

        assert isinstance(result, RAGResult)
        assert result.rag_type == "hybrid"
        assert result.source_chunks == []

    @patch("backend.modules.rag.hybrid_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.hybrid_rag.Groq")
    def test_hybrid_rag_deduplicates_chunks(self, mock_groq, mock_chroma):
        """HybridRAG must not return duplicate chunks."""

        # Mock ChromaDB with duplicate-prone chunks
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[
                "Force equals mass times acceleration.",
                "Force equals mass times acceleration.",
                "Newton discovered this law in 1687."
            ]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_choice = MagicMock()
        mock_choice.message.content = "F = ma"
        mock_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        rag = HybridRAG()
        query = RAGQuery(query_text="Newton's law formula", top_k=3)
        result = rag.run(query)

        # No duplicates in source chunks
        assert len(result.source_chunks) == len(set(result.source_chunks))



#================================================================
class TestContextualRAG:
    """Tests for ContextualRAG pipeline."""

    @patch("backend.modules.rag.contextual_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.contextual_rag.Groq")
    def test_contextual_rag_returns_rag_result(self, mock_groq, mock_chroma):
        """ContextualRAG.run() must return a RAGResult object."""

        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Photosynthesis converts sunlight into glucose."]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Groq — called twice: once for rewrite, once for answer
        mock_choice = MagicMock()
        mock_choice.message.content = "Photosynthesis is the process of converting light."
        mock_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        rag = ContextualRAG()
        query = RAGQuery(query_text="What is photosynthesis?", top_k=1)
        result = rag.run(query)

        assert isinstance(result, RAGResult)
        assert result.rag_type == "contextual"
        assert len(result.answer) > 0

    @patch("backend.modules.rag.contextual_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.contextual_rag.Groq")
    def test_contextual_rag_saves_history(self, mock_groq, mock_chroma):
        """After run(), history must contain 2 turns (user + assistant)."""

        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Photosynthesis converts sunlight into glucose."]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Groq
        mock_choice = MagicMock()
        mock_choice.message.content = "Photosynthesis converts sunlight into energy."
        mock_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        rag = ContextualRAG()
        query = RAGQuery(query_text="What is photosynthesis?", top_k=1)
        rag.run(query)

        # History must have 2 entries: user turn + assistant turn
        assert len(rag._history) == 2
        assert rag._history[0]["role"] == "user"
        assert rag._history[1]["role"] == "assistant"

    @patch("backend.modules.rag.contextual_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.contextual_rag.Groq")
    def test_contextual_rag_clear_history(self, mock_groq, mock_chroma):
        """clear_history() must wipe all conversation turns."""

        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Photosynthesis converts sunlight into glucose."]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Groq
        mock_choice = MagicMock()
        mock_choice.message.content = "Some answer."
        mock_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        rag = ContextualRAG()
        query = RAGQuery(query_text="What is photosynthesis?", top_k=1)

        # Run once to build history
        rag.run(query)
        assert len(rag._history) == 2

        # Clear history
        rag.clear_history()
        assert len(rag._history) == 0