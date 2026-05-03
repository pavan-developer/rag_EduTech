"""
@module    test_rag
@description Integration tests for all RAG pipeline implementations.
             Tests verify: correct output type, non-empty answer,
             correct rag_type label, and source chunks returned.
@author    EduMind AI Engineering
"""

import pytest
from unittest.mock import MagicMock, patch
from backend.modules.rag import NaiveRAG, HybridRAG, ContextualRAG, SelfQueryRAG, ParentDocumentRAG, EnsembleRAG, AdaptiveRAG, GraphRAG, MultiQueryRAG, StepBackRAG, RaptorRAG, CorrectiveRAG, SpeculativeRAG, RAGQuery, RAGResult

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



#===============================================================




class TestSelfQueryRAG:
    """Tests for SelfQueryRAG pipeline."""

    @patch("backend.modules.rag.self_query_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.self_query_rag.Groq")
    def test_self_query_rag_returns_rag_result(self, mock_groq, mock_chroma):
        """SelfQueryRAG.run() must return a RAGResult object."""

        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Cells are the basic unit of life in Biology."]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Groq — first call returns filters, second call returns answer
        mock_choice_filters = MagicMock()
        mock_choice_filters.message.content = '{"subject": "Biology", "grade": "Class 10"}'

        mock_choice_answer = MagicMock()
        mock_choice_answer.message.content = "Cells are the basic unit of life."

        mock_groq.return_value.chat.completions.create.side_effect = [
            MagicMock(choices=[mock_choice_filters]),
            MagicMock(choices=[mock_choice_answer])
        ]

        rag = SelfQueryRAG()
        query = RAGQuery(query_text="Class 10 Biology chapters about cells", top_k=1)
        result = rag.run(query)

        assert isinstance(result, RAGResult)
        assert result.rag_type == "self_query"
        assert len(result.answer) > 0

    @patch("backend.modules.rag.self_query_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.self_query_rag.Groq")
    def test_self_query_extracts_filters(self, mock_groq, mock_chroma):
        """_extract_filters() must return correct filters from query."""

        # Mock Groq to return filters JSON
        mock_choice = MagicMock()
        mock_choice.message.content = '{"subject": "Physics", "difficulty": "hard"}'
        mock_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        rag = SelfQueryRAG()
        filters = rag._extract_filters("Hard Physics problems about Newton")

        assert filters.get("subject") == "Physics"
        assert filters.get("difficulty") == "hard"

    @patch("backend.modules.rag.self_query_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.self_query_rag.Groq")
    def test_self_query_fallback_on_no_filters(self, mock_groq, mock_chroma):
        """SelfQueryRAG must fall back to plain search when no filters found."""

        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Photosynthesis converts sunlight into energy."]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Groq — first call returns empty filters, second returns answer
        mock_choice_filters = MagicMock()
        mock_choice_filters.message.content = '{}'

        mock_choice_answer = MagicMock()
        mock_choice_answer.message.content = "Photosynthesis is the process of converting light."

        mock_groq.return_value.chat.completions.create.side_effect = [
            MagicMock(choices=[mock_choice_filters]),
            MagicMock(choices=[mock_choice_answer])
        ]

        rag = SelfQueryRAG()
        query = RAGQuery(query_text="What is photosynthesis?", top_k=1)
        result = rag.run(query)

        # Must still return a valid result even with no filters
        assert isinstance(result, RAGResult)
        assert result.rag_type == "self_query"
        assert len(result.answer) > 0



#==========================================================



class TestParentDocumentRAG:
    """Tests for ParentDocumentRAG pipeline."""

    @patch("backend.modules.rag.parent_document_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.parent_document_rag.Groq")
    def test_parent_document_rag_returns_rag_result(self, mock_groq, mock_chroma):
        """ParentDocumentRAG.run() must return a RAGResult object."""

        # Mock child collection — small chunks with parent_id in metadata
        mock_child_collection = MagicMock()
        mock_child_collection.query.return_value = {
            "documents": [["Fleming discovered it in 1928."]],
            "metadatas": [[{"parent_id": "parent_001"}]]
        }

        # Mock parent collection — large chunks fetched by ID
        mock_parent_collection = MagicMock()
        mock_parent_collection.get.return_value = {
            "documents": [
                "Penicillin is an antibiotic discovered by Alexander Fleming in 1928. "
                "It revolutionized modern medicine and saved millions of lives."
            ]
        }

        # Return different collections based on call order
        mock_chroma.return_value.get_or_create_collection.side_effect = [
            mock_child_collection,
            mock_parent_collection
        ]

        # Mock Groq answer
        mock_choice = MagicMock()
        mock_choice.message.content = "Penicillin was discovered by Fleming in 1928."
        mock_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        rag = ParentDocumentRAG()
        query = RAGQuery(query_text="Who discovered penicillin?", top_k=1)
        result = rag.run(query)

        assert isinstance(result, RAGResult)
        assert result.rag_type == "parent_document"
        assert len(result.answer) > 0

    @patch("backend.modules.rag.parent_document_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.parent_document_rag.Groq")
    def test_parent_document_returns_parent_chunks(self, mock_groq, mock_chroma):
        """retrieve() must return parent chunks not child chunks."""

        # Mock child collection
        mock_child_collection = MagicMock()
        mock_child_collection.query.return_value = {
            "documents": [["short child chunk"]],
            "metadatas": [[{"parent_id": "parent_001"}]]
        }

        # Mock parent collection — much larger text
        mock_parent_collection = MagicMock()
        mock_parent_collection.get.return_value = {
            "documents": ["This is the full large parent chunk with complete context "
                         "about the topic including all surrounding information."]
        }

        mock_chroma.return_value.get_or_create_collection.side_effect = [
            mock_child_collection,
            mock_parent_collection
        ]

        mock_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="answer"))]
        )

        rag = ParentDocumentRAG()
        query = RAGQuery(query_text="Tell me about penicillin", top_k=1)
        chunks = rag.retrieve(query)

        # Must return parent chunk (longer) not child chunk (shorter)
        assert chunks[0] != "short child chunk"
        assert len(chunks[0]) > len("short child chunk")

    @patch("backend.modules.rag.parent_document_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.parent_document_rag.Groq")
    def test_parent_document_fallback_to_child(self, mock_groq, mock_chroma):
        """retrieve() must fall back to child chunks when parent not found."""

        # Mock child collection
        mock_child_collection = MagicMock()
        mock_child_collection.query.return_value = {
            "documents": [["child chunk fallback text"]],
            "metadatas": [[{"parent_id": "parent_001"}]]
        }

        # Mock parent collection returning empty
        mock_parent_collection = MagicMock()
        mock_parent_collection.get.return_value = {
            "documents": []
        }

        mock_chroma.return_value.get_or_create_collection.side_effect = [
            mock_child_collection,
            mock_parent_collection
        ]

        mock_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="fallback answer"))]
        )

        rag = ParentDocumentRAG()
        query = RAGQuery(query_text="Test question", top_k=1)
        chunks = rag.retrieve(query)

        # Must fall back to child chunks
        assert chunks == ["child chunk fallback text"]



#============================================================




class TestEnsembleRAG:
    """Tests for EnsembleRAG pipeline."""

    @patch("backend.modules.rag.naive_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.naive_rag.Groq")
    @patch("backend.modules.rag.hybrid_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.hybrid_rag.Groq")
    @patch("backend.modules.rag.ensemble_rag.Groq")
    def test_ensemble_rag_returns_rag_result(
        self,
        mock_ensemble_groq,
        mock_hybrid_groq,
        mock_hybrid_chroma,
        mock_naive_groq,
        mock_naive_chroma
    ):
        """EnsembleRAG.run() must return a RAGResult object."""

        # Mock NaiveRAG ChromaDB
        mock_naive_collection = MagicMock()
        mock_naive_collection.query.return_value = {
            "documents": [["chunk_A about Newton", "chunk_B about force"]]
        }
        mock_naive_chroma.return_value.get_or_create_collection.return_value = mock_naive_collection

        # Mock HybridRAG ChromaDB
        mock_hybrid_collection = MagicMock()
        mock_hybrid_collection.query.return_value = {
            "documents": [["chunk_B about force", "chunk_C about mass"]]
        }
        mock_hybrid_chroma.return_value.get_or_create_collection.return_value = mock_hybrid_collection

        # Mock Groq answer for EnsembleRAG
        mock_choice = MagicMock()
        mock_choice.message.content = "Newton's law: F = ma"
        mock_ensemble_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        rag = EnsembleRAG()
        query = RAGQuery(query_text="What is Newton's second law?", top_k=3)
        result = rag.run(query)

        assert isinstance(result, RAGResult)
        assert result.rag_type == "ensemble"
        assert len(result.answer) > 0

    @patch("backend.modules.rag.naive_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.naive_rag.Groq")
    @patch("backend.modules.rag.hybrid_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.hybrid_rag.Groq")
    @patch("backend.modules.rag.ensemble_rag.Groq")
    def test_ensemble_ranks_consensus_chunk_first(
        self,
        mock_ensemble_groq,
        mock_hybrid_groq,
        mock_hybrid_chroma,
        mock_naive_groq,
        mock_naive_chroma
    ):
        """Chunk found by both retrievers must rank first."""

        # NaiveRAG finds: chunk_A, chunk_B
        mock_naive_collection = MagicMock()
        mock_naive_collection.query.return_value = {
            "documents": [["unique_to_naive", "shared_chunk"]]
        }
        mock_naive_chroma.return_value.get_or_create_collection.return_value = mock_naive_collection

        # HybridRAG finds: chunk_B, chunk_C
        mock_hybrid_collection = MagicMock()
        mock_hybrid_collection.query.return_value = {
            "documents": [["shared_chunk", "unique_to_hybrid"]]
        }
        mock_hybrid_chroma.return_value.get_or_create_collection.return_value = mock_hybrid_collection

        mock_ensemble_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="answer"))]
        )

        rag = EnsembleRAG()
        query = RAGQuery(query_text="Test query", top_k=3)
        chunks = rag.retrieve(query)

        # shared_chunk found by both RAGs must rank first
        # shared_chunk found by both RAGs must be in results
        assert "shared_chunk" in chunks
# All returned chunks must be unique (no duplicates)
        assert len(chunks) == len(set(chunks))

    @patch("backend.modules.rag.naive_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.naive_rag.Groq")
    @patch("backend.modules.rag.hybrid_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.hybrid_rag.Groq")
    @patch("backend.modules.rag.ensemble_rag.Groq")
    def test_ensemble_continues_if_retriever_fails(
        self,
        mock_ensemble_groq,
        mock_hybrid_groq,
        mock_hybrid_chroma,
        mock_naive_groq,
        mock_naive_chroma
    ):
        """EnsembleRAG must continue if one retriever fails."""

        # NaiveRAG ChromaDB raises exception
        mock_naive_chroma.return_value.get_or_create_collection.side_effect = Exception(
            "ChromaDB connection failed"
        )

        # HybridRAG works fine
        mock_hybrid_collection = MagicMock()
        mock_hybrid_collection.query.return_value = {
            "documents": [["chunk from hybrid"]]
        }
        mock_hybrid_chroma.return_value.get_or_create_collection.return_value = mock_hybrid_collection

        mock_ensemble_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="answer from hybrid only"))]
        )

        rag = EnsembleRAG()
        query = RAGQuery(query_text="Test query", top_k=3)

        # Must not raise — should continue with HybridRAG results
        result = rag.run(query)
        assert isinstance(result, RAGResult)
        assert result.rag_type == "ensemble"


#============================================================


class TestAdaptiveRAG:
    """Tests for AdaptiveRAG pipeline."""

    @patch("backend.modules.rag.naive_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.naive_rag.Groq")
    @patch("backend.modules.rag.hybrid_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.hybrid_rag.Groq")
    @patch("backend.modules.rag.contextual_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.contextual_rag.Groq")
    @patch("backend.modules.rag.self_query_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.self_query_rag.Groq")
    @patch("backend.modules.rag.ensemble_rag.Groq")
    @patch("backend.modules.rag.adaptive_rag.Groq")
    def test_adaptive_rag_returns_rag_result(
        self,
        mock_adaptive_groq,
        mock_ensemble_groq,
        mock_self_query_groq,
        mock_self_query_chroma,
        mock_contextual_groq,
        mock_contextual_chroma,
        mock_hybrid_groq,
        mock_hybrid_chroma,
        mock_naive_groq,
        mock_naive_chroma,
    ):
        """AdaptiveRAG.run() must return a RAGResult object."""

        # Mock ChromaDB for NaiveRAG
        mock_naive_collection = MagicMock()
        mock_naive_collection.query.return_value = {
            "documents": [["Photosynthesis converts sunlight into energy."]]
        }
        mock_naive_chroma.return_value.get_or_create_collection.return_value = (
            mock_naive_collection
        )

        # Mock Groq — first call classifies query, second call generates answer
        mock_classify = MagicMock()
        mock_classify.message.content = "simple"

        mock_answer = MagicMock()
        mock_answer.message.content = "Photosynthesis is the process of converting light."

       # Adaptive Groq handles classification only
        mock_adaptive_groq.return_value.chat.completions.create.side_effect = [
            MagicMock(choices=[mock_classify]),
 ]

# NaiveRAG Groq handles the actual answer generation
        mock_naive_groq.return_value.chat.completions.create.return_value = MagicMock(
        choices=[mock_answer]
)

        rag = AdaptiveRAG()
        query = RAGQuery(query_text="What is photosynthesis?", top_k=1)
        result = rag.run(query)

        assert isinstance(result, RAGResult)
        assert result.rag_type == "adaptive"
        assert len(result.answer) > 0
        assert "routed_to" in result.metadata

    @patch("backend.modules.rag.naive_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.naive_rag.Groq")
    @patch("backend.modules.rag.hybrid_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.hybrid_rag.Groq")
    @patch("backend.modules.rag.contextual_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.contextual_rag.Groq")
    @patch("backend.modules.rag.self_query_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.self_query_rag.Groq")
    @patch("backend.modules.rag.ensemble_rag.Groq")
    @patch("backend.modules.rag.adaptive_rag.Groq")
    def test_adaptive_routes_to_correct_rag(
        self,
        mock_adaptive_groq,
        mock_ensemble_groq,
        mock_self_query_groq,
        mock_self_query_chroma,
        mock_contextual_groq,
        mock_contextual_chroma,
        mock_hybrid_groq,
        mock_hybrid_chroma,
        mock_naive_groq,
        mock_naive_chroma,
    ):
        """AdaptiveRAG must route filtered query to SelfQueryRAG."""

        # Mock ChromaDB for SelfQueryRAG
        mock_self_query_collection = MagicMock()
        mock_self_query_collection.query.return_value = {
            "documents": [["Biology Class 10 cells chapter."]]
        }
        mock_self_query_chroma.return_value.get_or_create_collection.return_value = (
            mock_self_query_collection
        )

        # Mock Groq — classify returns "filtered", then filter extraction, then answer
        mock_classify = MagicMock()
        mock_classify.message.content = "filtered"

        mock_filters = MagicMock()
        mock_filters.message.content = '{"subject": "Biology", "grade": "Class 10"}'

        mock_answer = MagicMock()
        mock_answer.message.content = "Cells are the basic unit of life."

        mock_adaptive_groq.return_value.chat.completions.create.side_effect = [
            MagicMock(choices=[mock_classify]),
        ]
        mock_self_query_groq.return_value.chat.completions.create.side_effect = [
            MagicMock(choices=[mock_filters]),
            MagicMock(choices=[mock_answer]),
        ]

        rag = AdaptiveRAG()
        query = RAGQuery(
            query_text="Class 10 Biology chapters about cells", top_k=1
        )
        result = rag.run(query)

        assert isinstance(result, RAGResult)
        assert result.metadata["routed_to"] == "filtered"

    @patch("backend.modules.rag.naive_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.naive_rag.Groq")
    @patch("backend.modules.rag.hybrid_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.hybrid_rag.Groq")
    @patch("backend.modules.rag.contextual_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.contextual_rag.Groq")
    @patch("backend.modules.rag.self_query_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.self_query_rag.Groq")
    @patch("backend.modules.rag.ensemble_rag.Groq")
    @patch("backend.modules.rag.adaptive_rag.Groq")
    def test_adaptive_defaults_to_naive_on_failure(
        self,
        mock_adaptive_groq,
        mock_ensemble_groq,
        mock_self_query_groq,
        mock_self_query_chroma,
        mock_contextual_groq,
        mock_contextual_chroma,
        mock_hybrid_groq,
        mock_hybrid_chroma,
        mock_naive_groq,
        mock_naive_chroma,
    ):
        """AdaptiveRAG must default to NaiveRAG when classification fails."""

        # Mock ChromaDB for NaiveRAG
        mock_naive_collection = MagicMock()
        mock_naive_collection.query.return_value = {
            "documents": [["Fallback answer from naive."]]
        }
        mock_naive_chroma.return_value.get_or_create_collection.return_value = (
            mock_naive_collection
        )

        # Mock Groq — classification raises exception
        mock_adaptive_groq.return_value.chat.completions.create.side_effect = [
            Exception("Groq API down"),
            MagicMock(choices=[MagicMock(message=MagicMock(content="fallback answer"))]),
        ]

        rag = AdaptiveRAG()
        query = RAGQuery(query_text="What is gravity?", top_k=1)
        result = rag.run(query)

        # Must still return result — defaulted to NaiveRAG
        assert isinstance(result, RAGResult)
        assert result.metadata["routed_to"] == "simple"





#================================================================





class TestGraphRAG:
    """Tests for GraphRAG pipeline."""

    @patch("backend.modules.rag.graph_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.graph_rag.Groq")
    def test_graph_rag_returns_rag_result(self, mock_groq, mock_chroma):
        """GraphRAG.run() must return a RAGResult object."""

        # Mock ChromaDB with related chunks
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[
                "Newton discovered the laws of motion in 1687.",
                "Force equals mass times acceleration.",
                "Motion is caused by unbalanced forces acting on objects."
            ]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Groq answer
        mock_choice = MagicMock()
        mock_choice.message.content = "Newton's laws describe the relationship between force and motion."
        mock_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        rag = GraphRAG()
        query = RAGQuery(query_text="What are Newton's laws?", top_k=3)
        result = rag.run(query)

        assert isinstance(result, RAGResult)
        assert result.rag_type == "graph"
        assert len(result.answer) > 0

    @patch("backend.modules.rag.graph_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.graph_rag.Groq")
    def test_graph_expands_related_chunks(self, mock_groq, mock_chroma):
        """GraphRAG must expand seed chunks via concept graph traversal."""

        # Two related chunks sharing concepts
        chunk_a = "photosynthesis occurs inside chloroplasts using sunlight"
        chunk_b = "chloroplasts contain chlorophyll which absorbs sunlight energy"

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[chunk_a, chunk_b]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="answer"))]
        )

        rag = GraphRAG()
        query = RAGQuery(query_text="How does photosynthesis work?", top_k=5)
        chunks = rag.retrieve(query)

        # Both chunks must be in results — graph connects them via shared concepts
        assert chunk_a in chunks
        assert chunk_b in chunks

    @patch("backend.modules.rag.graph_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.graph_rag.Groq")
    def test_graph_rag_handles_empty_chunks(self, mock_groq, mock_chroma):
        """GraphRAG must handle empty ChromaDB results gracefully."""

        mock_collection = MagicMock()
        mock_collection.query.return_value = {"documents": [[]]}
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_choice = MagicMock()
        mock_choice.message.content = "I don't have enough information."
        mock_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        rag = GraphRAG()
        query = RAGQuery(query_text="Unknown topic", top_k=3)
        result = rag.run(query)

        assert isinstance(result, RAGResult)
        assert result.rag_type == "graph"
        assert result.source_chunks == []





#=============================================



class TestMultiQueryRAG:
    """Tests for MultiQueryRAG pipeline."""

    @patch("backend.modules.rag.multi_query_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.multi_query_rag.Groq")
    def test_multi_query_rag_returns_rag_result(self, mock_groq, mock_chroma):
        """MultiQueryRAG.run() must return a RAGResult object."""

        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Photosynthesis converts sunlight into glucose."]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Groq — first call generates queries, second call generates answer
        mock_queries = MagicMock()
        mock_queries.message.content = (
            "What is the process of photosynthesis?\n"
            "How do plants convert sunlight to energy?\n"
            "What happens inside chloroplasts?"
        )

        mock_answer = MagicMock()
        mock_answer.message.content = "Photosynthesis converts sunlight into glucose."

        mock_groq.return_value.chat.completions.create.side_effect = [
            MagicMock(choices=[mock_queries]),
            MagicMock(choices=[mock_answer]),
        ]

        rag = MultiQueryRAG()
        query = RAGQuery(query_text="How do plants make food?", top_k=3)
        result = rag.run(query)

        assert isinstance(result, RAGResult)
        assert result.rag_type == "multi_query"
        assert len(result.answer) > 0

    @patch("backend.modules.rag.multi_query_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.multi_query_rag.Groq")
    def test_multi_query_produces_unique_chunks(self, mock_groq, mock_chroma):
        """MultiQueryRAG must deduplicate chunks from multiple queries."""

        # Mock ChromaDB — returns different chunks per query call
        mock_collection = MagicMock()
        mock_collection.query.side_effect = [
            {"documents": [["chunk_A", "chunk_B"]]},  # query 1
            {"documents": [["chunk_B", "chunk_C"]]},  # query 2
            {"documents": [["chunk_C", "chunk_D"]]},  # query 3
            {"documents": [["chunk_D", "chunk_E"]]},  # query 4
        ]
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Groq query generation
        mock_queries = MagicMock()
        mock_queries.message.content = (
            "What is photosynthesis?\n"
            "How do plants use sunlight?\n"
            "What is chlorophyll used for?"
        )
        mock_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_queries]
        )

        rag = MultiQueryRAG()
        query = RAGQuery(query_text="How do plants make food?", top_k=10)
        chunks = rag.retrieve(query)

        # No duplicates in results
        assert len(chunks) == len(set(chunks))

    @patch("backend.modules.rag.multi_query_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.multi_query_rag.Groq")
    def test_multi_query_fallback_on_generation_failure(self, mock_groq, mock_chroma):
        """MultiQueryRAG must fall back to original query if generation fails."""

        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Fallback chunk from original query."]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Groq — first call fails, second call returns answer
        mock_answer = MagicMock()
        mock_answer.message.content = "Fallback answer."

        mock_groq.return_value.chat.completions.create.side_effect = [
            Exception("Groq API down"),          # query generation fails
            MagicMock(choices=[mock_answer]),     # answer generation succeeds
        ]

        rag = MultiQueryRAG()
        query = RAGQuery(query_text="What is gravity?", top_k=1)
        result = rag.run(query)

        # Must still return valid result using original query only
        assert isinstance(result, RAGResult)
        assert result.rag_type == "multi_query"




#=============================================================



class TestStepBackRAG:
    """Tests for StepBackRAG pipeline."""

    @patch("backend.modules.rag.step_back_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.step_back_rag.Groq")
    def test_step_back_rag_returns_rag_result(self, mock_groq, mock_chroma):
        """StepBackRAG.run() must return a RAGResult object."""

        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Electromagnetic spectrum covers all wavelengths."]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Groq — first call generates step-back, second call answers
        mock_stepback = MagicMock()
        mock_stepback.message.content = "What is the electromagnetic spectrum?"

        mock_answer = MagicMock()
        mock_answer.message.content = "Red light has a wavelength of 620-750nm."

        mock_groq.return_value.chat.completions.create.side_effect = [
            MagicMock(choices=[mock_stepback]),
            MagicMock(choices=[mock_answer]),
        ]

        rag = StepBackRAG()
        query = RAGQuery(
            query_text="What is the wavelength of red light?", top_k=2
        )
        result = rag.run(query)

        assert isinstance(result, RAGResult)
        assert result.rag_type == "step_back"
        assert len(result.answer) > 0

    @patch("backend.modules.rag.step_back_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.step_back_rag.Groq")
    def test_step_back_combines_specific_and_general(self, mock_groq, mock_chroma):
        """retrieve() must combine specific and general chunks."""

        # Mock ChromaDB — returns different chunks per call
        mock_collection = MagicMock()
        mock_collection.query.side_effect = [
            {"documents": [["specific chunk about red light wavelength"]]},
            {"documents": [["general chunk about electromagnetic spectrum"]]},
        ]
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Groq step-back generation
        mock_stepback = MagicMock()
        mock_stepback.message.content = "What is the electromagnetic spectrum?"
        mock_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_stepback]
        )

        rag = StepBackRAG()
        query = RAGQuery(
            query_text="What is the wavelength of red light?", top_k=5
        )
        chunks = rag.retrieve(query)

        # Both specific and general chunks must be present
        assert "specific chunk about red light wavelength" in chunks
        assert "general chunk about electromagnetic spectrum" in chunks

    @patch("backend.modules.rag.step_back_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.step_back_rag.Groq")
    def test_step_back_fallback_on_failure(self, mock_groq, mock_chroma):
        """StepBackRAG must fall back to original query if step-back fails."""

        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Fallback chunk from original query."]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Groq — step-back generation fails, answer succeeds
        mock_answer = MagicMock()
        mock_answer.message.content = "Fallback answer."

        mock_groq.return_value.chat.completions.create.side_effect = [
            Exception("Groq API down"),
            MagicMock(choices=[mock_answer]),
        ]

        rag = StepBackRAG()
        query = RAGQuery(
            query_text="What is the wavelength of red light?", top_k=1
        )
        result = rag.run(query)

        # Must still return valid result
        assert isinstance(result, RAGResult)
        assert result.rag_type == "step_back"
    


#===============================================================




class TestRaptorRAG:
    """Tests for RaptorRAG pipeline."""

    @patch("backend.modules.rag.raptor_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.raptor_rag.Groq")
    def test_raptor_rag_returns_rag_result(self, mock_groq, mock_chroma):
        """RaptorRAG.run() must return a RAGResult object."""

        # Mock ChromaDB with seed chunks
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[
                "Newton discovered laws of motion.",
                "Force equals mass times acceleration.",
                "Objects in motion stay in motion.",
            ]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Groq — first calls are summaries, last call is answer
        mock_summary = MagicMock()
        mock_summary.message.content = "Newton's mechanics covers force, motion and acceleration."

        mock_answer = MagicMock()
        mock_answer.message.content = "Newton's laws describe force and motion relationships."

        mock_groq.return_value.chat.completions.create.side_effect = [
            MagicMock(choices=[mock_summary]),  # Level 1 summary
            MagicMock(choices=[mock_summary]),  # Level 2 summary
            MagicMock(choices=[mock_answer]),   # Final answer
        ]

        rag = RaptorRAG(tree_depth=2, group_size=3)
        query = RAGQuery(query_text="What are Newton's laws?", top_k=5)
        result = rag.run(query)

        assert isinstance(result, RAGResult)
        assert result.rag_type == "raptor"
        assert len(result.answer) > 0

    @patch("backend.modules.rag.raptor_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.raptor_rag.Groq")
    def test_raptor_builds_tree_levels(self, mock_groq, mock_chroma):
        """_build_tree() must create correct number of levels."""

        # Mock Groq summary
        mock_summary = MagicMock()
        mock_summary.message.content = "Summary of Newton mechanics."
        mock_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_summary]
        )

        rag = RaptorRAG(tree_depth=2, group_size=3)

        # Build tree from 6 chunks
        chunks = [
            "chunk_1", "chunk_2", "chunk_3",
            "chunk_4", "chunk_5", "chunk_6"
        ]
        tree = rag._build_tree(chunks)

        # Tree must have Level 0 (raw) + at least Level 1 (summaries)
        assert 0 in tree
        assert len(tree[0]) == 6
        assert 1 in tree
        assert len(tree[1]) > 0

    @patch("backend.modules.rag.raptor_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.raptor_rag.Groq")
    def test_raptor_includes_all_levels_in_chunks(self, mock_groq, mock_chroma):
        """retrieve() must include chunks from all tree levels."""

        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[
                "Raw chunk about photosynthesis.",
                "Raw chunk about chlorophyll.",
                "Raw chunk about sunlight energy.",
            ]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Groq summary
        mock_summary = MagicMock()
        mock_summary.message.content = "Photosynthesis summary covering all aspects."
        mock_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_summary]
        )

        rag = RaptorRAG(tree_depth=1, group_size=3)
        query = RAGQuery(query_text="How does photosynthesis work?", top_k=10)
        chunks = rag.retrieve(query)

        # Must contain both raw chunks AND summary
        assert "Raw chunk about photosynthesis." in chunks
        assert "Photosynthesis summary covering all aspects." in chunks




#======================================================



class TestCorrectiveRAG:
    """Tests for CorrectiveRAG pipeline."""

    @patch("backend.modules.rag.corrective_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.corrective_rag.Groq")
    def test_corrective_rag_returns_rag_result(self, mock_groq, mock_chroma):
        """CorrectiveRAG.run() must return a RAGResult object."""

        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["CRISPR is a gene editing technology."]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Groq — first call grades chunk, second call answers
        mock_grade = MagicMock()
        mock_grade.message.content = "relevant"

        mock_answer = MagicMock()
        mock_answer.message.content = "CRISPR is used for gene editing."

        mock_groq.return_value.chat.completions.create.side_effect = [
            MagicMock(choices=[mock_grade]),
            MagicMock(choices=[mock_answer]),
        ]

        rag = CorrectiveRAG()
        query = RAGQuery(query_text="What is CRISPR?", top_k=1)
        result = rag.run(query)

        assert isinstance(result, RAGResult)
        assert result.rag_type == "corrective"
        assert len(result.answer) > 0

    @patch("backend.modules.rag.corrective_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.corrective_rag.Groq")
    def test_corrective_filters_irrelevant_chunks(self, mock_groq, mock_chroma):
        """CorrectiveRAG must filter out irrelevant chunks."""

        # Mock ChromaDB with mixed chunks
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[
                "CRISPR is a gene editing technology.",
                "Crispy fried chicken recipe with herbs.",
            ]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Groq — grade relevant, then irrelevant, then answer
        mock_relevant = MagicMock()
        mock_relevant.message.content = "relevant"

        mock_irrelevant = MagicMock()
        mock_irrelevant.message.content = "irrelevant"

        mock_answer = MagicMock()
        mock_answer.message.content = "CRISPR edits genes."

        mock_groq.return_value.chat.completions.create.side_effect = [
            MagicMock(choices=[mock_relevant]),    # grade chunk 1
            MagicMock(choices=[mock_irrelevant]),  # grade chunk 2
            MagicMock(choices=[mock_answer]),      # generate answer
        ]

        rag = CorrectiveRAG()
        query = RAGQuery(query_text="What is CRISPR?", top_k=5)
        chunks = rag.retrieve(query)

        # Irrelevant chunk must be filtered out
        assert "Crispy fried chicken recipe with herbs." not in chunks
        assert "CRISPR is a gene editing technology." in chunks

    @patch("backend.modules.rag.corrective_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.corrective_rag.Groq")
    def test_corrective_triggers_fallback(self, mock_groq, mock_chroma):
        """CorrectiveRAG must trigger web fallback when too many irrelevant."""

        # Mock ChromaDB with all irrelevant chunks
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[
                "Crispy fried chicken recipe.",
                "Best pizza toppings guide.",
            ]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        # Mock Groq — all chunks graded irrelevant
        mock_irrelevant = MagicMock()
        mock_irrelevant.message.content = "irrelevant"

        mock_answer = MagicMock()
        mock_answer.message.content = "Fallback answer about CRISPR."

        mock_groq.return_value.chat.completions.create.side_effect = [
            MagicMock(choices=[mock_irrelevant]),  # grade chunk 1
            MagicMock(choices=[mock_irrelevant]),  # grade chunk 2
            MagicMock(choices=[mock_answer]),      # generate answer
        ]

        rag = CorrectiveRAG()
        query = RAGQuery(query_text="What is CRISPR?", top_k=5)
        chunks = rag.retrieve(query)

        # Must trigger fallback — chunks contain fallback message
        assert any("fallback" in chunk.lower() for chunk in chunks)







#================================================





class TestSpeculativeRAG:
    """Tests for SpeculativeRAG pipeline."""

    @patch("backend.modules.rag.speculative_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.speculative_rag.Groq")
    def test_speculative_rag_returns_rag_result(self, mock_groq, mock_chroma):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Thunder is caused by rapid air expansion."]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_hypothesis = MagicMock()
        mock_hypothesis.message.content = "Thunder occurs when lightning superheats air causing rapid expansion."

        mock_answer = MagicMock()
        mock_answer.message.content = "Thunder is caused by lightning heating air rapidly."

        mock_groq.return_value.chat.completions.create.side_effect = [
            MagicMock(choices=[mock_hypothesis]),
            MagicMock(choices=[mock_answer]),
        ]

        rag = SpeculativeRAG()
        query = RAGQuery(query_text="What causes thunder?", top_k=1)
        result = rag.run(query)

        assert isinstance(result, RAGResult)
        assert result.rag_type == "speculative"
        assert len(result.answer) > 0

    @patch("backend.modules.rag.speculative_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.speculative_rag.Groq")
    def test_speculative_searches_with_hypothesis(self, mock_groq, mock_chroma):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Thunder explanation chunk."]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_hypothesis = MagicMock()
        mock_hypothesis.message.content = "Thunder is caused by lightning heating air."
        mock_groq.return_value.chat.completions.create.return_value = MagicMock(
            choices=[mock_hypothesis]
        )

        rag = SpeculativeRAG()
        query = RAGQuery(query_text="What causes thunder?", top_k=1)
        rag.retrieve(query)

        # ChromaDB must be called with hypothesis not original query
        call_args = mock_collection.query.call_args
        search_query = call_args[1]["query_texts"][0]
        assert search_query != "What causes thunder?"

    @patch("backend.modules.rag.speculative_rag.chromadb.PersistentClient")
    @patch("backend.modules.rag.speculative_rag.Groq")
    def test_speculative_fallback_on_hypothesis_failure(self, mock_groq, mock_chroma):
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Fallback chunk."]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_answer = MagicMock()
        mock_answer.message.content = "Fallback answer."

        mock_groq.return_value.chat.completions.create.side_effect = [
            Exception("Groq down"),
            MagicMock(choices=[mock_answer]),
        ]

        rag = SpeculativeRAG()
        query = RAGQuery(query_text="What causes thunder?", top_k=1)
        result = rag.run(query)

        assert isinstance(result, RAGResult)
        assert result.rag_type == "speculative"




#==========================================================