"""
@module   TestIngestion
@description Integration test for the full document ingestion pipeline.
             Tests: TXT loading, Factory routing, IngestionService output.
             Run with: python -m pytest backend/tests/test_ingestion.py -v
@author   EduMind AI Engineering
"""

import os
import pytest

from backend.modules.ingestion.ingestion_service import IngestionService
from backend.modules.ingestion.loader_factory import LoaderFactory


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_txt_file(tmp_path):
    """Creates a temporary TXT file for testing."""
    content = (
        "Photosynthesis is the process by which plants use sunlight, "
        "water and carbon dioxide to produce oxygen and energy in the "
        "form of sugar. This process happens inside chloroplasts. "
        "Chloroplasts contain a substance called chlorophyll which "
        "absorbs sunlight. The light energy is used to convert water "
        "and carbon dioxide into glucose and oxygen. Glucose is used "
        "by the plant for energy and growth. Oxygen is released into "
        "the atmosphere as a byproduct of photosynthesis. "
        "This is why plants are essential for life on Earth. "
        "Without photosynthesis there would be no oxygen in the air."
    )
    file_path = tmp_path / "sample.txt"
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


# ── Tests ────────────────────────────────────────────────────────────────────

def test_factory_returns_correct_loader():
    """Factory must return correct loader for each extension."""
    from backend.modules.ingestion.txt_loader import TXTLoader
    from backend.modules.ingestion.pdf_loader import PDFLoader
    from backend.modules.ingestion.docx_loader import DOCXLoader

    assert isinstance(LoaderFactory.get_loader("notes.txt"),  TXTLoader)
    assert isinstance(LoaderFactory.get_loader("notes.pdf"),  PDFLoader)
    assert isinstance(LoaderFactory.get_loader("notes.docx"), DOCXLoader)


def test_factory_rejects_unsupported_type():
    """Factory must raise ValueError for unsupported file types."""
    with pytest.raises(ValueError):
        LoaderFactory.get_loader("video.mp4")


def test_ingestion_service_with_txt(sample_txt_file):
    """Full pipeline test — TXT file through IngestionService."""
    service = IngestionService(chunk_size=200, chunk_overlap=20)
    result = service.ingest(sample_txt_file)

    assert result["success"] is True
    assert result["total_chunks"] > 0
    assert result["loader_used"] == "TXTLoader"
    assert len(result["chunks"]) > 0
    assert result["chunks"][0]["content"] != ""


def test_ingestion_service_file_not_found():
    """Service must handle missing files gracefully."""
    service = IngestionService()
    result = service.ingest("nonexistent_file.txt")

    assert result["success"] is False
    assert "error" in result


def test_supported_types():
    """Service must return correct supported file types."""
    service = IngestionService()
    types = service.get_supported_types()

    assert ".pdf"  in types
    assert ".txt"  in types
    assert ".docx" in types