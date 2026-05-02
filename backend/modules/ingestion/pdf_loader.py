"""
@module   PDFLoader
@description Loads and chunks PDF documents for RAG pipeline ingestion.
             Inherits BaseDocumentLoader — implements validate() and load().
             Uses PyMuPDF (fitz) for fast, accurate PDF text extraction.
@author   EduMind AI Engineering
"""

import os
from typing import List

import fitz  # PyMuPDF

from backend.modules.ingestion.base_loader import BaseDocumentLoader, DocumentChunk


class PDFLoader(BaseDocumentLoader):
    """
    Loads PDF files and splits them into overlapping chunks.
    Strategy Pattern: plug this into any RAG pipeline via BaseDocumentLoader.
    """

    def validate(self, source: str) -> bool:
        """Check file exists and is a valid PDF."""
        if not os.path.exists(source):
            raise FileNotFoundError(f"PDF not found: {source}")
        if not source.lower().endswith(".pdf"):
            raise ValueError(f"Not a PDF file: {source}")
        return True

    def load(self, source: str) -> List[DocumentChunk]:
        """Extract text from PDF and split into chunks."""
        self.validate(source)

        try:
            doc = fitz.open(source)
            full_text = ""

            # Extract text from every page
            for page in doc:
                full_text += page.get_text()

            doc.close()

            # Split into overlapping chunks
            chunks = self._split_into_chunks(full_text, source)
            return chunks

        except Exception as e:
            raise RuntimeError(f"Failed to load PDF '{source}': {str(e)}")

    def _split_into_chunks(self, text: str, source: str) -> List[DocumentChunk]:
        """Split raw text into overlapping DocumentChunk objects."""
        chunks = []
        start = 0
        chunk_index = 0
        text_length = len(text)

        # First pass: count total chunks
        temp_start = 0
        total = 0
        while temp_start < text_length:
            temp_start += self.chunk_size - self.chunk_overlap
            total += 1

        # Second pass: build chunks
        while start < text_length:
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    source=source,
                    chunk_index=chunk_index,
                    total_chunks=total,
                    metadata={
                        "file_type": "pdf",
                        "file_name": os.path.basename(source),
                        "loader": self.get_loader_name()
                    }
                ))
                chunk_index += 1

            start += self.chunk_size - self.chunk_overlap

        return chunks