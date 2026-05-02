"""
@module   TXTLoader
@description Loads and chunks plain text (.txt) files for RAG pipeline ingestion.
             Inherits BaseDocumentLoader — implements validate() and load().
             Handles multiple encodings (UTF-8, Latin-1) automatically.
@author   EduMind AI Engineering
"""

import os
from typing import List

from backend.modules.ingestion.base_loader import BaseDocumentLoader, DocumentChunk


class TXTLoader(BaseDocumentLoader):
    """
    Loads plain text files and splits them into overlapping chunks.
    Tries UTF-8 first, falls back to Latin-1 for older files.
    """

    def validate(self, source: str) -> bool:
        """Check file exists and is a .txt file."""
        if not os.path.exists(source):
            raise FileNotFoundError(f"TXT file not found: {source}")
        if not source.lower().endswith(".txt"):
            raise ValueError(f"Not a TXT file: {source}")
        return True

    def load(self, source: str) -> List[DocumentChunk]:
        """Read text file and split into chunks."""
        self.validate(source)

        try:
            # Try UTF-8 first, fall back to Latin-1
            try:
                with open(source, "r", encoding="utf-8") as f:
                    full_text = f.read()
            except UnicodeDecodeError:
                with open(source, "r", encoding="latin-1") as f:
                    full_text = f.read()

            if not full_text.strip():
                raise ValueError(f"TXT file is empty: {source}")

            chunks = self._split_into_chunks(full_text, source)
            return chunks

        except (FileNotFoundError, ValueError):
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load TXT '{source}': {str(e)}")

    def _split_into_chunks(self, text: str, source: str) -> List[DocumentChunk]:
        """Split raw text into overlapping DocumentChunk objects."""
        chunks = []
        start = 0
        chunk_index = 0
        text_length = len(text)

        # Count total chunks first
        temp_start = 0
        total = 0
        while temp_start < text_length:
            temp_start += self.chunk_size - self.chunk_overlap
            total += 1

        # Build chunks
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
                        "file_type": "txt",
                        "file_name": os.path.basename(source),
                        "loader": self.get_loader_name()
                    }
                ))
                chunk_index += 1

            start += self.chunk_size - self.chunk_overlap

        return chunks