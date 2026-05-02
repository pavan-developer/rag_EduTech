"""
@module   IngestionService
@description Orchestrates the full document ingestion pipeline.
             Accepts a file path → validates → loads → chunks → returns.
             This is the ONLY class the API layer talks to for ingestion.
             Single Responsibility: coordination only, no loading logic here.
@author   EduMind AI Engineering
"""

import os
from typing import List

from backend.modules.ingestion.base_loader import DocumentChunk
from backend.modules.ingestion.loader_factory import LoaderFactory


class IngestionService:
    """
    Orchestrates document ingestion.
    Controller → IngestionService → LoaderFactory → Loader → DocumentChunks
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, file_path: str) -> dict:
        """
        Main entry point. Takes file path, returns ingestion result.
        Returns: { success, file_name, total_chunks, chunks }
        """
        try:
            # Step 1 — Check file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Step 2 — Get correct loader from factory
            loader = LoaderFactory.get_loader(
                file_path,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

            # Step 3 — Load and chunk the document
            chunks: List[DocumentChunk] = loader.load(file_path)

            # Step 4 — Return structured result
            return {
                "success": True,
                "file_name": os.path.basename(file_path),
                "loader_used": loader.get_loader_name(),
                "total_chunks": len(chunks),
                "chunks": [
                    {
                        "chunk_index": c.chunk_index,
                        "content": c.content,
                        "metadata": c.metadata,
                        "created_at": c.created_at
                    }
                    for c in chunks
                ]
            }

        except (FileNotFoundError, ValueError) as e:
            return {
                "success": False,
                "error": str(e),
                "file_name": os.path.basename(file_path)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "file_name": os.path.basename(file_path)
            }

    def get_supported_types(self) -> list:
        """Returns supported file extensions — used by API for validation."""
        return LoaderFactory.get_supported_types()