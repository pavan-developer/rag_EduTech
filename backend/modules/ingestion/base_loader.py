"""
@module   BaseDocumentLoader
@description Abstract base class for all document loaders.
             Uses Strategy Pattern — each document type (PDF, DOCX,
             TXT, URL, YouTube) implements this interface separately.
             Open/Closed Principle: add new types without touching existing code.
@author   EduMind AI Engineering
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from datetime import datetime


@dataclass
class DocumentChunk:
    """Represents a single processed chunk of a document."""
    content: str
    source: str
    chunk_index: int
    total_chunks: int
    metadata: dict
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()


class BaseDocumentLoader(ABC):
    """
    Abstract base for all document loaders.
    Every loader MUST implement: load() and validate()
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def validate(self, source: str) -> bool:
        """Check if the source is valid before loading."""
        pass

    @abstractmethod
    def load(self, source: str) -> List[DocumentChunk]:
        """Load and chunk the document. Returns list of DocumentChunk."""
        pass

    def get_loader_name(self) -> str:
        """Returns the name of the loader for logging."""
        return self.__class__.__name__