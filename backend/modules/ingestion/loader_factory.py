"""
@module   LoaderFactory
@description Factory Pattern — automatically selects the correct document
             loader based on file extension. RAG pipeline never needs to
             know which loader to use — just ask the factory.
             Open/Closed Principle: add new loaders by registering here only.
@author   EduMind AI Engineering
"""

from backend.modules.ingestion.base_loader import BaseDocumentLoader
from backend.modules.ingestion.pdf_loader import PDFLoader
from backend.modules.ingestion.txt_loader import TXTLoader
from backend.modules.ingestion.docx_loader import DOCXLoader


class LoaderFactory:
    """
    Returns the correct loader for a given file type.
    Add new loaders here — zero changes needed anywhere else.
    """

    # Registry — file extension mapped to loader class
    _REGISTRY = {
        ".pdf":  PDFLoader,
        ".txt":  TXTLoader,
        ".docx": DOCXLoader,
    }

    @staticmethod
    def get_loader(source: str, chunk_size: int = 500, chunk_overlap: int = 50) -> BaseDocumentLoader:
        """
        Given a file path, return the correct loader instance.
        Raises ValueError if file type is not supported.
        """
        # Extract extension from filename
        extension = "." + source.rsplit(".", 1)[-1].lower()

        if extension not in LoaderFactory._REGISTRY:
            supported = ", ".join(LoaderFactory._REGISTRY.keys())
            raise ValueError(
                f"Unsupported file type '{extension}'. "
                f"Supported types: {supported}"
            )

        # Get loader class and return instance
        loader_class = LoaderFactory._REGISTRY[extension]
        return loader_class(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    @staticmethod
    def get_supported_types() -> list:
        """Returns list of all supported file extensions."""
        return list(LoaderFactory._REGISTRY.keys())