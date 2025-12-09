"""
Voice Phishing Detection - RAG Module Init
"""
from .legal_rag import (
    Document,
    RetrievalResult,
    RAGResponse,
    LegalDocumentLoader,
    VectorStore,
    LegalRAG,
    create_rag_system
)

__all__ = [
    "Document",
    "RetrievalResult",
    "RAGResponse",
    "LegalDocumentLoader",
    "VectorStore",
    "LegalRAG",
    "create_rag_system"
]
