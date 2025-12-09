"""
RAG (Retrieval-Augmented Generation) Module for Legal Advice
"""
import os
import json
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from src.config import settings
import logging

logger = logging.getLogger(__name__)

class LegalRAG:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.RAG_VECTOR_DB_PATH)
        
        # Use OpenAI embedding function if API key is present, otherwise use default (for testing)
        if settings.OPENAI_API_KEY:
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.RAG_EMBEDDING_MODEL
            )
        else:
            logger.warning("OPENAI_API_KEY not found. Using default SentenceTransformer embedding.")
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
        self.collection = self.client.get_or_create_collection(
            name=settings.RAG_COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )
        
    def ingest_documents(self, documents: List[Dict[str, str]]):
        """
        Ingest legal documents into the vector database.
        documents format: [{"id": "law_1", "text": "Article 1...", "metadata": {"source": "Criminal Code"}}]
        """
        if not documents:
            return
            
        ids = [doc["id"] for doc in documents]
        documents_text = [doc["text"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        
        self.collection.upsert(
            ids=ids,
            documents=documents_text,
            metadatas=metadatas
        )
        logger.info(f"Ingested {len(documents)} documents into RAG system.")

    def retrieve(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant legal documents for a given query.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        retrieved_docs = []
        if results["documents"]:
            for i, doc_text in enumerate(results["documents"][0]):
                retrieved_docs.append({
                    "text": doc_text,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0
                })
                
        return retrieved_docs

# Singleton instance
legal_rag = LegalRAG()
