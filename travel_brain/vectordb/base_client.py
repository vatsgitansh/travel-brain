"""
base_client.py — Abstract Vector DB interface.
All concrete implementations (Pinecone, ChromaDB) must implement this interface.
This ensures swapping vector DBs requires zero changes in the pipeline.
"""

from abc import ABC, abstractmethod
from typing import Optional


class VectorDBClient(ABC):

    @abstractmethod
    def upsert(self, chunks: list[dict], namespace: str = "") -> int:
        """
        Insert or update chunks in the vector DB.
        Each chunk must have: doc_id, embedding, metadata, chunk_text.
        Returns the number of vectors upserted.
        """
        ...

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        namespace: str = "",
        filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Query the vector DB for nearest neighbors.
        Returns list of {id, score, metadata, text} dicts.
        """
        ...

    @abstractmethod
    def delete_namespace(self, namespace: str) -> None:
        """Delete all vectors in a namespace (used for re-ingestion)."""
        ...

    @abstractmethod
    def describe(self) -> dict:
        """Return stats about the current index/collection (count, dimension, etc.)."""
        ...
