"""
pinecone_client.py — Production vector DB client using Pinecone Serverless.
Namespaced by location (bali / dubai) for fast filtered retrieval.

Setup:
  1. Create a free account at https://www.pinecone.io
  2. Create an index named "travel-brain" with dimension matching your embedder:
     - OpenAI text-embedding-3-small → 1536
     - Local all-MiniLM-L6-v2       → 384
  3. Set PINECONE_API_KEY in .env
"""

import logging
from typing import Optional

from pinecone import Pinecone, ServerlessSpec
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from travel_brain import config
from travel_brain.vectordb.base_client import VectorDBClient

logger = logging.getLogger(__name__)

UPSERT_BATCH_SIZE = 100  # Pinecone recommends ≤100 vectors per upsert call


class PineconeClient(VectorDBClient):
    """
    Pinecone Serverless vector DB client.
    Index is created automatically if it doesn't exist.
    Namespaces: one per location (e.g. "bali", "dubai").
    """

    def __init__(self):
        if not config.PINECONE_API_KEY:
            raise ValueError(
                "PINECONE_API_KEY is not set. Add it to your .env file.\n"
                "Sign up at: https://www.pinecone.io"
            )
        self._pc    = Pinecone(api_key=config.PINECONE_API_KEY)
        self._index = self._get_or_create_index()

    def _get_or_create_index(self):
        """Return existing index or create it if it doesn't exist."""
        index_name = config.PINECONE_INDEX_NAME
        existing   = [i.name for i in self._pc.list_indexes()]

        if index_name not in existing:
            logger.info(f"Creating Pinecone index '{index_name}' (dim={config.EMBEDDING_DIM})")
            self._pc.create_index(
                name=index_name,
                dimension=config.EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=config.PINECONE_ENVIRONMENT,
                ),
            )
            logger.info(f"Index '{index_name}' created ✓")
        else:
            logger.info(f"Using existing Pinecone index: '{index_name}'")

        return self._pc.Index(index_name)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def upsert(self, chunks: list[dict], namespace: str = "") -> int:
        """
        Upsert chunks into Pinecone.
        Each chunk must have: doc_id (str), embedding (list[float]),
        metadata (dict), chunk_text (str).
        """
        vectors = [
            {
                "id":       chunk["chunk_id"],
                "values":   chunk["embedding"],
                "metadata": {
                    **chunk["metadata"],
                    "text": chunk["chunk_text"][:4096],  # Pinecone metadata string limit
                },
            }
            for chunk in chunks
            if chunk.get("embedding") and chunk.get("chunk_id")
        ]

        if not vectors:
            logger.warning("No valid vectors to upsert.")
            return 0

        upserted = 0
        for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
            batch = vectors[i : i + UPSERT_BATCH_SIZE]
            self._index.upsert(vectors=batch, namespace=namespace)
            upserted += len(batch)
            logger.info(f"  Pinecone: upserted {upserted}/{len(vectors)} vectors (namespace='{namespace}')")

        return upserted

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        namespace: str = "",
        filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Query Pinecone for nearest neighbors with optional metadata filters.

        Example filter (Pinecone filter syntax):
        {
            "location": {"$eq": "bali"},
            "is_hidden_gem": {"$eq": True},
            "budget_level": {"$in": ["free", "budget"]},
        }
        """
        kwargs = dict(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )
        if filter:
            kwargs["filter"] = filter

        response = self._index.query(**kwargs)
        results  = []
        for match in response.matches:
            results.append({
                "id":       match.id,
                "score":    match.score,
                "metadata": match.metadata,
                "text":     match.metadata.get("text", ""),
            })
        return results

    def delete_namespace(self, namespace: str) -> None:
        """Delete all vectors in a namespace. Use before re-ingestion."""
        logger.warning(f"Deleting all vectors in namespace '{namespace}'...")
        self._index.delete(delete_all=True, namespace=namespace)
        logger.info(f"Namespace '{namespace}' cleared.")

    def describe(self) -> dict:
        """Return index statistics."""
        stats = self._index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension":     stats.dimension,
            "namespaces":    {k: v.vector_count for k, v in (stats.namespaces or {}).items()},
        }
