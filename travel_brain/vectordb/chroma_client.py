"""
chroma_client.py — Local ChromaDB vector DB client.
Zero configuration, fully offline. Used for:
  - Local development (free, no API keys)
  - Testing pipeline logic without cloud dependency
  - Future offline phone app (ChromaDB can be embedded in a mobile binary)

Data is persisted to data/chroma_db/ by default.
"""

import logging
from typing import Optional

import chromadb
from chromadb.config import Settings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from travel_brain import config
from travel_brain.vectordb.base_client import VectorDBClient

logger = logging.getLogger(__name__)

UPSERT_BATCH_SIZE = 500   # ChromaDB handles larger batches locally


class ChromaClient(VectorDBClient):
    """
    Local ChromaDB client. Uses one collection per location namespace.
    All data is persisted to CHROMA_PERSIST_DIR.
    """

    def __init__(self):
        persist_dir = str(config.CHROMA_DIR)
        logger.info(f"ChromaDB persistence directory: {persist_dir}")

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collections: dict[str, chromadb.Collection] = {}

    def _get_collection(self, namespace: str) -> chromadb.Collection:
        """Get or create a collection for a given namespace (location)."""
        name = f"travel_brain_{namespace}" if namespace else "travel_brain_default"
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB collection ready: '{name}'")
        return self._collections[name]

    def upsert(self, chunks: list[dict], namespace: str = "") -> int:
        """Upsert chunks into the ChromaDB collection for this namespace.

        Accepts both 'chunk_text' and 'cleaned_text' field names for the
        document body (pipeline output uses 'cleaned_text').
        """
        collection = self._get_collection(namespace)

        # Normalise: accept either 'chunk_text' or 'cleaned_text'
        def _get_text(c: dict) -> str:
            return c.get("chunk_text") or c.get("cleaned_text") or ""

        # Filter out chunks missing required fields
        valid = [
            c for c in chunks
            if c.get("embedding") and c.get("chunk_id") and _get_text(c)
        ]
        if not valid:
            logger.warning("No valid chunks to upsert into ChromaDB.")
            return 0

        upserted = 0
        for i in range(0, len(valid), UPSERT_BATCH_SIZE):
            batch = valid[i : i + UPSERT_BATCH_SIZE]

            ids         = [c["chunk_id"] for c in batch]
            embeddings  = [c["embedding"] for c in batch]
            documents   = [_get_text(c) for c in batch]
            metadatas   = []

            for c in batch:
                meta = dict(c.get("metadata", {}))
                # ChromaDB only allows str/int/float/bool values in metadata
                cleaned_meta: dict = {}
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        cleaned_meta[k] = v
                    elif v is None:
                        cleaned_meta[k] = ""
                    else:
                        cleaned_meta[k] = str(v)
                metadatas.append(cleaned_meta)

            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            upserted += len(batch)
            logger.info(f"  ChromaDB: upserted {upserted}/{len(valid)} (namespace='{namespace}')")

        return upserted

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        namespace: str = "",
        filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Query ChromaDB for nearest neighbors.

        Example filter (ChromaDB where syntax):
        {"location": "bali", "is_hidden_gem": True}
        or compound:
        {"$and": [{"location": "bali"}, {"is_hidden_gem": True}]}
        """
        collection = self._get_collection(namespace)

        kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        if filter:
            kwargs["where"] = filter

        response = collection.query(**kwargs)

        results = []
        for i, doc_id in enumerate(response["ids"][0]):
            results.append({
                "id":       doc_id,
                "score":    1.0 - response["distances"][0][i],  # Convert distance → similarity
                "metadata": response["metadatas"][0][i],
                "text":     response["documents"][0][i],
            })
        return results

    def delete_namespace(self, namespace: str) -> None:
        """Delete the entire collection for this namespace."""
        name = f"travel_brain_{namespace}" if namespace else "travel_brain_default"
        try:
            self._client.delete_collection(name)
            logger.warning(f"ChromaDB collection '{name}' deleted.")
            self._collections.pop(name, None)
        except Exception as e:
            logger.error(f"Could not delete collection '{name}': {e}")

    def describe(self) -> dict:
        """Return stats across all collections."""
        collections = self._client.list_collections()
        stats = {}
        for col in collections:
            c = self._client.get_collection(col.name)
            stats[col.name] = c.count()
        return {"collections": stats, "total_vectors": sum(stats.values())}
