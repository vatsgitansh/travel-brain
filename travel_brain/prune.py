import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Ensure we can import travel_brain modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from travel_brain.vectordb.chroma_client import ChromaClient
from travel_brain import config

logger = logging.getLogger("DB_Pruner")

def prune_outdated_vectors(days_threshold: int = 60) -> int:
    """
    Deletes vectors from ChromaDB where the content_freshness (YYYY-MM-DD)
    is older than the specified days threshold.
    """
    cl = ChromaClient()
    stats = cl.describe()
    collections = stats.get("collections", {})
    
    # Calculate cutoff date string in YYYY-MM-DD format (lexicographically sortable)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_threshold)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    
    logger.info(f"Pruning vectors older than {days_threshold} days (cutoff: {cutoff_str})...")
    deleted_total = 0
    
    for col_name, count in collections.items():
        if count == 0:
            continue
            
        logger.info(f"Checking collection '{col_name}' ({count} items)...")
        col = cl._client.get_collection(col_name)
        
        # Scan through the collection using pagination to be memory safe
        offset = 0
        limit = 1000
        ids_to_delete = []
        
        while True:
            batch = col.get(limit=limit, offset=offset, include=["metadatas"])
            if not batch or not batch["ids"]:
                break
                
            for i, metadata in enumerate(batch["metadatas"]):
                freshness = metadata.get("content_freshness", "") if metadata else ""
                
                # If freshness exists and is lexicographically older than our cutoff
                # (YYYY-MM-DD format allows alphabetical comparison for dates)
                if freshness and freshness < cutoff_str:
                    ids_to_delete.append(batch["ids"][i])
            
            offset += limit

        if ids_to_delete:
            col.delete(ids=ids_to_delete)
            logger.warning(f"  → Deleted {len(ids_to_delete)} outdated chunks from '{col_name}'")
            deleted_total += len(ids_to_delete)
        else:
            logger.info(f"  → No outdated chunks found in '{col_name}'.")
            
    return deleted_total

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    
    # You can pass days via command line, defaulting to 60 days
    days = 60
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            pass
            
    deleted = prune_outdated_vectors(days_threshold=days)
    logger.info(f"✅ DB Pruning Complete. Removed {deleted} old chunks across all locations.")
