import hashlib
import logging
import sys
from pathlib import Path

# Ensure travel_brain is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from travel_brain.vectordb.chroma_client import ChromaClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def deduplicate_chroma():
    """
    Iterates through all ChromaDB collections, identifies exact duplicate text chunks,
    and removes the duplicates from the database to keep the RAG context clean.
    """
    client = ChromaClient()
    stats = client.describe()
    
    total_deleted = 0
    collections = stats.get("collections", {})
    
    for col_name, count in collections.items():
        if count == 0:
            continue
            
        logger.info(f"Checking collection '{col_name}' ({count} items)...")
        
        # Access the underlying chromadb collection natively
        col = client._client.get_collection(col_name)
        
        offset = 0
        limit = 1000
        seen_hashes = set()
        duplicates_to_delete = []
        
        # Paginate to handle larger collections
        while True:
            batch = col.get(limit=limit, offset=offset, include=["documents"])
            if not batch or not batch["ids"]:
                break
                
            for i, doc_id in enumerate(batch["ids"]):
                text = batch["documents"][i]
                if not text:
                    continue
                    
                # We hash the text content to find exact duplicates
                text_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
                
                if text_hash in seen_hashes:
                    duplicates_to_delete.append(doc_id)
                else:
                    seen_hashes.add(text_hash)
            
            offset += limit
            if offset >= count:
                break
            
        if duplicates_to_delete:
            logger.info(f"Found {len(duplicates_to_delete)} duplicate chunks in '{col_name}'. Deleting...")
            
            # Delete in smaller batches to be safe
            for i in range(0, len(duplicates_to_delete), 500):
                batch_ids = duplicates_to_delete[i:i+500]
                col.delete(ids=batch_ids)
                
            total_deleted += len(duplicates_to_delete)
        else:
            logger.info(f"No duplicates found in '{col_name}'.")

    logger.info(f"Deduplication complete! Total duplicates removed: {total_deleted}")
    
    # Print the new database stats
    new_stats = client.describe()
    logger.info(f"New Database stats: {new_stats}")

if __name__ == "__main__":
    logger.info("Starting ChromaDB deduplication...")
    deduplicate_chroma()
