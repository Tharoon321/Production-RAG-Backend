from __future__ import annotations

from typing import List, Dict, Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.ingestion.indexer import (
    COLLECTION_NAME,
    qdrant_client,
)

# ---------------------------------------------------------
# Load embedding model once at startup
# ---------------------------------------------------------
# all-MiniLM-L6-v2 produces 384-dimensional vectors
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# Retrieve documents using vector similarity
# ---------------------------------------------------------
def vector_search(
    query: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Performs semantic vector search in Qdrant.

    Returns:
        [
            {
                "chunk_id": "...",
                "doc_id": "...",
                "text": "...",
                "score": 0.87
            }
        ]
    """

    # Convert query into embedding vector
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    # Search Qdrant
    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit,
    )

    retrieved_chunks = []

    for result in results:

        payload = result.payload
        print("\nPAYLOAD:")
        print(payload)
        retrieved_chunks.append(
            {
                "chunk_id": payload["chunk_id"],
                "doc_id": payload["doc_id"],
                "text": payload["text"],
                "score": result.score,
            }
        )

    return retrieved_chunks