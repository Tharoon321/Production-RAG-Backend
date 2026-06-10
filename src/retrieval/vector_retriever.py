from __future__ import annotations

from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer

from src.ingestion.indexer import (
    COLLECTION_NAME,
    qdrant_client,
)

_model = None


def get_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    return _model


def vector_search(
    query: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Performs semantic vector search in Qdrant.
    """

    model = get_model()

    query_vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit,
    )

    retrieved_chunks = []

    for result in results:

        payload = result.payload

        retrieved_chunks.append(
            {
                "chunk_id": payload["chunk_id"],
                "doc_id": payload["doc_id"],
                "text": payload["text"],
                "score": result.score,
            }
        )

    return retrieved_chunks