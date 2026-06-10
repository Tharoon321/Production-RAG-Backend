from __future__ import annotations

import os
from typing import List, Dict, Any

import google.generativeai as genai

from src.ingestion.indexer import (
    COLLECTION_NAME,
    qdrant_client,
)

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

MODEL = "models/text-embedding-004"


def vector_search(
    query: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:

    response = genai.embed_content(
        model=MODEL,
        content=query,
        task_type="retrieval_query"
    )

    query_vector = response["embedding"]

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