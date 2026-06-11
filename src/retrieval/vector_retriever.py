from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv

from src.ingestion.indexer import (
    COLLECTION_NAME,
    get_qdrant_client,
)

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------
# Gemini embedding model
# ---------------------------------------------------------
MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")

# ---------------------------------------------------------
# Lazy-loaded Gemini SDK
# ---------------------------------------------------------
_genai = None


def get_genai():
    """
    Lazy loads Gemini SDK.
    """

    global _genai

    if _genai is None:

        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in environment")

        genai.configure(api_key=api_key)

        _genai = genai

    return _genai


# ---------------------------------------------------------
# Vector search
# ---------------------------------------------------------
def vector_search(
    query: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Performs semantic vector search
    using Gemini embeddings + Qdrant.
    """

    # -----------------------------------------------------
    # Lazy-load Gemini
    # -----------------------------------------------------
    genai = get_genai()

    # -----------------------------------------------------
    # Generate query embedding
    # -----------------------------------------------------
    response = genai.embed_content(
        model=MODEL,

        content=query,

        task_type="retrieval_query",
    )

    query_vector = response[
        "embedding"
    ]

    # -----------------------------------------------------
    # Lazy-load Qdrant client
    # -----------------------------------------------------
    qdrant_client = (
        get_qdrant_client()
    )

    # -----------------------------------------------------
    # Perform vector search
    # -----------------------------------------------------
    results = qdrant_client.search(
        collection_name=
            COLLECTION_NAME,

        query_vector=
            query_vector,

        limit=limit,
    )

    retrieved_chunks = []

    # -----------------------------------------------------
    # Format results
    # -----------------------------------------------------
    for result in results:

        payload = result.payload

        retrieved_chunks.append(
            {
                "chunk_id":
                    payload["chunk_id"],

                "doc_id":
                    payload["doc_id"],

                "text":
                    payload["text"],

                "score":
                    result.score,
            }
        )

    return retrieved_chunks