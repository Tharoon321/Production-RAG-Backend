from __future__ import annotations

import os
from typing import Dict, List

from dotenv import load_dotenv

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------
# Cohere reranking model
# ---------------------------------------------------------
RERANK_MODEL = "rerank-english-v3.0"

# ---------------------------------------------------------
# Lazy-loaded Cohere client
# ---------------------------------------------------------
_cohere_client = None


def get_cohere_client():
    """
    Lazy loads Cohere client only when needed.
    """

    global _cohere_client

    if _cohere_client is None:

        import cohere

        _cohere_client = cohere.Client(
            os.getenv("CO_API_KEY")
        )

    return _cohere_client


# ---------------------------------------------------------
# Main reranking function
# ---------------------------------------------------------
def rerank_chunks(
    query: str,
    candidates: List[Dict],
    top_n: int = 5,
) -> List[Dict]:
    """
    Reranks retrieved chunks using Cohere.
    """

    # -----------------------------------------------------
    # Extract candidate texts
    # -----------------------------------------------------
    documents = [
        candidate["text"]
        for candidate in candidates
        if "text" in candidate
    ]

    # -----------------------------------------------------
    # Handle empty retrieval
    # -----------------------------------------------------
    if not documents:
        return []

    # -----------------------------------------------------
    # Lazy-load Cohere client
    # -----------------------------------------------------
    cohere_client = get_cohere_client()

    # -----------------------------------------------------
    # Cohere reranking
    # -----------------------------------------------------
    response = cohere_client.rerank(
        model=RERANK_MODEL,

        query=query,

        documents=documents,

        top_n=min(
            top_n,
            len(documents),
        ),
    )

    reranked_results = []

    # -----------------------------------------------------
    # Build reranked output
    # -----------------------------------------------------
    for result in response.results:

        candidate = candidates[
            result.index
        ]

        reranked_chunk = (
            candidate.copy()
        )

        reranked_chunk[
            "relevance_score"
        ] = result.relevance_score

        reranked_results.append(
            reranked_chunk
        )

    return reranked_results