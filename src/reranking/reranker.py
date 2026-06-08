
from __future__ import annotations

import os
from typing import Dict, List

import cohere
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------
# Cohere reranking model configuration
# ---------------------------------------------------------
RERANK_MODEL = "rerank-english-v3.0"


# ---------------------------------------------------------
# Initialize Cohere client
# -----------------------

cohere_client = cohere.Client(
    os.getenv("CO_API_KEY")
)


# ---------------------------------------------------------
# Main reranking function
# ---------------------------------------------------------
def rerank_chunks(
    query: str,
    candidates: List[Dict],
    top_n: int = 5,
) -> List[Dict]:
    """
    Reranks retrieved chunks using Cohere cross-encoder.

    Args:
        query:
            User query

        candidates:
            Retrieved candidate chunks

        top_n:
            Number of final chunks to keep

    Returns:
        Top reranked chunks with relevance scores
    """

    # Extract raw chunk texts for reranking
    documents = [
        candidate["text"]
        for candidate in candidates
        if "text" in candidate
    ]

    # Call Cohere reranking API
    response = cohere_client.rerank(
        model=RERANK_MODEL,

        query=query,

        documents=documents,

        top_n=top_n,
    )

    reranked_results = []

    # Cohere returns ranked indices
    for result in response.results:

        # Locate original candidate chunk
        candidate = candidates[result.index]

        # Copy candidate to avoid mutating original object
        reranked_chunk = candidate.copy()

        # Add reranking relevance score
        reranked_chunk["relevance_score"] = result.relevance_score

        reranked_results.append(reranked_chunk)

    return reranked_results

