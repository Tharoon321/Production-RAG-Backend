from __future__ import annotations

import os
from typing import Dict, List

import cohere
from dotenv import load_dotenv

load_dotenv()

RERANK_MODEL = "rerank-english-v3.0"

cohere_client = cohere.Client(
    os.getenv("CO_API_KEY")
)


def rerank_chunks(
    query: str,
    candidates: List[Dict],
    top_n: int = 5,
) -> List[Dict]:

    documents = [
        candidate["text"]
        for candidate in candidates
        if "text" in candidate
    ]

    if not documents:
        return []

    response = cohere_client.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=min(top_n, len(documents)),
    )

    reranked_results = []

    for result in response.results:

        candidate = candidates[result.index]

        reranked_chunk = candidate.copy()

        reranked_chunk[
            "relevance_score"
        ] = result.relevance_score

        reranked_results.append(
            reranked_chunk
        )

    return reranked_results