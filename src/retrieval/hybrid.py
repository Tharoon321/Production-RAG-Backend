from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Dict, List

# ---------------------------------------------------------
# Reciprocal Rank Fusion constant
# ---------------------------------------------------------
RRF_K = 60


# ---------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------
def reciprocal_rank_fusion(
    bm25_results,
    vector_results,
) -> List[Dict]:
    """
    Combines BM25 and vector search rankings
    using Reciprocal Rank Fusion.
    """

    fused_scores = defaultdict(
        float
    )

    chunk_metadata = {}

    # -----------------------------------------------------
    # BM25 results
    # -----------------------------------------------------
    for rank, (
        chunk_id,
        score,
    ) in enumerate(
        bm25_results,
        start=1,
    ):

        fused_scores[
            chunk_id
        ] += (
            1 / (RRF_K + rank)
        )

        if chunk_id not in chunk_metadata:
            chunk_metadata[
                chunk_id
            ] = {}

        chunk_metadata[
            chunk_id
        ].update(
            {
                "chunk_id":
                    chunk_id,

                "bm25_score":
                    float(score),
            }
        )

    # -----------------------------------------------------
    # Vector results
    # -----------------------------------------------------
    for rank, result in enumerate(
        vector_results,
        start=1,
    ):

        chunk_id = result[
            "chunk_id"
        ]

        fused_scores[
            chunk_id
        ] += (
            1 / (RRF_K + rank)
        )

        if chunk_id not in chunk_metadata:
            chunk_metadata[
                chunk_id
            ] = {}

        chunk_metadata[
            chunk_id
        ].update(
            {
                "chunk_id":
                    result["chunk_id"],

                "doc_id":
                    result["doc_id"],

                "text":
                    result["text"],

                "vector_score":
                    float(
                        result["score"]
                    ),
            }
        )

    # -----------------------------------------------------
    # Build fused results
    # -----------------------------------------------------
    fused_results = []

    for (
        chunk_id,
        rrf_score,
    ) in fused_scores.items():

        metadata = (
            chunk_metadata.get(
                chunk_id,
                {},
            )
        )

        metadata[
            "rrf_score"
        ] = float(rrf_score)

        fused_results.append(
            metadata
        )

    # -----------------------------------------------------
    # Sort by fused score
    # -----------------------------------------------------
    fused_results.sort(
        key=lambda x:
            x["rrf_score"],

        reverse=True,
    )

    return fused_results


# ---------------------------------------------------------
# Async BM25 search
# ---------------------------------------------------------
async def async_bm25_search(
    query: str,
):
    """
    Lazy-load BM25 retriever.
    """

    from src.retrieval.bm25_retriever import (
        bm25_search,
    )

    return bm25_search(
        query=query
    )


# ---------------------------------------------------------
# Async vector search
# ---------------------------------------------------------
async def async_vector_search(
    query: str,
):
    """
    Lazy-load vector retriever.
    """

    from src.retrieval.vector_retriever import (
        vector_search,
    )

    return vector_search(
        query=query
    )


# ---------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------
async def hybrid_search(
    query: str,
    top_k: int = 30,
):
    """
    Performs hybrid retrieval:
        BM25 + Vector + RRF
    """

    # -----------------------------------------------------
    # Run retrievals concurrently
    # -----------------------------------------------------
    (
        bm25_results,
        vector_results,
    ) = await asyncio.gather(

        async_bm25_search(
            query
        ),

        async_vector_search(
            query
        ),
    )

    # -----------------------------------------------------
    # Fuse rankings
    # -----------------------------------------------------
    fused_results = (
        reciprocal_rank_fusion(
            bm25_results=
                bm25_results,

            vector_results=
                vector_results,
        )
    )

    # -----------------------------------------------------
    # Keep only chunks with text
    # -----------------------------------------------------
    fused_results = [
        result
        for result
        in fused_results
        if result.get("text")
    ]

    return fused_results[
        :top_k
    ]