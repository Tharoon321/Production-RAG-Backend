from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Dict, List

from src.retrieval.bm25_retriever import bm25_search
from src.retrieval.vector_retriever import vector_search

RRF_K = 60


def reciprocal_rank_fusion(
    bm25_results,
    vector_results,
) -> List[Dict]:

    fused_scores = defaultdict(float)
    chunk_metadata = {}

    # BM25 results
    for rank, (chunk_id, score) in enumerate(
        bm25_results,
        start=1,
    ):

        fused_scores[chunk_id] += (
            1 / (RRF_K + rank)
        )

        if chunk_id not in chunk_metadata:
            chunk_metadata[chunk_id] = {}

        chunk_metadata[chunk_id].update(
            {
                "chunk_id": chunk_id,
                "bm25_score": float(score),
            }
        )

    # Vector results
    for rank, result in enumerate(
        vector_results,
        start=1,
    ):

        chunk_id = result["chunk_id"]

        fused_scores[chunk_id] += (
            1 / (RRF_K + rank)
        )

        if chunk_id not in chunk_metadata:
            chunk_metadata[chunk_id] = {}

        chunk_metadata[chunk_id].update(
            {
                "chunk_id": result["chunk_id"],
                "doc_id": result["doc_id"],
                "text": result["text"],
                "vector_score": float(
                    result["score"]
                ),
            }
        )

    fused_results = []

    for chunk_id, rrf_score in fused_scores.items():

        metadata = chunk_metadata.get(
            chunk_id,
            {}
        )

        metadata["rrf_score"] = float(
            rrf_score
        )

        fused_results.append(
            metadata
        )

    fused_results.sort(
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    return fused_results


async def async_bm25_search(
    query: str,
):
    return bm25_search(query=query)


async def async_vector_search(
    query: str,
):
    return vector_search(query=query)


async def hybrid_search(
    query: str,
    top_k: int = 30,
):

    bm25_results, vector_results = await asyncio.gather(
        async_bm25_search(query),
        async_vector_search(query),
    )

    print("\n========== BM25 ==========")
    print(bm25_results)

    print("\n========== VECTOR ==========")
    print(vector_results)

    fused_results = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        vector_results=vector_results,
    )

    print("\n========== FUSED ==========")
    print(fused_results)

    # IMPORTANT:
    # Only keep results that actually contain text.
    fused_results = [
        result
        for result in fused_results
        if result.get("text")
    ]

    return fused_results[:top_k]