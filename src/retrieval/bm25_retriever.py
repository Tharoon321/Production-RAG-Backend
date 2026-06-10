from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

from src.ingestion.chunker import (
    tokenize,
)

# ---------------------------------------------------------
# BM25 index file
# ---------------------------------------------------------
BM25_INDEX_PATH = Path(
    "bm25_index.pkl"
)

# ---------------------------------------------------------
# Lazy-loaded cached BM25 data
# ---------------------------------------------------------
_bm25_data = None


# ---------------------------------------------------------
# Load BM25 index lazily
# ---------------------------------------------------------
def load_bm25_index():
    """
    Loads BM25 index only once
    and caches it in memory.
    """

    global _bm25_data

    # -----------------------------------------------------
    # Return cached object
    # -----------------------------------------------------
    if _bm25_data is not None:
        return _bm25_data

    # -----------------------------------------------------
    # Validate existence
    # -----------------------------------------------------
    if not BM25_INDEX_PATH.exists():

        raise FileNotFoundError(
            "BM25 index file not found. "
            "Run ingestion pipeline first."
        )

    # -----------------------------------------------------
    # Load from disk
    # -----------------------------------------------------
    with open(
        BM25_INDEX_PATH,
        "rb",
    ) as file:

        _bm25_data = pickle.load(
            file
        )

    return _bm25_data


# ---------------------------------------------------------
# BM25 retrieval
# ---------------------------------------------------------
def bm25_search(
    query: str,
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    """
    Retrieves top BM25 matches.

    Returns:
        [
            (chunk_id, score)
        ]
    """

    # -----------------------------------------------------
    # Lazy-load cached BM25
    # -----------------------------------------------------
    bm25_data = load_bm25_index()

    bm25 = bm25_data["bm25"]

    chunk_ids = bm25_data[
        "chunk_ids"
    ]

    # -----------------------------------------------------
    # Tokenize query
    # -----------------------------------------------------
    tokenized_query = tokenize(
        query
    )

    # -----------------------------------------------------
    # Compute scores
    # -----------------------------------------------------
    scores = bm25.get_scores(
        tokenized_query
    )

    # -----------------------------------------------------
    # Pair IDs with scores
    # -----------------------------------------------------
    scored_results = list(
        zip(chunk_ids, scores)
    )

    # -----------------------------------------------------
    # Sort descending
    # -----------------------------------------------------
    scored_results.sort(
        key=lambda item: item[1],

        reverse=True,
    )

    # -----------------------------------------------------
    # Return top results
    # -----------------------------------------------------
    return scored_results[:top_n]