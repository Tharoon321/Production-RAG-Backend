
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

from src.ingestion.chunker import tokenize


# ---------------------------------------------------------
# Location of saved BM25 index
# ---------------------------------------------------------
BM25_INDEX_PATH = Path("bm25_index.pkl")


# ---------------------------------------------------------
# Load BM25 index from disk
# ---------------------------------------------------------
def load_bm25_index():
    """
    Loads serialized BM25 index data.
    """

    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            "BM25 index file not found. "
            "Run ingestion pipeline first."
        )

    with open(BM25_INDEX_PATH, "rb") as file:
        return pickle.load(file)


# ---------------------------------------------------------
# Main BM25 retrieval function
# ---------------------------------------------------------
def bm25_search(
    query: str,
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    """
    Retrieves top matching chunk IDs using BM25.

    Returns:
        List of:
            (chunk_id, score)
    """

    bm25_data = load_bm25_index()

    bm25 = bm25_data["bm25"]
    chunk_ids = bm25_data["chunk_ids"]

    # Tokenize query exactly like document chunking
    tokenized_query = tokenize(query)

    # Compute BM25 relevance scores
    scores = bm25.get_scores(tokenized_query)

    # Pair chunk IDs with scores
    scored_results = list(zip(chunk_ids, scores))

    # Sort descending by score
    scored_results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    # Return top-N matches only
    return scored_results[:top_n]

