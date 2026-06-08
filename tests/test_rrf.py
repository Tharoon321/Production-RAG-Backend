
from src.retrieval.hybrid import (
    reciprocal_rank_fusion,
)


# ---------------------------------------------------------
# Test RRF score ordering
# ---------------------------------------------------------
def test_rrf_scores():
    """
    Ensures chunks ranked highly in both
    retrievers receive stronger RRF scores.
    """

    bm25_results = [
        ("chunk_a", 10.0),
        ("chunk_b", 9.0),
    ]

    vector_results = [
        (
            "chunk_a",
            0.95,
            "text a",
            "doc_001",
        ),
        (
            "chunk_c",
            0.92,
            "text c",
            "doc_002",
        ),
    ]

    results = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        vector_results=vector_results,
    )

    # chunk_a appears in BOTH rankings
    assert results[0]["chunk_id"] == "chunk_a"


# ---------------------------------------------------------
# Test deduplication
# ---------------------------------------------------------
def test_rrf_deduplication():
    """
    Ensures duplicate chunk IDs are merged.
    """

    bm25_results = [
        ("chunk_x", 5.0),
    ]

    vector_results = [
        (
            "chunk_x",
            0.99,
            "duplicate text",
            "doc_001",
        ),
    ]

    results = reciprocal_rank_fusion(
        bm25_results=bm25_results,
        vector_results=vector_results,
    )

    assert len(results) == 1
