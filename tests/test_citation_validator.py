
import pytest
from pydantic import ValidationError

from src.generation.answer_model import (
    RAGResponse,
)


# ---------------------------------------------------------
# Test valid citations
# ---------------------------------------------------------
def test_valid_citations():
    """
    Ensures valid citations pass validation.
    """

    response = RAGResponse(
        answer="""
        FastAPI supports async APIs [doc_001].
        """,

        citations=["doc_001"],

        latency_ms=120.5,
    )

    assert response.citations == ["doc_001"]


# ---------------------------------------------------------
# Test invalid citations
# ---------------------------------------------------------
def test_missing_citations():
    """
    Ensures hallucinated citations fail validation.
    """

    with pytest.raises(ValidationError):

        RAGResponse(
            answer="""
            FastAPI supports async APIs [doc_999].
            """,

            citations=["doc_001"],

            latency_ms=120.5,
        )
