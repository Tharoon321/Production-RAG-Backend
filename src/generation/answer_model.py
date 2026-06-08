
from __future__ import annotations

import re
from typing import List

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------
# Regex pattern for citation extraction
# ---------------------------------------------------------
CITATION_PATTERN = r"\[(doc_[^\]]+)\]"


# ---------------------------------------------------------
# Final validated RAG response model
# ---------------------------------------------------------
class RAGResponse(BaseModel):
    """
    Final API response returned to frontend.
    """

    answer: str

    # List of cited document IDs
    citations: List[str] = Field(default_factory=list)

    # End-to-end request latency
    latency_ms: float

    # -----------------------------------------------------
    # Validate citation consistency
    # -----------------------------------------------------
    @field_validator("citations")
    @classmethod
    def validate_citations(
        cls,
        citations: List[str],
        info,
    ) -> List[str]:
        """
        Ensures every citation used in the answer
        exists in the citations list.
        """

        answer = info.data.get("answer", "")

        # Extract all citations from answer text
        found_citations = re.findall(
            CITATION_PATTERN,
            answer,
        )

        # Remove duplicates
        found_citations = list(set(found_citations))

        # Validate every citation exists
        missing = [
            citation
            for citation in found_citations
            if citation not in citations
        ]

        if missing:
            raise ValueError(
                "Answer contains citations not present "
                f"in citations list: {missing}"
            )

        return citations
