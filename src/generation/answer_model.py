from __future__ import annotations

import re
from typing import List

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

# ---------------------------------------------------------
# Citation extraction regex
# ---------------------------------------------------------
CITATION_PATTERN = r"\[(doc_[^\]]+)\]"


# ---------------------------------------------------------
# Final validated RAG response model
# ---------------------------------------------------------
class RAGResponse(BaseModel):
    """
    Final API response returned to frontend.
    """

    # Generated grounded answer
    answer: str

    # Extracted citation document IDs
    citations: List[str] = Field(
        default_factory=list
    )

    # Total request latency
    latency_ms: float

    # -----------------------------------------------------
    # Citation consistency validator
    # -----------------------------------------------------
    @field_validator("citations")
    @classmethod
    def validate_citations(
        cls,
        citations: List[str],
        info,
    ) -> List[str]:
        """
        Ensures every citation present
        in the answer also exists
        inside citations list.
        """

        answer = info.data.get(
            "answer",
            "",
        )

        # Extract citations from answer text
        found_citations = re.findall(
            CITATION_PATTERN,
            answer,
        )

        # Remove duplicates
        found_citations = list(
            set(found_citations)
        )

        # Detect missing citations
        missing = [
            citation
            for citation in found_citations
            if citation not in citations
        ]

        if missing:
            raise ValueError(
                "Missing citations in citations list: "
                f"{missing}"
            )

        return citations