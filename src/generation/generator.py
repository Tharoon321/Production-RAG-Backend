from __future__ import annotations

import os
import re
import time
from typing import Dict, List

from dotenv import load_dotenv

from src.generation.answer_model import (
    CITATION_PATTERN,
    RAGResponse,
)

from src.generation.prompt_builder import (
    build_prompt,
)

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------
# Gemini model configuration
# ---------------------------------------------------------
MODEL_NAME = "gemini-2.5-flash"

# ---------------------------------------------------------
# Lazy-loaded Gemini model
# ---------------------------------------------------------
_model = None


def get_gemini_model():
    """
    Lazy loads Gemini model only when needed.
    Prevents heavy startup memory usage.
    """

    global _model

    if _model is None:

        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in environment")

        genai.configure(api_key=api_key)

        _model = genai.GenerativeModel(MODEL_NAME)

    return _model


# ---------------------------------------------------------
# Extract citations
# ---------------------------------------------------------
def extract_citations(
    answer: str,
) -> List[str]:

    citations = re.findall(
        CITATION_PATTERN,
        answer,
    )

    # Remove duplicates while preserving order
    return list(
        dict.fromkeys(citations)
    )


# ---------------------------------------------------------
# Generate grounded answer
# ---------------------------------------------------------
def generate_answer(
    query: str,
    reranked_chunks: List[Dict],
) -> RAGResponse:
    """
    Generates final grounded answer using Gemini.
    """

    start_time = time.perf_counter()

    # -----------------------------------------------------
    # Build prompt
    # -----------------------------------------------------
    messages = build_prompt(
        query=query,
        chunks=reranked_chunks,
    )

    # -----------------------------------------------------
    # Convert chat messages into single prompt
    # -----------------------------------------------------
    prompt = "\n\n".join(
        [
            f"{msg['role'].upper()}:\n{msg['content']}"
            for msg in messages
        ]
    )

    # -----------------------------------------------------
    # Lazy-load Gemini model
    # -----------------------------------------------------
    model = get_gemini_model()

    # -----------------------------------------------------
    # Generate response
    # -----------------------------------------------------
    response = model.generate_content(
        prompt
    )

    answer = response.text.strip()

    # -----------------------------------------------------
    # Extract citations
    # -----------------------------------------------------
    citations = extract_citations(
        answer
    )

    # -----------------------------------------------------
    # Compute latency
    # -----------------------------------------------------
    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    # -----------------------------------------------------
    # Validate response
    # -----------------------------------------------------
    validated_response = RAGResponse(
        answer=answer,
        citations=citations,
        latency_ms=latency_ms,
    )

    return validated_response