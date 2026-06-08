
from __future__ import annotations
from dotenv import load_dotenv

load_dotenv()
import os
import re
import time
from typing import Dict, List

import google.generativeai as genai

from src.generation.answer_model import (
    CITATION_PATTERN,
    RAGResponse,
)

from src.generation.prompt_builder import (
    build_prompt,
)


# ---------------------------------------------------------
# Configure Gemini
# ---------------------------------------------------------
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ---------------------------------------------------------
# Gemini model
# ---------------------------------------------------------
MODEL_NAME = "gemini-2.5-flash"


# ---------------------------------------------------------
# Extract citations
# ---------------------------------------------------------
def extract_citations(answer: str) -> List[str]:

    citations = re.findall(
        CITATION_PATTERN,
        answer,
    )

    return list(dict.fromkeys(citations))


# ---------------------------------------------------------
# Generate grounded answer
# ---------------------------------------------------------
def generate_answer(
    query: str,
    reranked_chunks: List[Dict],
) -> RAGResponse:
    """
    Generates final RAG answer using Gemini.
    """

    start_time = time.perf_counter()

    messages = build_prompt(
        query=query,
        chunks=reranked_chunks,
    )

    # Convert chat messages into single prompt
    prompt = "\n\n".join(
        [
            f"{msg['role'].upper()}:\n{msg['content']}"
            for msg in messages
        ]
    )

    model = genai.GenerativeModel(
        MODEL_NAME
    )

    response = model.generate_content(
        prompt
    )

    answer = response.text.strip()

    citations = extract_citations(
        answer
    )

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    validated_response = RAGResponse(
        answer=answer,
        citations=citations,
        latency_ms=latency_ms,
    )

    return validated_response
