from __future__ import annotations

import os
from typing import List, Tuple

from dotenv import load_dotenv

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------
# Gemini embedding model
# ---------------------------------------------------------
MODEL = "models/text-embedding-004"

# ---------------------------------------------------------
# Lazy-loaded Gemini module
# ---------------------------------------------------------
_genai = None


def get_genai():
    """
    Lazy loads Gemini SDK only when needed.
    """

    global _genai

    if _genai is None:

        import google.generativeai as genai

        genai.configure(
            api_key=os.getenv(
                "GOOGLE_API_KEY"
            )
        )

        _genai = genai

    return _genai


# ---------------------------------------------------------
# Generate embeddings
# ---------------------------------------------------------
def embed_chunks(
    chunks,
) -> List[Tuple[str, List[float]]]:
    """
    Generates Gemini embeddings for chunks.
    """

    genai = get_genai()

    embeddings = []

    for chunk in chunks:

        response = genai.embed_content(
            model=MODEL,

            content=chunk.text,

            task_type="retrieval_document",
        )

        embeddings.append(
            (
                chunk.chunk_id,
                response["embedding"],
            )
        )

    return embeddings