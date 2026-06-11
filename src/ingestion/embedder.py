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
MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")

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

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in environment")

        genai.configure(api_key=api_key)

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