
from __future__ import annotations

from typing import Dict, List


# ---------------------------------------------------------
# System prompt for grounded RAG generation
# ---------------------------------------------------------
SYSTEM_PROMPT = """
You are a helpful AI assistant for document question answering.

Rules:
1. Only use the provided context.
2. Do not invent information.
3. If the answer is not in the context, say you are unsure.
4. Cite every factual claim using [doc_id] notation.
5. Use concise and accurate answers.
""".strip()


# ---------------------------------------------------------
# Build formatted context block
# ---------------------------------------------------------
def build_context(chunks: List[Dict]) -> str:
    """
    Formats retrieved chunks into context text.

    Example:

    [doc_001]
    FastAPI supports async APIs...

    [doc_002]
    JWT authentication uses signed tokens...
    """

    formatted_chunks = []

    for chunk in chunks:

        doc_id = chunk["doc_id"]
        text = chunk["text"]

        formatted_chunk = (
            f"[{doc_id}]\n"
            f"{text}"
        )

        formatted_chunks.append(formatted_chunk)

    return "\n\n".join(formatted_chunks)


# ---------------------------------------------------------
# Build final messages for OpenAI chat completion
# ---------------------------------------------------------
def build_prompt(
    query: str,
    chunks: List[Dict],
) -> List[Dict[str, str]]:
    """
    Builds chat completion message list.
    """

    context_block = build_context(chunks)

    user_prompt = f"""
Answer the following question using ONLY the provided context.

Question:
{query}

Context:
{context_block}
""".strip()

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
