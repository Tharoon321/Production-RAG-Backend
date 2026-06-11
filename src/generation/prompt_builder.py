from __future__ import annotations

from typing import Dict, List

from dotenv import load_dotenv

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()


def build_prompt(query: str, chunks: List[Dict]) -> List[Dict[str, str]]:
    """
    Build chat-style messages for the Gemini model.

    Returns a list of messages where each message is a dict
    with `role` and `content` keys. The generator expects
    messages in this format and will join them into a single
    prompt string.
    """

    system_instructions = (
        "You are a strict grounded QA assistant. Answer ONLY from the provided "
        "context chunks. If the context contains a direct statement relevant to "
        "the question, answer using that statement instead of saying you don't "
        "know. Keep the answer short and factual. Every factual statement must "
        "be followed by an inline citation like [doc_123]. If the context truly "
        "does not contain the answer, say exactly: I don't know."
    )

    # Build a context block containing each chunk with its doc id
    context_parts: List[str] = []
    for chunk in chunks:
        doc_id = chunk.get("doc_id", "unknown_doc")
        chunk_text = chunk.get("text", "")
        context_parts.append(f"[{doc_id}] {chunk_text}")

    if context_parts:
        context_block = "\n\n".join(context_parts)
    else:
        context_block = "(no context chunks were retrieved)"

    user_content = (
        "Context chunks:\n\n"
        f"{context_block}"
        "\n\nQuestion: "
        f"{query}"
    )

    # Ask the model to answer and include inline citations
    user_content += (
        "\n\nInstructions: Use the context directly. If the context says "
        '"FastAPI is great", answer "FastAPI is great." Cite it as [doc_001] '
        "or the matching doc id. Do not mention unsupported details."
    )

    return [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_content},
    ]