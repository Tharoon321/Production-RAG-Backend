from __future__ import annotations

import re
from typing import List
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Input document model
# ---------------------------------------------------------
class Document(BaseModel):
    """
    Represents a raw document before chunking.
    """

    id: str
    text: str

    # Metadata can store:
    # source filename, author, upload timestamp, etc.
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------
# Output chunk model
# ---------------------------------------------------------
class Chunk(BaseModel):
    """
    Represents a chunk created from a document.
    """

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------
# Simple tokenizer
# ---------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """
    Splits text into tokens.

    We use a simple regex tokenizer instead of a heavy tokenizer
    library to keep the system lightweight and cloud-friendly.
    """

    # Extracts words while ignoring punctuation
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------
# Convert tokens back into readable text
# ---------------------------------------------------------
def detokenize(tokens: List[str]) -> str:
    """
    Converts token list back into text.
    """

    return " ".join(tokens)


# ---------------------------------------------------------
# Main chunking function
# ---------------------------------------------------------
def chunk_documents(
    documents: List[Document],
    max_tokens: int = 512,
    overlap: int = 50,
) -> List[Chunk]:
    """
    Splits documents into overlapping chunks.

    Example:
        max_tokens = 512
        overlap = 50

    Chunk layout:
        Chunk 1 -> tokens 0-511
        Chunk 2 -> tokens 462-973
    """

    chunks: List[Chunk] = []

    # Prevent invalid overlap configurations
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")

    for document in documents:

        # Convert document text into tokens
        tokens = tokenize(document.text)

        start = 0

        # Continue until all tokens are processed
        while start < len(tokens):

            # Define chunk boundaries
            end = start + max_tokens

            # Extract token window
            chunk_tokens = tokens[start:end]

            # Convert tokens back into text
            chunk_text = detokenize(chunk_tokens)

            # Create chunk object
            chunk = Chunk(
                chunk_id=str(uuid4()),

                # Preserve parent document relationship
                doc_id=document.id,

                text=chunk_text,

                # Copy metadata so retrieval can use it later
                metadata=document.metadata.copy(),
            )

            chunks.append(chunk)

            # Move forward while preserving overlap
            start += max_tokens - overlap

    return chunks

