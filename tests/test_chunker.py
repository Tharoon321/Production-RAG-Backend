
from src.ingestion.chunker import (
    Document,
    chunk_documents,
    tokenize,
)


# ---------------------------------------------------------
# Test chunk size limits
# ---------------------------------------------------------
def test_chunk_size_limit():
    """
    Ensures chunks never exceed max_tokens.
    """

    text = "hello " * 1200

    documents = [
        Document(
            id="doc_001",
            text=text,
        )
    ]

    chunks = chunk_documents(
        documents=documents,
        max_tokens=512,
        overlap=50,
    )

    for chunk in chunks:

        token_count = len(
            tokenize(chunk.text)
        )

        assert token_count <= 512


# ---------------------------------------------------------
# Test overlap correctness
# ---------------------------------------------------------
def test_chunk_overlap():
    """
    Ensures overlap tokens are preserved
    between consecutive chunks.
    """

    text = " ".join(
        [f"token{i}" for i in range(1000)]
    )

    documents = [
        Document(
            id="doc_001",
            text=text,
        )
    ]

    chunks = chunk_documents(
        documents=documents,
        max_tokens=100,
        overlap=10,
    )

    first_chunk_tokens = tokenize(chunks[0].text)

    second_chunk_tokens = tokenize(chunks[1].text)

    # Last 10 tokens of chunk 1
    overlap_1 = first_chunk_tokens[-10:]

    # First 10 tokens of chunk 2
    overlap_2 = second_chunk_tokens[:10]

    assert overlap_1 == overlap_2


# ---------------------------------------------------------
# Test document ID preservation
# ---------------------------------------------------------
def test_doc_id_preserved():
    """
    Ensures chunk keeps original parent doc_id.
    """

    documents = [
        Document(
            id="doc_abc",
            text="hello world " * 300,
        )
    ]

    chunks = chunk_documents(documents)

    for chunk in chunks:
        assert chunk.doc_id == "doc_abc"
