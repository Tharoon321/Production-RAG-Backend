
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk, tokenize


# ---------------------------------------------------------
# Qdrant collection configuration
# ---------------------------------------------------------
COLLECTION_NAME = "ask_my_docs"

# OpenAI text-embedding-3-small output dimension
VECTOR_DIMENSION = 384

# BM25 index save location
BM25_INDEX_PATH = Path("bm25_index.pkl")


# ---------------------------------------------------------
# Initialize Qdrant client
# ---------------------------------------------------------
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)



# ---------------------------------------------------------
# Create Qdrant collection
# ---------------------------------------------------------
def create_collection() -> None:
    """
    Creates a fresh Qdrant collection for vector storage.
    """

    # -----------------------------------------------------
    # Delete old collection if it already exists
    #
    # IMPORTANT:
    # We switched from OpenAI embeddings (1536 dims)
    # to Gemini embeddings (768 dims).
    #
    # Old collections with wrong dimensions must be removed.
    # -----------------------------------------------------
    try:

        qdrant_client.delete_collection(
            collection_name=COLLECTION_NAME
        )

        print(
            f"Deleted existing collection: {COLLECTION_NAME}"
        )

    except Exception:

        # Ignore if collection does not exist
        pass


    # -----------------------------------------------------
    # Create fresh collection
    # -----------------------------------------------------
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(

            # Gemini embedding dimension
            size=VECTOR_DIMENSION,

            # Cosine similarity for semantic search
            distance=Distance.COSINE,
        ),
    )

    print(
        f"Created collection: {COLLECTION_NAME}"
    )


# ---------------------------------------------------------
# Upload vectors into Qdrant
# ---------------------------------------------------------
def upsert_vectors(
    chunks: List[Chunk],
    embeddings: List[Tuple[str, List[float]]],
) -> None:
    """
    Uploads chunk embeddings into Qdrant.
    """

    # Build quick lookup:
    # chunk_id -> embedding vector
    embedding_map = {
        chunk_id: vector
        for chunk_id, vector in embeddings
    }

    points: List[PointStruct] = []

    for chunk in chunks:

        vector = embedding_map.get(chunk.chunk_id)

        # Skip chunks without embeddings
        if vector is None:
            continue

        point = PointStruct(
            id=chunk.chunk_id,

            vector=vector,

            # Payload is searchable metadata
            payload={
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            },
        )

        points.append(point)

    # Upload all points into Qdrant
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )


# ---------------------------------------------------------
# Build BM25 keyword index
# ---------------------------------------------------------
def build_bm25_index(
    chunks: List[Chunk],
) -> None:
    """
    Builds BM25 index over all chunk texts.
    """

    # Tokenized corpus required by BM25
    corpus = [
        tokenize(chunk.text)
        for chunk in chunks
    ]

    # Create BM25 search index
    bm25 = BM25Okapi(corpus)

    # Save both:
    # 1. BM25 model
    # 2. chunk lookup order
    bm25_data = {
        "bm25": bm25,
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
        "chunks": chunks,
    }

    # Persist index to disk
    with open(BM25_INDEX_PATH, "wb") as file:
        pickle.dump(bm25_data, file)


# ---------------------------------------------------------
# Full indexing pipeline
# ---------------------------------------------------------
def index_chunks(
    chunks: List[Chunk],
    embeddings: List[Tuple[str, List[float]]],
) -> None:
    """
    Full indexing pipeline:
        1. Create Qdrant collection
        2. Upload vectors
        3. Build BM25 index
    """

    create_collection()

    upsert_vectors(
        chunks=chunks,
        embeddings=embeddings,
    )

    build_bm25_index(chunks)

