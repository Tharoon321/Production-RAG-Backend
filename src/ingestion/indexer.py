from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Tuple

from src.ingestion.chunker import (
    Chunk,
    tokenize,
)

# ---------------------------------------------------------
# Qdrant collection configuration
# ---------------------------------------------------------
COLLECTION_NAME = "ask_my_docs"

# Gemini embedding dimension for models/gemini-embedding-2
VECTOR_DIMENSION = 3072

# BM25 index file
BM25_INDEX_PATH = Path(
    "bm25_index.pkl"
)

# ---------------------------------------------------------
# Lazy-loaded Qdrant client
# ---------------------------------------------------------
_qdrant_client = None


def get_qdrant_client():
    """
    Lazy loads Qdrant client.
    Prevents startup RAM spikes.
    """

    global _qdrant_client

    if _qdrant_client is None:

        from qdrant_client import (
            QdrantClient,
        )
        
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        
        if url:
            url = url.strip().replace("\n", "").replace("\r", "")
        if api_key:
            api_key = api_key.strip().replace("\n", "").replace("\r", "")

        _qdrant_client = QdrantClient(
            url=url,
            api_key=api_key,
        )
        

    return _qdrant_client


# ---------------------------------------------------------
# Create Qdrant collection
# ---------------------------------------------------------
def create_collection() -> None:
    """
    Creates fresh Qdrant collection.
    """

    from qdrant_client.models import (
        Distance,
        VectorParams,
    )

    qdrant_client = get_qdrant_client()

    # -----------------------------------------------------
    # Delete existing collection
    # -----------------------------------------------------
    try:

        qdrant_client.delete_collection(
            collection_name=COLLECTION_NAME
        )

        print(
            f"Deleted collection: "
            f"{COLLECTION_NAME}"
        )

    except Exception:
        pass

    # -----------------------------------------------------
    # Create fresh collection
    # -----------------------------------------------------
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(
            size=VECTOR_DIMENSION,

            distance=Distance.COSINE,
        ),
    )

    print(
        f"Created collection: "
        f"{COLLECTION_NAME}"
    )


# ---------------------------------------------------------
# Upload vectors
# ---------------------------------------------------------
def upsert_vectors(
    chunks: List[Chunk],
    embeddings: List[
        Tuple[str, List[float]]
    ],
) -> None:
    """
    Uploads vectors into Qdrant.
    """

    from qdrant_client.models import (
        PointStruct,
    )

    qdrant_client = get_qdrant_client()

    # -----------------------------------------------------
    # Build embedding lookup
    # -----------------------------------------------------
    embedding_map = {
        chunk_id: vector
        for chunk_id, vector
        in embeddings
    }

    points = []

    for chunk in chunks:

        vector = embedding_map.get(
            chunk.chunk_id
        )

        if vector is None:
            continue

        point = PointStruct(
            id=chunk.chunk_id,

            vector=vector,

            payload={
                "doc_id": chunk.doc_id,

                "chunk_id":
                    chunk.chunk_id,

                "text": chunk.text,

                "metadata":
                    chunk.metadata,
            },
        )

        points.append(point)

    # -----------------------------------------------------
    # Upload to Qdrant
    # ----------
    #-------------------------------------------
    
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,

        points=points,
    )


# ---------------------------------------------------------
# Build BM25 index
# ---------------------------------------------------------
def build_bm25_index(
    chunks: List[Chunk],
) -> None:
    """
    Builds BM25 index.
    """

    from rank_bm25 import (
        BM25Okapi,
    )

    # -----------------------------------------------------
    # Tokenize corpus
    # -----------------------------------------------------
    corpus = [
        tokenize(chunk.text)
        for chunk in chunks
    ]

    # -----------------------------------------------------
    # Build BM25
    # -----------------------------------------------------
    bm25 = BM25Okapi(corpus)

    bm25_data = {
        "bm25": bm25,

        "chunk_ids": [
            chunk.chunk_id
            for chunk in chunks
        ],

        "chunks": chunks,
    }

    # -----------------------------------------------------
    # Persist index
    # -----------------------------------------------------
    with open(
        BM25_INDEX_PATH,
        "wb",
    ) as file:

        pickle.dump(
            bm25_data,
            file,
        )


# ---------------------------------------------------------
# Full indexing pipeline
# ---------------------------------------------------------
def index_chunks(
    chunks: List[Chunk],
    embeddings: List[
        Tuple[str, List[float]]
    ],
) -> None:
    """
    Full indexing pipeline.
    """

    create_collection()

    upsert_vectors(
        chunks=chunks,

        embeddings=embeddings,
    )

    build_bm25_index(
        chunks
    )
