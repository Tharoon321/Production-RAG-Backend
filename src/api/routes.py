from __future__ import annotations

import time
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from src.ingestion.chunker import (
    Document,
    chunk_documents,
)

# ---------------------------------------------------------
# FastAPI router instance
# ---------------------------------------------------------
router = APIRouter()


# ---------------------------------------------------------
# Request model for /ask endpoint
# ---------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


# ---------------------------------------------------------
# Request model for /ingest endpoint
# ---------------------------------------------------------
class IngestRequest(BaseModel):
    documents: List[Document]


# ---------------------------------------------------------
# POST /ask
# ---------------------------------------------------------
@router.post("/ask")
async def ask_question(
    request: QueryRequest,
):
    """
    Full RAG pipeline.

    IMPORTANT:
    Heavy imports are lazy-loaded inside endpoint
    to reduce Heroku startup memory usage.
    """

    # -----------------------------------------------------
    # Lazy imports
    # -----------------------------------------------------
    from src.retrieval.hybrid import hybrid_search
    from src.reranking.reranker import rerank_chunks
    from src.generation.generator import generate_answer

    request_start = time.perf_counter()

    # -----------------------------------------------------
    # Hybrid retrieval
    # -----------------------------------------------------
    retrieved_candidates = await hybrid_search(
        query=request.question,
        top_k=30,
    )

    print("\n========== RETRIEVED ==========")
    print("COUNT:", len(retrieved_candidates))

    # -----------------------------------------------------
    # Reranking
    # -----------------------------------------------------
    reranked_chunks = rerank_chunks(
        query=request.question,
        candidates=retrieved_candidates,
        top_n=request.top_k,
    )

    print("\n========== RERANKED ==========")
    print("COUNT:", len(reranked_chunks))

    # -----------------------------------------------------
    # Final answer generation
    # -----------------------------------------------------
    response = generate_answer(
        query=request.question,
        reranked_chunks=reranked_chunks,
    )

    total_latency_ms = (
        time.perf_counter() - request_start
    ) * 1000

    response.latency_ms = total_latency_ms

    return response


# ---------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------
@router.post("/ingest")
async def ingest_documents(
    request: IngestRequest,
):
    """
    Ingest documents into RAG system.

    IMPORTANT:
    Heavy imports stay inside endpoint
    so FastAPI startup remains lightweight.
    """

    # -----------------------------------------------------
    # Lazy imports
    # -----------------------------------------------------
    from src.ingestion.embedder import embed_chunks
    from src.ingestion.indexer import index_chunks

    # -----------------------------------------------------
    # Chunk documents
    # -----------------------------------------------------
    chunks = chunk_documents(
        documents=request.documents,
    )

    # -----------------------------------------------------
    # Generate embeddings
    # -----------------------------------------------------
    embeddings = embed_chunks(chunks)

    # -----------------------------------------------------
    # Index chunks
    # -----------------------------------------------------
    index_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    return {
        "message": "Documents ingested successfully.",
        "documents": len(request.documents),
        "chunks": len(chunks),
    }


# ---------------------------------------------------------
# DEBUG endpoint
# ---------------------------------------------------------
@router.get("/debug/count")
def debug_count():
    """
    Returns total vectors stored in Qdrant.

    IMPORTANT:
    Lazy import prevents Qdrant client
    initialization during startup.
    """

    from src.ingestion.indexer import (
        qdrant_client,
        COLLECTION_NAME,
    )

    return qdrant_client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    )