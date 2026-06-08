
from __future__ import annotations

import time
from typing import Dict, List

from fastapi import APIRouter

from src.generation.generator import generate_answer
from src.ingestion.chunker import (
    Document,
    chunk_documents,
)
from src.ingestion.embedder import embed_chunks
from src.ingestion.indexer import (
    index_chunks,
    qdrant_client,
    COLLECTION_NAME,
)
from src.retrieval.hybrid import hybrid_search
from src.reranking.reranker import rerank_chunks

from pydantic import BaseModel, Field


# ---------------------------------------------------------
# FastAPI router instance
# ---------------------------------------------------------
router = APIRouter()


# ---------------------------------------------------------
# Request model for /ask endpoint
# ---------------------------------------------------------
class QueryRequest(BaseModel):
    """
    User question request model.
    """

    question: str

    # Number of reranked chunks to use
    top_k: int = 5


# ---------------------------------------------------------
# Request model for /ingest endpoint
# ---------------------------------------------------------
class IngestRequest(BaseModel):
    """
    Batch document ingestion request.
    """

    documents: List[Document]


# ---------------------------------------------------------
# POST /ask
# ---------------------------------------------------------
@router.post("/ask")
async def ask_question(
    request: QueryRequest,
):
    """
    Full RAG pipeline endpoint.

    Flow:
        retrieve
            ↓
        rerank
            ↓
        generate
    """

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

    for item in retrieved_candidates:
        print(item)

    # -----------------------------------------------------
    # Cross-encoder reranking
    # -----------------------------------------------------
    reranked_chunks = rerank_chunks(
        query=request.question,
        candidates=retrieved_candidates,
        top_n=request.top_k,
    )

    print("\n========== RERANKED ==========")
    print("COUNT:", len(reranked_chunks))

    for item in reranked_chunks:
        print(item)

    # -----------------------------------------------------
    # Final grounded answer generation
    # -----------------------------------------------------
    response = generate_answer(
        query=request.question,
        reranked_chunks=reranked_chunks,
    )

    total_latency_ms = (
        time.perf_counter() - request_start
    ) * 1000

    # Add total API latency
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
    Ingests documents into the RAG system.
    """

    # -----------------------------------------------------
    # Step 1: Chunk documents
    # -----------------------------------------------------
    chunks = chunk_documents(
        documents=request.documents,
    )

    # -----------------------------------------------------
    # Step 2: Generate embeddings
    # -----------------------------------------------------
    embeddings = embed_chunks(chunks)

    # -----------------------------------------------------
    # Step 3: Index into Qdrant + BM25
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
# DEBUG: Qdrant vector count
# ---------------------------------------------------------
@router.get("/debug/count")
def debug_count():
    return qdrant_client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    )