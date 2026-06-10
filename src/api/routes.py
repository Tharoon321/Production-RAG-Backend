from __future__ import annotations

import time
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from src.generation.generator import generate_answer
from src.ingestion.chunker import (
    Document,
    chunk_documents,
)
from src.ingestion.indexer import (
    qdrant_client,
    COLLECTION_NAME,
)
from src.retrieval.hybrid import hybrid_search
from src.reranking.reranker import rerank_chunks


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

    request_start = time.perf_counter()

    retrieved_candidates = await hybrid_search(
        query=request.question,
        top_k=30,
    )

    print("\n========== RETRIEVED ==========")
    print("COUNT:", len(retrieved_candidates))

    for item in retrieved_candidates:
        print(item)

    reranked_chunks = rerank_chunks(
        query=request.question,
        candidates=retrieved_candidates,
        top_n=request.top_k,
    )

    print("\n========== RERANKED ==========")
    print("COUNT:", len(reranked_chunks))

    for item in reranked_chunks:
        print(item)

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
    Ingests documents into the RAG system.

    IMPORTANT:
    Heavy imports are done here so they
    don't load during FastAPI startup.
    """

    from src.ingestion.embedder import embed_chunks
    from src.ingestion.indexer import index_chunks

    chunks = chunk_documents(
        documents=request.documents,
    )

    embeddings = embed_chunks(chunks)

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