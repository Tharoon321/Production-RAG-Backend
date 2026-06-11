from __future__ import annotations

import time
from typing import List

import traceback
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends, Request
import json

from src.ingestion.chunker import (
    Document,
    chunk_documents,
)

# ---------------------------------------------------------
# FastAPI router instance
# ---------------------------------------------------------
router = APIRouter()


def require_json(request: Request) -> bool:
    """Dependency that ensures incoming request has application/json content-type."""
    ct = request.headers.get("content-type", "")
    if "application/json" not in ct.lower():
        raise HTTPException(
            status_code=415,
            detail="Content-Type must be application/json",
        )
    return True


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
    request_payload: QueryRequest,
    _content_check: bool = Depends(require_json),
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

    try:
        request_start = time.perf_counter()

        # -------------------------------------------------
        # Hybrid retrieval
        # -------------------------------------------------
        retrieved_candidates = await hybrid_search(
            query=request_payload.question,
            top_k=30,
        )

        print("\n========== RETRIEVED ==========")
        print("COUNT:", len(retrieved_candidates))

        # -------------------------------------------------
        # Reranking
        # -------------------------------------------------
        reranked_chunks = rerank_chunks(
            query=request_payload.question,
            candidates=retrieved_candidates,
            top_n=request_payload.top_k,
        )

        print("\n========== RERANKED ==========")
        print("COUNT:", len(reranked_chunks))

        # -------------------------------------------------
        # Final answer generation
        # -------------------------------------------------
        response = generate_answer(
            query=request_payload.question,
            reranked_chunks=reranked_chunks,
        )

        total_latency_ms = (
            time.perf_counter() - request_start
        ) * 1000

        response.latency_ms = total_latency_ms

        return response
    except Exception as exc:
        # Print full traceback to server logs for debugging
        tb = traceback.format_exc()
        print("ERROR in /ask:\n", tb)

        # Return a concise error to the client
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------
@router.post("/ingest")
async def ingest_documents(
    request_payload: IngestRequest,
    _content_check: bool = Depends(require_json),
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
        documents=request_payload.documents,
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
        "documents": len(request_payload.documents),
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

    from fastapi import HTTPException

    from src.ingestion.indexer import (
        COLLECTION_NAME,
        get_qdrant_client,
    )

    try:
        qdrant_client = get_qdrant_client()
        result = qdrant_client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        )
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Qdrant unavailable: {str(exc)}",
        )


# ---------------------------------------------------------
# DEBUG: echo request
# ---------------------------------------------------------
@router.post("/debug/echo")
async def debug_echo(request: Request):
    """Return received headers and raw body to help debug client requests."""

    raw = await request.body()

    try:
        parsed = json.loads(raw.decode("utf-8")) if raw else None
    except Exception:
        parsed = None

    return {
        "headers": dict(request.headers),
        "raw_body": raw.decode("utf-8", errors="replace"),
        "parsed_json": parsed,
    }


@router.get("/debug/genai")
def debug_genai():
    """Test Gemini / Google Generative API embedding call and return result or traceback."""
    try:
        from src.retrieval.vector_retriever import get_genai, MODEL

        genai = get_genai()

        # Try a small embedding call
        resp = genai.embed_content(model=MODEL, content="test", task_type="retrieval_query")

        return {"ok": True, "embedding_len": len(resp.get("embedding", []))}
    except Exception as exc:
        import traceback as _tb

        return {"ok": False, "error": str(exc), "traceback": _tb.format_exc()}


@router.get("/debug/cohere")
def debug_cohere():
    """Test Cohere rerank API and return result or traceback."""
    try:
        from src.reranking.reranker import get_cohere_client, RERANK_MODEL

        client = get_cohere_client()

        # Minimal rerank call
        response = client.rerank(model=RERANK_MODEL, query="q", documents=["a","b"], top_n=1)

        return {"ok": True, "results_count": len(response.results)}
    except Exception as exc:
        import traceback as _tb

        return {"ok": False, "error": str(exc), "traceback": _tb.format_exc()}


@router.get("/debug/qdrant")
def debug_qdrant():
    """Test Qdrant client connection and count a collection."""
    try:
        from src.ingestion.indexer import get_qdrant_client, COLLECTION_NAME

        client = get_qdrant_client()
        result = client.count(collection_name=COLLECTION_NAME, exact=True)

        return {"ok": True, "count": result}
    except Exception as exc:
        import traceback as _tb

        return {"ok": False, "error": str(exc), "traceback": _tb.format_exc()}