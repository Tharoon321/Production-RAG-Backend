from __future__ import annotations

import time
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.ingestion.chunker import (
    Document,
    chunk_documents,
)

# RQ enqueue + status helpers
from src.tasks.worker_rq import queue, redis_conn
from rq.job import Job

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
    Enqueue the full RAG pipeline as a background job.

    Returns a job id and a status URL the client can poll.
    Heavy imports are executed inside the worker process.
    """

    # Enqueue the worker job by import path. The worker function
    # should be defined as src.tasks.worker_rq.process_ask_job
    job = queue.enqueue(
        "src.tasks.worker_rq.process_ask_job",
        request.question,
        request.top_k,
    )

    return {
        "job_id": job.get_id(),
        "status_url": f"/ask/status/{job.get_id()}",
    }


# ---------------------------------------------------------
# GET /ask/status/{job_id}
# ---------------------------------------------------------
@router.get("/ask/status/{job_id}")
def ask_status(job_id: str):
    """
    Returns job status and result (when finished).
    """

    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.get_status()

    if job.is_finished:
        return {
            "status": "finished",
            "result": job.result,
        }

    if job.is_failed:
        return {
            "status": "failed",
            "error": str(job.exc_info),
        }

    return {
        "status": status,
    }


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