# src/tasks/worker_rq.py
from __future__ import annotations
import os
from redis import Redis
from rq import Queue
from rq.job import Job
from typing import Dict, Any

# Redis queue helper used by web code
redis_conn = Redis.from_url(os.getenv("REDIS_URL"))
queue = Queue("default", connection=redis_conn)

# Worker job function
def process_ask_job(query: str, top_k: int = 5) -> Dict[str, Any]:
    # Lazy imports happen inside the job to keep web dyno small
    from src.retrieval.hybrid import hybrid_search
    from src.reranking.reranker import rerank_chunks
    from src.generation.generator import generate_answer

    # If hybrid_search is async you can use asyncio.run(...)
    retrieved_candidates = hybrid_search(query=query, top_k=30)
    reranked_chunks = rerank_chunks(query=query, candidates=retrieved_candidates, top_n=top_k)
    response = generate_answer(query=query, reranked_chunks=reranked_chunks)

    return {
        "answer": response.answer,
        "citations": response.citations,
        "latency_ms": response.latency_ms,
    }