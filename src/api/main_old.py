
from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router


# ---------------------------------------------------------
# Load environment variables from .env
# ---------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------
# Create FastAPI application
# ---------------------------------------------------------
app = FastAPI(
    title="Ask My Docs API",

    description="""
    Production-grade RAG backend using:
    - FastAPI
    - Qdrant
    - OpenAI
    - Cohere
    """,

    version="1.0.0",
)


# ---------------------------------------------------------
# Enable CORS
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,

    # Allow frontend requests from any origin during development
    allow_origins=["*"],

    allow_credentials=True,

    # Allow all HTTP methods
    allow_methods=["*"],

    # Allow all headers
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Register API routes
# ---------------------------------------------------------
app.include_router(router)


# ---------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------
@app.get("/")
async def root():
    """
    Basic health check endpoint.
    """

    return {
        "message": "Ask My Docs API is running."
    }
