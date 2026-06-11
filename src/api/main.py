from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------
# Import router AFTER env loading
# ---------------------------------------------------------
from src.api.routes import router



# ---------------------------------------------------------
# Create FastAPI app
# ---------------------------------------------------------
app = FastAPI(
    title="Ask My Docs API",

    description="""
    Production-grade RAG backend using:
    - FastAPI
    - Qdrant
    - Gemini
    - Cohere
    """,

    version="1.0.0",
)

# ---------------------------------------------------------
# Enable CORS
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)

# ---------------------------------------------------------
# Register routes
# ---------------------------------------------------------
app.include_router(router)

# ---------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------
@app.get("/")
async def root():

    return {
        "status": "healthy",
        "message": "Ask My Docs API is running"
    }

# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------
@app.get("/health")
async def health():

    return {
        "status": "ok"
    }
import os
import psutil

@app.get("/memory")
async def memory():
    process = psutil.Process(os.getpid())

    return {
        "rss_mb": round(
            process.memory_info().rss / 1024 / 1024,
            2
        )
    }
import os
import psutil

process = psutil.Process(os.getpid())

print("=" * 50)
print(
    f"STARTUP MEMORY: "
    f"{process.memory_info().rss / 1024 / 1024:.2f} MB"
)
print("=" * 50)