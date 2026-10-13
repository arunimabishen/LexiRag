"""LexiRAG API: one endpoint around generation/pipeline.answer().

Run from the project root:
    uvicorn api.main:app --port 8000
Docs page (for the frontend teammate): http://localhost:8000/docs
"""
import os
import sys
import threading
from contextlib import asynccontextmanager

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

# Defaults so the env vars are not needed in every terminal.
os.environ.setdefault("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
os.environ.setdefault("RERANK_POOL", "20")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from generation.pipeline import answer as run_pipeline

_lock = threading.Lock()  # one question at a time: models are cached per process


@asynccontextmanager
async def lifespan(app):
    # Load the embedding model and cross-encoder once, at startup,
    # so the first real question is not slow.
    try:
        from retrieval.hybrid_search import hybrid_search
        hybrid_search("warm up", top_k=1, rerank=True)
    except Exception as e:  # server still starts; first question will load lazily
        print("Warm-up skipped:", e)
    yield


app = FastAPI(title="LexiRAG API", version="0.1", lifespan=lifespan)

# Open for local development. Before deployment, replace "*" with the
# frontend's real address, e.g. ["http://localhost:5173"].
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question.")
    try:
        with _lock:
            return run_pipeline(q)
    except Exception as e:
        print("Pipeline error:", repr(e))
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again.")