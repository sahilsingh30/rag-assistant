"""
FastAPI application — RAG Documentation Assistant

Endpoints:
  POST /query        Submit a question
  POST /ingest       Ingest files or URLs
  GET  /documents    List indexed documents
  POST /feedback     Submit thumbs up/down feedback
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

from app.ingestion import ingest_files, ingest_urls
from app.retriever import list_documents
from app.workflow import rag_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Documentation Assistant",
    description=(
        "Self-corrective RAG pipeline powered by LangGraph. "
        "Retrieves, grades, and generates answers from technical documentation."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory feedback store (swap for a DB in production)
FEEDBACK_STORE: List[dict] = []


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Natural language question")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    query_type: str
    rewritten_query: str
    retry_count: int
    latency_ms: float


class IngestURLRequest(BaseModel):
    urls: List[HttpUrl] = Field(..., min_length=1, description="URLs to fetch and ingest")


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_added: int
    avg_chunk_size: Optional[int] = None
    errors: Optional[List[str]] = None


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int = Field(..., ge=0, le=1, description="1 = thumbs up, 0 = thumbs down")
    comment: Optional[str] = Field(None, max_length=500)


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: int


class DocumentSummary(BaseModel):
    source: str
    chunks: int


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "RAG Documentation Assistant"}


@app.get("/health", tags=["health"])
def health():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query(request: QueryRequest):
    """
    Submit a natural language question.
    The LangGraph workflow handles query analysis, retrieval,
    document grading, and answer generation automatically.
    """
    if not request.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")

    logger.info(f"Query received: {request.question!r}")
    t0 = time.perf_counter()

    try:
        initial_state = {
            "question": request.question,
            "rewritten_query": "",
            "query_type": "",
            "documents": [],
            "relevant_documents": [],
            "answer": "",
            "retry_count": 0,
            "sources": [],
        }
        result = rag_graph.invoke(initial_state)
    except Exception as exc:
        logger.exception("Workflow error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow error: {exc}",
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Query answered in {latency_ms:.0f}ms, retries={result.get('retry_count', 0)}")

    return QueryResponse(
        question=request.question,
        answer=result.get("answer", "No answer generated."),
        sources=result.get("sources", []),
        query_type=result.get("query_type", "unknown"),
        rewritten_query=result.get("rewritten_query", request.question),
        retry_count=result.get("retry_count", 0),
        latency_ms=round(latency_ms, 2),
    )


@app.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest(
    urls: Optional[str] = None,         # JSON-encoded list of URLs (form field)
    files: List[UploadFile] = File(default=[]),
):
    """
    Ingest new documents into the vector store.
    - Send files as multipart/form-data file uploads, OR
    - Send `urls` as a JSON-encoded list of URL strings in the form body.
    """
    if not files and not urls:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one file or a `urls` list.",
        )

    result: dict = {"documents_processed": 0, "chunks_added": 0}

    # Handle URL ingestion
    if urls:
        try:
            url_list = json.loads(urls)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="`urls` must be a JSON list of strings.")
        url_result = ingest_urls([str(u) for u in url_list])
        result["documents_processed"] += url_result.get("documents_processed", 0)
        result["chunks_added"] += url_result.get("chunks_added", 0)
        if url_result.get("errors"):
            result["errors"] = url_result["errors"]

    # Handle file ingestion
    if files:
        tmp_dir = tempfile.mkdtemp()
        try:
            saved_paths = []
            for upload in files:
                dest = Path(tmp_dir) / (upload.filename or "upload.txt")
                with open(dest, "wb") as f:
                    shutil.copyfileobj(upload.file, f)
                saved_paths.append(str(dest))

            file_result = ingest_files(saved_paths)
            result["documents_processed"] += file_result.get("documents_processed", 0)
            result["chunks_added"] += file_result.get("chunks_added", 0)
            result["avg_chunk_size"] = file_result.get("avg_chunk_size")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return IngestResponse(status="success", **result)


@app.get("/documents", response_model=List[DocumentSummary], tags=["ingestion"])
def documents():
    """List all indexed documents with their chunk counts."""
    return list_documents()


@app.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
def feedback(request: FeedbackRequest):
    """
    Submit thumbs up (rating=1) or thumbs down (rating=0) for a Q&A pair.
    Optionally include a comment.
    """
    entry = {
        "id": len(FEEDBACK_STORE) + 1,
        "question": request.question,
        "answer": request.answer,
        "rating": request.rating,
        "comment": request.comment,
        "timestamp": time.time(),
    }
    FEEDBACK_STORE.append(entry)
    logger.info(f"Feedback received: id={entry['id']}, rating={request.rating}")
    return FeedbackResponse(status="recorded", feedback_id=entry["id"])


@app.get("/feedback", tags=["feedback"])
def list_feedback(limit: int = 50):
    """(Debug) List recent feedback entries."""
    return FEEDBACK_STORE[-limit:]
