"""
Corpus Ingestion Script
Run once before starting the API to populate the vector store.

Fetches documentation from LangChain, LangGraph, FastAPI, and Pydantic —
all projects relevant to this assignment's own tech stack.

Usage:
    python scripts/ingest_corpus.py
"""

import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion import ingest_files, ingest_urls

# ── Remote documentation URLs ──────────────────────────────────────────────────

REMOTE_URLS = [
    # LangGraph overview
    "https://langchain-ai.github.io/langgraph/concepts/",
    # LangGraph — how-to: create a graph
    "https://langchain-ai.github.io/langgraph/how-tos/graph-api/",
    # LangChain retrieval concepts
    "https://python.langchain.com/docs/concepts/retrieval/",
    # FastAPI first steps
    "https://fastapi.tiangolo.com/tutorial/first-steps/",
    # Pydantic models
    "https://docs.pydantic.dev/latest/concepts/models/",
]

# ── Local markdown files (bundled in corpus/) ─────────────────────────────────

LOCAL_FILES = list(Path("corpus").glob("**/*.md")) + list(Path("corpus").glob("**/*.txt"))


def main():
    print("=== RAG Documentation Assistant — Corpus Ingestion ===\n")

    if LOCAL_FILES:
        print(f"Ingesting {len(LOCAL_FILES)} local file(s)...")
        result = ingest_files([str(p) for p in LOCAL_FILES])
        print(f"  ✓ {result['documents_processed']} docs → {result['chunks_added']} chunks\n")

    print(f"Fetching {len(REMOTE_URLS)} remote URL(s)...")
    for url in REMOTE_URLS:
        print(f"  → {url}")
    result = ingest_urls(REMOTE_URLS)
    print(f"\n  ✓ {result['documents_processed']} docs → {result['chunks_added']} chunks")
    if result.get("errors"):
        print("\n  ⚠ Errors:")
        for err in result["errors"]:
            print(f"    {err}")

    print("\n✅ Ingestion complete. You can now start the API:\n   uvicorn app.main:app --reload\n")


if __name__ == "__main__":
    main()
