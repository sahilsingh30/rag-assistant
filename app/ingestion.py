"""
Document Ingestion Pipeline
Supports: local files (Markdown, .txt, .html) and remote URLs.

Chunking strategy:
  - RecursiveCharacterTextSplitter with separators tuned for technical docs
    (\n## > \n### > \n\n > \n > space)
  - chunk_size=800, chunk_overlap=150
  - Rationale: technical docs have nested headings; splitting on headings first
    keeps conceptually related content together. 800 tokens fits comfortably in
    the context window while giving the grader enough signal.  150-char overlap
    prevents cutting mid-sentence at boundaries.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional

import requests
from langchain_community.document_loaders import (
    BSHTMLLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.retriever import get_vectorstore

# ── Chunking config ────────────────────────────────────────────────────────────

SPLITTER = RecursiveCharacterTextSplitter(
    separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "],
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
)


# ── Loaders ────────────────────────────────────────────────────────────────────

def _load_file(path: str) -> List[Document]:
    """Load a single local file into LangChain Documents."""
    ext = Path(path).suffix.lower()
    if ext in {".md", ".markdown"}:
        loader = UnstructuredMarkdownLoader(path)
    elif ext in {".html", ".htm"}:
        loader = BSHTMLLoader(path)
    else:
        loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = Path(path).name
    return docs


def _load_url(url: str) -> List[Document]:
    """Fetch a URL and load as HTML or plain text."""
    headers = {"User-Agent": "Mozilla/5.0 (RAG-assistant docs fetcher)"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    suffix = ".html" if "html" in content_type else ".txt"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    ) as f:
        f.write(response.text)
        tmp_path = f.name

    try:
        docs = _load_file(tmp_path)
        for doc in docs:
            doc.metadata["source"] = url
        return docs
    finally:
        os.unlink(tmp_path)


# ── Public API ─────────────────────────────────────────────────────────────────

def ingest_files(file_paths: List[str], source_name: Optional[str] = None) -> dict:
    """Ingest a list of local file paths into the vector store."""
    all_docs: List[Document] = []
    for path in file_paths:
        docs = _load_file(path)
        if source_name:
            for d in docs:
                d.metadata["source"] = source_name
        all_docs.extend(docs)
    return _chunk_and_store(all_docs)


def ingest_urls(urls: List[str]) -> dict:
    """Fetch and ingest a list of URLs into the vector store."""
    all_docs: List[Document] = []
    errors: List[str] = []
    for url in urls:
        try:
            docs = _load_url(url)
            all_docs.extend(docs)
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    result = _chunk_and_store(all_docs)
    if errors:
        result["errors"] = errors
    return result


def _chunk_and_store(docs: List[Document]) -> dict:
    """Split docs into chunks and upsert into ChromaDB."""
    if not docs:
        return {"chunks_added": 0, "documents_processed": 0}

    chunks = SPLITTER.split_documents(docs)
    vs = get_vectorstore()
    vs.add_documents(chunks)

    return {
        "documents_processed": len(docs),
        "chunks_added": len(chunks),
        "avg_chunk_size": int(
            sum(len(c.page_content) for c in chunks) / max(len(chunks), 1)
        ),
    }
