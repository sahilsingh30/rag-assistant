"""
Retriever: wraps ChromaDB vector store.
The collection is created/loaded once and reused across requests.
"""

import os
from functools import lru_cache

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "rag_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once and cache it."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def _get_vectorstore() -> Chroma:
    """Connect to (or create) the ChromaDB collection."""
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=_get_embeddings(),
    )


def get_retriever(k: int = 5):
    """Return a LangChain retriever over the ChromaDB collection."""
    vs = _get_vectorstore()
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})


def get_vectorstore() -> Chroma:
    return _get_vectorstore()


def list_documents() -> list[dict]:
    """Return a summary of all indexed documents (unique sources + chunk count)."""
    vs = _get_vectorstore()
    try:
        result = vs.get(include=["metadatas"])
        sources: dict[str, int] = {}
        for meta in result["metadatas"]:
            src = meta.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        return [{"source": k, "chunks": v} for k, v in sorted(sources.items())]
    except Exception:
        return []


def clear_vectorstore() -> None:
    """Drop and recreate the collection (used in tests)."""
    # Invalidate LRU cache so next call creates a fresh instance
    _get_vectorstore.cache_clear()
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    client.delete_collection(COLLECTION_NAME)
