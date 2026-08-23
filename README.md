# RAG-Based Technical Documentation Assistant

A self-corrective Retrieval-Augmented Generation system built with **LangGraph**, **ChromaDB**, and **FastAPI**.



---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Design Decisions & Tradeoffs](#design-decisions--tradeoffs)
4. [Setup & Installation](#setup--installation)
5. [Running the Application](#running-the-application)
6. [API Reference](#api-reference)
7. [Example Requests](#example-requests)
8. [What I'd Improve With More Time](#what-id-improve-with-more-time)

---

## Overview

This system answers natural language questions about technical documentation using a **self-corrective RAG pipeline**. The key innovation over a naive RAG system is the **Document Grading** step: an LLM evaluates retrieved chunks for relevance before generating an answer. If no relevant chunks are found, the pipeline rewrites the query and retries — up to a configurable limit — rather than hallucinating from bad context.

**Corpus used:** LangGraph concepts, RAG & ChromaDB patterns, FastAPI, and Pydantic — all directly relevant to the assignment's own tech stack.

**Tech stack:**
| Layer | Choice | Reason |
|---|---|---|
| LLM | Groq (Llama 3.1 8B Instant) | Free tier, very fast inference |
| Embeddings | `all-MiniLM-L6-v2` (local) | No API key needed, good quality |
| Vector store | ChromaDB (local persistent) | Zero infra, easy to inspect |
| Workflow | LangGraph `StateGraph` | Native support for cycles + conditional edges |
| API | FastAPI | Type-safe, auto-docs, async support |

---

## Architecture

### LangGraph Workflow

```
 User Question
      │
      ▼
┌─────────────────┐
│  Query Analysis │  ── rewrites query + classifies type
└────────┬────────┘      (conceptual / how-to / troubleshooting / api-ref)
         │
         ▼
┌─────────────────┐
│    Retrieval    │  ── similarity search → top-5 chunks from ChromaDB
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Document Grading│  ── LLM grades each chunk: relevant / irrelevant
└────────┬────────┘
         │
    ┌────┴────────────────┐
    │                     │
  relevant?           no relevant
    │                 documents?
    ▼                     │
┌──────────┐        ┌─────┴──────┐
│Generation│        │  Fallback  │
│(→ answer)│        │(rewrite +  │
└──────────┘        │ re-retrieve│
                    │ max 2x)    │
                    └─────┬──────┘
                          │
                    still no results
                          │
                          ▼
                   "I don't know" response
```

### State Schema

```python
class RAGState(TypedDict):
    question: str               # original question (never mutated)
    rewritten_query: str        # query sent to the retriever (may change on retry)
    query_type: str             # classification result
    documents: List[Document]   # raw retrieved chunks
    relevant_documents: List[Document]  # chunks that passed grading
    answer: str                 # final answer
    retry_count: int            # number of rewrite+re-retrieve cycles
    sources: List[str]          # source filenames / URLs for citations
```

### File Structure

```
rag-assistant/
├── app/
│   ├── main.py          # FastAPI application & endpoints
│   ├── workflow.py      # LangGraph StateGraph (core logic)
│   ├── retriever.py     # ChromaDB vector store wrapper
│   ├── ingestion.py     # Document loading, chunking, embedding
│   └── prompts.py       # All LLM prompt templates
├── corpus/
│   ├── langgraph_concepts.md
│   ├── rag_and_chromadb.md
│   └── fastapi_pydantic.md
├── scripts/
│   └── ingest_corpus.py # One-time ingestion script
├── .env.example
├── requirements.txt
└── README.md
```

---

## Design Decisions & Tradeoffs

### 1. Chunking Strategy: Heading-First Recursive Splitting

**Choice:** `RecursiveCharacterTextSplitter` with separators `["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "]`, `chunk_size=800`, `chunk_overlap=150`.

**Why:** Technical documentation is structured around headings. By listing `\n## ` as the first separator, the splitter tries to break at section boundaries before falling back to paragraph or line breaks. This keeps conceptually related content — a heading + its explanation — together in one chunk, which dramatically improves both retrieval precision and the grader's ability to assess relevance.

**Tradeoff:** With heading-first splitting, some sections with very long explanations will still be split mid-paragraph. A `chunk_overlap=150` mitigates this by repeating 150 characters at boundaries, so the grader sees enough context at seams.

### 2. LLM Choice: Groq (Llama 3.1 8B Instant)

**Why:** Groq offers a generous free tier with extremely low latency (~200ms per call). For a grading-heavy workflow where every chunk requires its own LLM call, latency compounds fast. Llama 3.1 8B is capable enough for both structured grading (returning "relevant"/"irrelevant") and answer generation.

**Tradeoff:** Llama 3.1 8B will occasionally misclassify relevance compared to GPT-4o or Claude Sonnet. This is acceptable for a prototype; production would warrant a stronger model for the grader specifically.

### 3. Embedding Model: `all-MiniLM-L6-v2` (local)

**Why:** Runs entirely locally — no API key, no cost, no rate limits. For a small corpus (< 500 chunks), it's fast enough to not be a bottleneck. At 384 dimensions, ChromaDB queries are very fast.

**Tradeoff:** Lower dimensional embeddings mean less semantic nuance than `text-embedding-3-large`. For a corpus > 10k chunks, switching to a larger model (e.g., OpenAI `text-embedding-3-small`) would improve retrieval quality.

### 4. Retry Limit: 2 retries (`MAX_RETRIES = 2`)

After 2 rewrite-and-retrieve cycles, the system returns an "I don't know" response rather than retrying indefinitely. This prevents:
- Infinite loops when the corpus genuinely doesn't cover the topic
- Excessive API costs from repeated LLM calls

### 5. Feedback as In-Memory List

The `/feedback` endpoint stores entries in a Python list (lost on restart). For the prototype this is fine. In production, this would write to PostgreSQL with the `(question, answer, rating)` stored for offline analysis and future fine-tuning data.

### 6. Separation of Ingestion and Query-Time

Ingestion is a separate script (`scripts/ingest_corpus.py`) rather than running at API startup. This keeps API startup fast and avoids re-embedding the same documents on every restart. ChromaDB's `PersistentClient` ensures embeddings survive restarts.

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### 1. Clone and create virtual environment

```bash
git clone https://github.com/sahilsingh30/rag-assistant.git
cd rag-assistant
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` embedding model (~90 MB).

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here
```

### 4. Ingest the corpus

```bash
python scripts/ingest_corpus.py
```

This loads the local Markdown files and attempts to fetch remote documentation pages. Remote fetching is optional — the local corpus is sufficient to demonstrate the system.

---

## Running the Application

```bash
uvicorn app.main:app --reload --port 8000
```

API is now live at `http://localhost:8000`

Interactive docs (Swagger UI): `http://localhost:8000/docs`

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/query` | Submit a question |
| `POST` | `/ingest` | Ingest files or URLs |
| `GET` | `/documents` | List indexed documents |
| `POST` | `/feedback` | Submit answer feedback |
| `GET` | `/feedback` | View stored feedback |

---

## Example Requests

### Submit a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I add a conditional edge in LangGraph?"}'
```

**Response:**
```json
{
  "question": "How do I add a conditional edge in LangGraph?",
  "answer": "To add a conditional edge in LangGraph, use the `add_conditional_edges` method [1]...",
  "sources": ["langgraph_concepts.md"],
  "query_type": "how-to",
  "rewritten_query": "how to add conditional edge in LangGraph StateGraph with routing function",
  "retry_count": 0,
  "latency_ms": 1842.5
}
```

### Ingest a URL

```bash
curl -X POST http://localhost:8000/ingest \
  -F 'urls=["https://python.langchain.com/docs/concepts/retrieval/"]'
```

### Ingest a local file

```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@my_docs.md"
```

### List indexed documents

```bash
curl http://localhost:8000/documents
```

```json
[
  {"source": "langgraph_concepts.md", "chunks": 18},
  {"source": "rag_and_chromadb.md",   "chunks": 22},
  {"source": "fastapi_pydantic.md",   "chunks": 19}
]
```

### Submit feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I add a conditional edge?",
    "answer": "Use add_conditional_edges...",
    "rating": 1,
    "comment": "Very clear answer with code example"
  }'
```

---

## What I'd Improve With More Time

1. **Hallucination check node** — Add a post-generation node that asks the LLM: "Is this answer fully supported by the context?" and flags or regenerates unsupported claims (Self-RAG pattern).

2. **Web search fallback** — If `retry_count >= MAX_RETRIES`, call Tavily instead of returning "I don't know". This turns the system from a closed-book assistant into one that can answer anything.

3. **Conversation memory** — Store the last N (question, answer) pairs per `session_id` and prepend them to the generation prompt. This enables follow-up questions like "Can you give an example of that?"

4. **Better embedding model** — Swap `all-MiniLM-L6-v2` for `text-embedding-3-small` for higher semantic accuracy, especially on mixed technical content.

5. **Persistent feedback store** — Write feedback to SQLite or PostgreSQL. Over time, low-rated answers can be analyzed to identify corpus gaps or prompt weaknesses.

6. **Streamlit frontend** — A simple UI with question input, answer display with source citations, and thumbs up/down buttons. Would make demos much easier.

7. **Async LLM calls in grader** — Currently the document grading node grades chunks sequentially. With `asyncio.gather`, all `k` grading calls could run in parallel, cutting grader latency by ~5×.

---

## Assumptions Made

- The corpus is small enough (~500 chunks) that local embeddings and ChromaDB are sufficient without approximate nearest neighbour optimisation.
- `chunk_size=800` characters is appropriate for technical documentation. Code-heavy docs might benefit from smaller chunks (~400) to avoid mixing multiple unrelated code examples.
- A retry limit of 2 is appropriate. This can be changed via `MAX_RETRIES` in `workflow.py`.
- Groq's free tier rate limits are sufficient for demo/evaluation purposes (14,400 requests/day).
