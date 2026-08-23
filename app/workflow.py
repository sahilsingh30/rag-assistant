"""
LangGraph Self-Corrective RAG Workflow
Nodes: Query Analysis → Retrieval → Document Grading → Generation
Conditional routing on grading outcome with retry logic.
"""

from __future__ import annotations

import os
from typing import List, Literal, TypedDict

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from app.retriever import get_retriever
from app.prompts import (
    QUERY_REWRITE_PROMPT,
    GRADING_PROMPT,
    GENERATION_PROMPT,
    QUERY_TYPE_PROMPT,
)

# ─────────────────────────────────────────────
# State Schema
# ─────────────────────────────────────────────

class RAGState(TypedDict):
    question: str                    # original user question
    rewritten_query: str             # (possibly rewritten) query sent to retriever
    query_type: str                  # conceptual / how-to / troubleshooting / api-ref
    documents: List[Document]        # raw retrieved chunks
    relevant_documents: List[Document]  # chunks that passed grading
    answer: str                      # final generated answer
    retry_count: int                 # how many rewrite+re-retrieve cycles so far
    sources: List[str]               # source file names for citations


MAX_RETRIES = 2  # after this many rewrites, give up


# ─────────────────────────────────────────────
# LLM helper
# ─────────────────────────────────────────────

def _llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.environ["GROQ_API_KEY"],
    )


# ─────────────────────────────────────────────
# Node 1 — Query Analysis
# ─────────────────────────────────────────────

def query_analysis_node(state: RAGState) -> RAGState:
    """Rewrite the query for better retrieval and classify its type."""
    llm = _llm()
    question = state["question"]

    # Classify query type
    type_response = llm.invoke(QUERY_TYPE_PROMPT.format(question=question))
    query_type = type_response.content.strip().lower()
    # Normalise to known categories
    if query_type not in {"conceptual", "how-to", "troubleshooting", "api-reference"}:
        query_type = "conceptual"

    # Rewrite / expand the query
    rewrite_response = llm.invoke(
        QUERY_REWRITE_PROMPT.format(question=question, query_type=query_type)
    )
    rewritten = rewrite_response.content.strip()

    return {
        **state,
        "rewritten_query": rewritten,
        "query_type": query_type,
        "retry_count": state.get("retry_count", 0),
    }


# ─────────────────────────────────────────────
# Node 2 — Retrieval
# ─────────────────────────────────────────────

def retrieval_node(state: RAGState) -> RAGState:
    """Search the vector store with the (rewritten) query."""
    retriever = get_retriever(k=5)
    query = state.get("rewritten_query") or state["question"]
    docs = retriever.invoke(query)
    return {**state, "documents": docs}


# ─────────────────────────────────────────────
# Node 3 — Document Grading
# ─────────────────────────────────────────────

def document_grading_node(state: RAGState) -> RAGState:
    """Grade each retrieved chunk; keep only relevant ones."""
    llm = _llm()
    question = state["question"]
    relevant: List[Document] = []

    for doc in state["documents"]:
        response = llm.invoke(
            GRADING_PROMPT.format(question=question, document=doc.page_content[:1500])
        )
        verdict = response.content.strip().lower()
        if "relevant" in verdict and "irrelevant" not in verdict:
            relevant.append(doc)

    return {**state, "relevant_documents": relevant}


# ─────────────────────────────────────────────
# Node 4 — Generation
# ─────────────────────────────────────────────

def generation_node(state: RAGState) -> RAGState:
    """Generate final answer grounded in relevant documents."""
    llm = _llm()
    docs = state["relevant_documents"]
    context_parts = []
    sources: List[str] = []

    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        context_parts.append(f"[{i}] Source: {src}\n{doc.page_content}")
        if src not in sources:
            sources.append(src)

    context = "\n\n---\n\n".join(context_parts)
    response = llm.invoke(
        GENERATION_PROMPT.format(
            question=state["question"],
            context=context,
            query_type=state.get("query_type", "conceptual"),
        )
    )
    return {**state, "answer": response.content.strip(), "sources": sources}


# ─────────────────────────────────────────────
# Fallback — No relevant documents found
# ─────────────────────────────────────────────

def fallback_node(state: RAGState) -> RAGState:
    """Rewrite the query for another retrieval attempt, or give up."""
    llm = _llm()
    retry_count = state.get("retry_count", 0) + 1

    if retry_count > MAX_RETRIES:
        return {
            **state,
            "answer": (
                "I'm sorry, I couldn't find relevant information in the documentation "
                "to answer your question. Please try rephrasing, or check the source "
                "documents directly."
            ),
            "sources": [],
            "retry_count": retry_count,
        }

    # More aggressive rewrite for retry
    rewrite_response = llm.invoke(
        QUERY_REWRITE_PROMPT.format(
            question=state["question"],
            query_type=state.get("query_type", "conceptual"),
        )
    )
    new_query = rewrite_response.content.strip()
    return {**state, "rewritten_query": new_query, "retry_count": retry_count}


# ─────────────────────────────────────────────
# Conditional edge router
# ─────────────────────────────────────────────

def route_after_grading(state: RAGState) -> Literal["generate", "fallback"]:
    """Route to generation if relevant docs exist, else to fallback."""
    if state.get("relevant_documents"):
        return "generate"
    return "fallback"


def route_after_fallback(state: RAGState) -> Literal["retrieve", "end"]:
    """If retries remain, go back to retrieval; otherwise end."""
    if state.get("retry_count", 0) > MAX_RETRIES:
        return "end"
    return "retrieve"


# ─────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("query_analysis", query_analysis_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("grade_documents", document_grading_node)
    graph.add_node("generate", generation_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("query_analysis")
    graph.add_edge("query_analysis", "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"generate": "generate", "fallback": "fallback"},
    )

    graph.add_edge("generate", END)

    graph.add_conditional_edges(
        "fallback",
        route_after_fallback,
        {"retrieve": "retrieve", "end": END},
    )

    return graph.compile()


# Singleton compiled graph
rag_graph = build_graph()
