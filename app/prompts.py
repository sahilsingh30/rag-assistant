"""
Prompt templates used across the LangGraph nodes.
Kept in one place for easy tuning / experimentation.
"""

# ── Node 1: query type classification ─────────────────────────────────────────

QUERY_TYPE_PROMPT = """\
Classify the following user question into exactly ONE category.
Categories:
  - conceptual   (asking what something is or how it works)
  - how-to       (asking how to do a specific task or implement something)
  - troubleshooting  (asking why something fails or how to fix an error)
  - api-reference    (asking about a specific function, class, parameter, or return value)

Respond with ONLY the category name, nothing else.

Question: {question}
"""

# ── Node 1: query rewriting ────────────────────────────────────────────────────

QUERY_REWRITE_PROMPT = """\
You are an expert at reformulating questions to improve document retrieval.

Original question: {question}
Question type: {query_type}

Rewrite the question to be more specific and retrieve better results from a
technical documentation vector store. Rules:
- Expand abbreviations
- Add relevant synonyms or alternative phrasings in parentheses
- For api-reference queries, include the likely class or module name
- For how-to queries, frame as "how to <verb> <object>"
- Keep the rewritten query concise (max 2 sentences)

Respond with ONLY the rewritten query, no explanation.
"""

# ── Node 3: document relevance grading ────────────────────────────────────────

GRADING_PROMPT = """\
You are grading whether a documentation chunk is relevant to a user question.

User question: {question}

Documentation chunk:
\"\"\"
{document}
\"\"\"

Is this chunk RELEVANT or IRRELEVANT to answering the question?
A chunk is relevant if it contains information that directly or partially helps
answer the question. A chunk is irrelevant if it is off-topic or contains no
useful information for this question.

Respond with exactly one word: relevant OR irrelevant
"""

# ── Node 4: answer generation ──────────────────────────────────────────────────

GENERATION_PROMPT = """\
You are a helpful technical documentation assistant. Answer the user's question
using ONLY the provided context chunks. Do not use any outside knowledge.

Question type: {query_type}
Question: {question}

Context (each chunk has a [number] and source):
{context}

Instructions:
- Answer clearly and accurately based solely on the context above
- Cite sources inline using [number] notation (e.g. "According to [1]...")
- If the context partially answers the question, say so explicitly
- For how-to questions, use numbered steps if applicable
- For api-reference questions, include function signatures if present
- Keep the answer focused and avoid repetition

Answer:
"""
