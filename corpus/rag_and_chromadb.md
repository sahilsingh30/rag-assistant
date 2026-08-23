# Retrieval-Augmented Generation (RAG) and ChromaDB

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with large language model generation. Instead of relying solely on the LLM's parametric knowledge, RAG:

1. **Retrieves** relevant documents from an external knowledge base
2. **Augments** the LLM's prompt with that retrieved context
3. **Generates** an answer grounded in the retrieved information

This reduces hallucination, keeps information up-to-date without retraining, and provides citations.

## RAG Pipeline Components

### 1. Document Ingestion
- Load raw documents (PDF, Markdown, HTML, plain text)
- Split into chunks using a text splitter
- Generate vector embeddings for each chunk
- Store in a vector database

### 2. Query-Time Retrieval
- Convert the user query to a vector embedding
- Perform similarity search in the vector database
- Return top-k most similar chunks

### 3. Answer Generation
- Combine retrieved chunks as context in the LLM prompt
- Generate an answer citing the source documents

## Text Splitting Strategies

`RecursiveCharacterTextSplitter` is the most common choice:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(docs)
```

For code-heavy technical docs, use `chunk_size=800` and include heading separators like `"\n## "` to keep sections intact.

## ChromaDB

ChromaDB is an open-source, AI-native vector database. It runs locally and is ideal for prototyping.

### Installation
```bash
pip install chromadb
```

### Basic Usage
```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("my_docs")

# Add documents
collection.add(
    documents=["First document", "Second document"],
    metadatas=[{"source": "file1.txt"}, {"source": "file2.txt"}],
    ids=["id1", "id2"]
)

# Query
results = collection.query(
    query_texts=["what is RAG?"],
    n_results=3
)
```

### LangChain Integration
```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="rag_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Add documents
vectorstore.add_documents(chunks)

# Retrieve
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
docs = retriever.invoke("how does RAG work?")
```

## Self-Corrective RAG (CRAG)

CRAG adds a document grading step to filter irrelevant retrievals:

1. Retrieve top-k documents
2. Grade each document (relevant / irrelevant) using an LLM
3. If all are irrelevant → rewrite query and re-retrieve
4. If some are relevant → filter and generate

This avoids the LLM being misled by off-topic chunks that happen to have vector similarity.

## Adaptive RAG

Adaptive RAG routes queries differently based on their type:
- Simple factual questions → direct generation (no retrieval)
- Complex questions → full RAG pipeline
- Queries outside the corpus → web search fallback

## Embedding Models

| Model | Dimensions | Cost |
|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | 384 | Free, local |
| text-embedding-3-small (OpenAI) | 1536 | ~$0.02/1M tokens |
| text-embedding-004 (Google) | 768 | Free tier available |

For local development, `all-MiniLM-L6-v2` is recommended: fast, lightweight, and good quality.
