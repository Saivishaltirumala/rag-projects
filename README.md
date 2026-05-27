# RAG Projects

A collection of hands-on RAG (Retrieval-Augmented Generation) projects for learning RAG patterns and techniques progressively.

## Setup

```bash
cd 13-rag-projects
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env .env.local  # Add your API keys
```

## Projects

### Project 01 — Basic RAG Pipeline
Terminal-based RAG using LangChain LCEL, HuggingFace embeddings, Chroma, and Claude. Demonstrates how `RecursiveCharacterTextSplitter` (fixed 500-char cuts) breaks logical sections like points tables, causing retrieval failures. Includes detailed comments on chunking limitations and LCEL chain trace flow.

```bash
cd project-01-basic-rag && python rag.py
```

### Project 02 — Sentence-Boundary Chunking
Uses `NLTKTextSplitter` to split at sentence endings instead of fixed character counts. Produces cleaner chunks with zero mid-sentence cut-offs, but still groups by size not meaning — sentences about different topics can end up in the same chunk.

```bash
cd project-02-sentence-chunking && python sentence_chunking.py
```

### Project 03 — Semantic Chunking
Uses LangChain's `SemanticChunker` which embeds every sentence and splits where topic similarity drops. Groups related content by meaning, but has tradeoffs: unpredictable chunk sizes (41–1097 chars), fragile threshold tuning, slow indexing, and sometimes wrong merges when adjacent sentences share keywords across different topics.

```bash
cd project-03-semantic-chunking && python true_semantic_chunking.py
```
