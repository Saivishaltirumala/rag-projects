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
cd project-03-semantic-chunking && python semantic_chunking.py
```

### Project 04 — Parent Document Retriever
Demonstrates the "search small, return big" pattern. Child chunks (400 chars) are embedded in Chroma for precise retrieval, while parent chunks (2000 chars) are stored in InMemoryStore and returned as context. Fixes the retrieval failures from project-01 by returning the full points table when any child row matches.

```bash
cd project-04-parent-document-retriever && python parent_document_retriever.py
```

### Project 05 — Hybrid Search
Combines BM25 keyword search with Chroma vector search using `EnsembleRetriever` and Reciprocal Rank Fusion. Uses IT support ticket data to show where each search type shines and fails — BM25 for exact error codes, vector for semantic descriptions, ensemble for best coverage.

```bash
cd project-05-hybrid-search && python hybrid_search.py
```

### Project 06 — RAGAS Evaluation
Full RAGAS workflow: auto-generates synthetic Q&A test data from a tax guidelines document, runs it through a RAG pipeline, then evaluates with 4 RAGAS metrics (faithfulness, answer relevancy, context precision, context recall). Uses its own venv due to ragas needing `langchain-community <0.4.0`.

```bash
cd project-06-ragas-evaluation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python ragas_testset.py
```
