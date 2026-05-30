# RAG Projects

A collection of hands-on RAG (Retrieval-Augmented Generation) projects for learning RAG patterns and techniques progressively.

## Overview

| # | Project | Concepts Covered | Location |
|---|---------|-----------------|----------|
| 01 | Basic RAG Pipeline | LCEL chains, RecursiveCharacterTextSplitter, Chroma, chunking failures | `01_basic_rag.py` |
| 02 | Sentence-Boundary Chunking | NLTKTextSplitter, sentence-aware splitting, Punkt tokenizer | `02_sentence_chunking.py` |
| 03 | Semantic Chunking | SemanticChunker, embedding-based topic detection, similarity thresholds | `03_semantic_chunking.py` |
| 04 | Parent Document Retriever | Parent-child chunks, InMemoryStore, search small return big | `04_parent_document_retriever.py` |
| 05 | Hybrid Search | BM25 + vector search, EnsembleRetriever, Pinecone sparse-dense, RRF | `project-05-hybrid-search/` |
| 06 | RAGAS Evaluation | Synthetic test generation, faithfulness, answer relevancy, context precision/recall | `project-06-ragas-evaluation/` |
| 07 | Graph RAG | LLM entity extraction, Neo4j, Cypher generation, knowledge graphs | `project-07-graph-rag/` |
| 08 | Re-ranking | Cross-encoder re-ranker, bi-encoder vs cross-encoder, over-fetch + re-score | `08_reranking.py` |

## Setup

```bash
pip install -r requirements.txt
# Add your API keys to .env
```

## Running Projects 01-04, 08

Single-file scripts at root level. Share `sample.txt` (IPL 2026 data).

```bash
python 01_basic_rag.py
python 02_sentence_chunking.py
python 03_semantic_chunking.py
python 04_parent_document_retriever.py
python 08_reranking.py
```

## Running Project 05 — Hybrid Search

Two implementations of hybrid search for comparison:

```bash
# EnsembleRetriever (local BM25 + Chroma, no API key needed)
cd project-05-hybrid-search && python ensemble_retriever.py

# Pinecone (requires PINECONE_API_KEY in .env + index created on pinecone.io)
cd project-05-hybrid-search && python pinecone_hybrid.py
```

## Running Project 07 — Graph RAG

Requires Docker with Neo4j running.

```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password123 neo4j:latest
cd project-07-graph-rag && pip install -r requirements.txt && python graph_rag.py
```

## Running Project 06 — RAGAS Evaluation

Uses its own venv due to ragas needing `langchain-community <0.4.0`.

```bash
cd project-06-ragas-evaluation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python ragas_testset.py
```
