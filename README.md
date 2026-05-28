# RAG Projects

A collection of hands-on RAG (Retrieval-Augmented Generation) projects for learning RAG patterns and techniques progressively.

## Overview

| # | Project | Concepts Covered | File |
|---|---------|-----------------|------|
| 01 | Basic RAG Pipeline | LCEL chains, RecursiveCharacterTextSplitter, Chroma, chunking failures | `01_basic_rag.py` |
| 02 | Sentence-Boundary Chunking | NLTKTextSplitter, sentence-aware splitting, Punkt tokenizer | `02_sentence_chunking.py` |
| 03 | Semantic Chunking | SemanticChunker, embedding-based topic detection, similarity thresholds | `03_semantic_chunking.py` |
| 04 | Parent Document Retriever | Parent-child chunks, InMemoryStore, search small return big | `04_parent_document_retriever.py` |
| 05 | Hybrid Search | BM25 keyword search, EnsembleRetriever, Reciprocal Rank Fusion | `05_hybrid_search.py` |
| 06 | RAGAS Evaluation | Synthetic test generation, faithfulness, answer relevancy, context precision/recall | `project-06-ragas-evaluation/` |

## Setup

```bash
pip install -r requirements.txt
# Add your API keys to .env
```

## Running Projects 01-05

Single-file scripts at root level. All share `sample.txt` (IPL 2026 data).

```bash
python 01_basic_rag.py
python 02_sentence_chunking.py
python 03_semantic_chunking.py
python 04_parent_document_retriever.py
python 05_hybrid_search.py
```

## Running Project 06

Uses its own venv due to ragas needing `langchain-community <0.4.0`.

```bash
cd project-06-ragas-evaluation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python ragas_testset.py
```
