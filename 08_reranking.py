from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

# --- 1. Load & chunk (same as project-01) ---
loader = TextLoader("sample.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Loaded and split into {len(chunks)} chunks")

# --- 2. Embed & store in Chroma ---
import shutil
if Path("./chroma_db_08").exists():
    shutil.rmtree("./chroma_db_08")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db_08")
print("Vector store created")

# ============================================================================
# --- 3. Initialize the Cross-Encoder Re-ranker ---
#
# How bi-encoder (retriever) vs cross-encoder (re-ranker) work:
#
# Bi-encoder (what the retriever uses):
#   Encodes query and document SEPARATELY into vectors, then compares.
#   Fast (can pre-compute doc vectors), but approximate.
#
#   query  → [encoder] → vector_q ─┐
#                                    ├── cosine similarity → score
#   doc    → [encoder] → vector_d ─┘
#
# Cross-encoder (what the re-ranker uses):
#   Encodes query AND document TOGETHER as a single input.
#   Sees word interactions between query and doc → much more accurate.
#   Slow (must run for every query-doc pair), so only used on pre-filtered results.
#
#   [query + doc] → [encoder] → single relevance score
#
# That's why we retrieve 10 with the fast bi-encoder first,
# then re-rank those 10 with the accurate cross-encoder.
# ============================================================================
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Cross-encoder re-ranker loaded")

# --- 4. The query that FAILED in project-01 ---
query = "How many teams were eliminated and which teams finished with 8 points?"

# ============================================================================
# --- 5. WITHOUT re-ranking (project-01 style) ---
# Retrieve top 3 directly from vector store.
# In project-01, this missed Chunk 2 (MI with 8 pts) because it ranked 4th.
# ============================================================================
print(f"\n{'='*70}")
print(f"Query: {query}")
print(f"{'='*70}")

results_no_rerank = vectorstore.similarity_search_with_score(query, k=3)

print(f"\n--- WITHOUT Re-ranking (top 3 from retriever) ---")
for i, (doc, score) in enumerate(results_no_rerank):
    # Find chunk index
    for j, c in enumerate(chunks):
        if c.page_content == doc.page_content:
            chunk_idx = j
            break
    has_mi = "MI" in doc.page_content and "8" in doc.page_content
    marker = " ← has MI 8pts" if has_mi else ""
    print(f"  Rank {i+1} | Chunk {chunk_idx:2d} | Distance: {score:.4f} | {doc.page_content[:80].replace(chr(10), ' | ')}...{marker}")

# ============================================================================
# --- 6. WITH re-ranking ---
# Step 1: Over-fetch — retrieve top 10 (cast a wide net)
# Step 2: Re-rank — cross-encoder scores each (query, chunk) pair
# Step 3: Pick top 3 after re-ranking
#
# The cross-encoder sees the FULL query + FULL chunk text together,
# so it can understand that "teams finished with 8 points" in the query
# matches "MI - Points: 8" in the chunk, even if the embedding missed it.
# ============================================================================

# Step 1: Over-fetch top 10
results_wide = vectorstore.similarity_search_with_score(query, k=10)

print(f"\n--- Retriever returned top 10 (wide net) ---")
for i, (doc, score) in enumerate(results_wide):
    for j, c in enumerate(chunks):
        if c.page_content == doc.page_content:
            chunk_idx = j
            break
    has_mi = "MI" in doc.page_content and "8" in doc.page_content
    marker = " ← has MI 8pts" if has_mi else ""
    print(f"  Rank {i+1:2d} | Chunk {chunk_idx:2d} | Distance: {score:.4f}{marker}")

# Step 2: Re-rank with cross-encoder
pairs = [(query, doc.page_content) for doc, _ in results_wide]
rerank_scores = reranker.predict(pairs)

# Step 3: Sort by re-rank score (higher = more relevant) and pick top 3
reranked = sorted(
    zip(results_wide, rerank_scores),
    key=lambda x: x[1],
    reverse=True,
)

print(f"\n--- WITH Re-ranking (top 3 after cross-encoder) ---")
for i, ((doc, retriever_score), rerank_score) in enumerate(reranked[:3]):
    for j, c in enumerate(chunks):
        if c.page_content == doc.page_content:
            chunk_idx = j
            break
    has_mi = "MI" in doc.page_content and "8" in doc.page_content
    marker = " ← has MI 8pts" if has_mi else ""
    print(f"  Rank {i+1} | Chunk {chunk_idx:2d} | Rerank: {rerank_score:.4f} | Retriever: {retriever_score:.4f} | {doc.page_content[:70].replace(chr(10), ' | ')}...{marker}")

# ============================================================================
# --- 7. Compare RAG answers ---
# Run both (without rerank and with rerank) through the LLM to see
# how the answer quality changes.
# ============================================================================
print(f"\n{'='*70}")
print("RAG Answer Comparison")
print(f"{'='*70}")

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Answer WITHOUT re-ranking
context_no_rerank = format_docs([doc for doc, _ in results_no_rerank])
chain = prompt | llm | StrOutputParser()
answer_no_rerank = chain.invoke({"context": context_no_rerank, "question": query})

print(f"\n--- Answer WITHOUT re-ranking ---")
print(f"  {answer_no_rerank}")

# Answer WITH re-ranking
top3_reranked_docs = [doc for (doc, _), _ in reranked[:3]]
context_reranked = format_docs(top3_reranked_docs)
answer_reranked = chain.invoke({"context": context_reranked, "question": query})

print(f"\n--- Answer WITH re-ranking ---")
print(f"  {answer_reranked}")
