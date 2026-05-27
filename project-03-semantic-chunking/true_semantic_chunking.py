from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# --- 1. Load document (same file as project-01 & 02 for comparison) ---
loader = TextLoader("sample.txt")
documents = loader.load()
print(f"Loaded {len(documents)} document(s)")

# ============================================================================
# --- 2. TRUE SEMANTIC splitting (SemanticChunker) ---
#
# How it works (step by step):
#   1. Split text into individual sentences
#   2. Embed EVERY sentence using the embedding model
#   3. Compare embeddings of adjacent sentences (cosine similarity)
#   4. When similarity DROPS between two adjacent sentences → topic changed → split here
#
# Breakpoint types (how it decides "similarity dropped enough"):
#   - "percentile" (default): split where the drop is in the top X% of all drops
#                              e.g. percentile=70 → split at the top 30% biggest drops
#   - "standard_deviation":    split where drop exceeds mean + (X * std_dev)
#   - "interquartile":         split where drop exceeds Q3 + 1.5*IQR (outlier detection)
#   - "gradient":              split where rate of change in similarity spikes
#
# vs project-01 (RecursiveCharacterTextSplitter): blind 500-char cuts
# vs project-02 (NLTKTextSplitter): sentence-aware but still fills by char count,
#                                    doesn't understand if sentences are about same topic
#
# SemanticChunker groups sentences by MEANING — the entire points table stays together
# because all rows are semantically similar (team stats), and splits when the topic
# shifts (e.g. from points table → Orange Cap discussion).
# ============================================================================
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

semantic_chunker = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",  # split at biggest similarity drops
    breakpoint_threshold_amount=70,          # top 30% of drops trigger a split
)
semantic_chunks = semantic_chunker.create_documents([documents[0].page_content])
print(f"Semantic chunker produced {len(semantic_chunks)} chunks")

# --- 3. Log chunks ---
print(f"\n{'='*70}")
print(f"TRUE SEMANTIC: {len(semantic_chunks)} chunks (grouped by meaning)")
print(f"{'='*70}")

print(f"\n--- Semantic Chunks (topic-aware) ---")
for i, chunk in enumerate(semantic_chunks):
    # Show first and last line to verify topic coherence
    lines = [l.strip() for l in chunk.page_content.strip().split("\n") if l.strip()]
    first_line = lines[0][:70] if lines else ""
    last_line = lines[-1][:70] if len(lines) > 1 else "  (single line)"
    print(f"\n  Chunk {i:2d} | {len(chunk.page_content):4d} chars | {len(lines):2d} lines")
    print(f"    FIRST: {first_line}...")
    print(f"    LAST:  {last_line}...")

# --- 4. Embed & store in Chroma ---
print(f"\nEmbedding semantic chunks into Chroma...")
vectorstore = Chroma.from_documents(semantic_chunks, embeddings, persist_directory="./chroma_db")

# --- 5. Similarity search — the query that FAILED in project-01 ---
query = "How many teams were eliminated and which teams finished with 8 points?"
results = vectorstore.similarity_search_with_score(query, k=3)

print(f"\n{'='*70}")
print(f"Query: {query}")
print(f"{'='*70}")
for i, (doc, score) in enumerate(results):
    print(f"\n--- Result {i+1} | Distance: {score:.4f} ---")
    print(doc.page_content)
