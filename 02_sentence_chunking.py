from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

# --- 1. Load document (same file as project-01 for comparison) ---
loader = TextLoader("sample.txt")
documents = loader.load()
print(f"Loaded {len(documents)} document(s)")

# ============================================================================
# --- 2a. SEMANTIC splitting (NLTKTextSplitter) ---
# NLTKTextSplitter uses NLTK's sentence tokenizer (Punkt) to detect sentence
# boundaries using ML-based rules (abbreviations, punctuation patterns, etc.)
# It splits ONLY at sentence endings — never mid-sentence.
#
# Compare with project-01's RecursiveCharacterTextSplitter which splits at a
# fixed character count (500), blindly cutting through tables and sentences.
#
# chunk_size here is a soft limit — NLTK groups complete sentences together
# until adding the next sentence would exceed the limit, then starts a new chunk.
# So chunks may slightly exceed the limit to keep a sentence intact.
# ============================================================================
semantic_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=0)
semantic_chunks = semantic_splitter.split_documents(documents)

# --- 3. Log chunks ---
print(f"\n{'='*70}")
print(f"SEMANTIC splitter: {len(semantic_chunks)} chunks (sentence-aware, soft limit=500)")
print(f"{'='*70}")

print(f"\n--- Semantic Chunks (sentence-boundary aware) ---")
for i, chunk in enumerate(semantic_chunks):
    preview = chunk.page_content[:80].replace("\n", " | ")
    clean_end = chunk.page_content.rstrip()
    # Check if chunk ends at a sentence boundary (not cut off mid-sentence)
    cut_off = "CLEAN" if clean_end[-1] in ".!?)" else "CUT-OFF"
    print(f"  Chunk {i:2d} | {len(chunk.page_content):3d} chars | {cut_off:7s} | {preview}...")

# --- 4. Embed semantic chunks & store in Chroma ---
print(f"\nEmbedding semantic chunks into Chroma...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(semantic_chunks, embeddings, persist_directory="./chroma_db_02")

# --- 5. Similarity search — the query that FAILED in project-01 ---
# In project-01, this query missed MI (8 pts) because the points table was split
# across chunks 1/2/3 and chunk 2 ranked 4th (just outside k=3).
query = "How many teams were eliminated and which teams finished with 8 points?"
results = vectorstore.similarity_search_with_score(query, k=3)

print(f"\n{'='*70}")
print(f"Query: {query}")
print(f"{'='*70}")
for i, (doc, score) in enumerate(results):
    print(f"\n--- Result {i+1} | Distance: {score:.4f} ---")
    print(doc.page_content)
