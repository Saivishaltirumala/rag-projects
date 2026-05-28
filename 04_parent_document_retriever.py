from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

# --- 1. Load document (same file as all previous projects) ---
loader = TextLoader("sample.txt")
documents = loader.load()
print(f"Loaded {len(documents)} document(s)")

# ============================================================================
# --- 2. Configure TWO splitters ---
#
# The key idea: search on SMALL chunks (precise embeddings),
#               return the LARGE parent chunk (complete context).
#
# Parent splitter (2000 chars):
#   Creates large chunks that contain full logical sections.
#   These are stored in InMemoryStore (NOT embedded, NOT searched).
#   They are what gets RETURNED to the LLM.
#
# Child splitter (400 chars):
#   Splits each parent into small, focused sub-chunks.
#   These are embedded and stored in Chroma (searched via vector similarity).
#   Each child stores a metadata key "doc_id" pointing to its parent.
#
# Flow:  query → search children in Chroma → find matching child
#        → look up child's parent_id → return full parent from InMemoryStore
# ============================================================================
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

# --- 3. Set up the two stores ---
# Vector store: holds child chunks + their embeddings (for searching)
# Doc store: holds parent chunks as plain text (for returning)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db_04")
doc_store = InMemoryStore()

# --- 4. Create the ParentDocumentRetriever ---
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=doc_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# --- 5. Add documents (this splits into parents, then children, and stores both) ---
retriever.add_documents(documents)

# Log what was created
child_docs = vectorstore.similarity_search("", k=100)
parent_keys = list(doc_store.yield_keys())
print(f"Parent chunks stored in InMemoryStore: {len(parent_keys)}")
print(f"Child chunks stored in Chroma: {len(child_docs)}")

# Show parent vs child chunks
print(f"\n{'='*70}")
print("PARENT chunks (large, stored in InMemoryStore, returned to LLM):")
print(f"{'='*70}")
for key in parent_keys:
    parent = doc_store.mget([key])[0]
    preview = parent.page_content[:100].replace("\n", " | ")
    print(f"  Parent {key[:8]}... | {len(parent.page_content):4d} chars | {preview}...")

print(f"\n{'='*70}")
print("CHILD chunks (small, embedded in Chroma, used for searching):")
print(f"{'='*70}")
for i, doc in enumerate(child_docs):
    preview = doc.page_content[:80].replace("\n", " | ")
    parent_id = doc.metadata.get("doc_id", "?")[:8]
    print(f"  Child {i:2d} | {len(doc.page_content):3d} chars | parent={parent_id}... | {preview}...")

# ============================================================================
# --- 6. Run the query that FAILED in project-01 ---
#
# In project-01: "8 points?" matched Chunk 3 (LSG only, 98 chars)
#                Chunk 2 (MI + other teams) ranked 4th, missed by k=3
#
# Here: "8 points?" will match a small CHILD chunk mentioning MI/LSG
#       → retriever looks up parent_id
#       → returns the FULL parent chunk with the complete points table
# ============================================================================
query = "How many teams were eliminated and which teams finished with 8 points?"

# First, show what the CHILD search finds (raw vector search)
child_results = vectorstore.similarity_search_with_score(query, k=3)
print(f"\n{'='*70}")
print(f"Query: {query}")
print(f"{'='*70}")

print(f"\n--- STEP 1: Child chunks matched (what vector search found) ---")
for i, (doc, score) in enumerate(child_results):
    print(f"\n  Child Match {i+1} | Distance: {score:.4f} | {len(doc.page_content)} chars")
    print(f"  {doc.page_content[:120].replace(chr(10), ' | ')}...")

# Now, show what ParentDocumentRetriever actually returns (the parents)
parent_results = retriever.invoke(query)
print(f"\n--- STEP 2: Parent chunks returned (what the LLM would receive) ---")
for i, doc in enumerate(parent_results):
    print(f"\n  Parent Result {i+1} | {len(doc.page_content)} chars")
    print(f"  {'~'*60}")
    print(f"  {doc.page_content}")
    print(f"  {'~'*60}")
