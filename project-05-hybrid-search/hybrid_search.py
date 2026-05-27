from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# --- 1. IT support ticket data ---
# Mix of exact error codes, technical jargon, and natural descriptions
# to show where keyword vs semantic search each shine and fail.
tickets = [
    {"id": "TKT-001", "text": "ERR-404: Page not found when accessing /api/users endpoint. The server returns a 404 status code for all user-related API routes since the last deployment."},
    {"id": "TKT-002", "text": "ERR-500: Internal server error on the payment gateway. The checkout process crashes with a NullPointerException in PaymentService.java line 142."},
    {"id": "TKT-003", "text": "The application is extremely slow and pages take over 30 seconds to load. Users are complaining about poor performance during peak hours between 2-5 PM."},
    {"id": "TKT-004", "text": "ERR-403: Access denied when trying to view admin dashboard. User john.doe@company.com reports forbidden error after role migration to new RBAC system."},
    {"id": "TKT-005", "text": "Database connection pool exhausted. PostgreSQL max_connections limit reached causing ERR-503 service unavailable across all microservices."},
    {"id": "TKT-006", "text": "ERR-404: Static assets returning not found after CDN migration. CSS and JavaScript files fail to load, breaking the entire frontend UI."},
    {"id": "TKT-007", "text": "Memory leak detected in the notification service. The Java heap space grows continuously and the pod gets OOMKilled every 6 hours in Kubernetes."},
    {"id": "TKT-008", "text": "SSL certificate expired for api.company.com. All HTTPS requests are failing with ERR_CERT_DATE_INVALID. Customers cannot access the portal."},
    {"id": "TKT-009", "text": "Login fails silently with no error message. Users click the login button and nothing happens. Browser console shows CORS policy blocking the auth API request."},
    {"id": "TKT-010", "text": "ERR-502: Bad gateway error intermittently on the load balancer. Nginx upstream returns 502 when backend pods are scaling up during auto-scaling events."},
]

documents = [Document(page_content=t["text"], metadata={"id": t["id"]}) for t in tickets]
print(f"Loaded {len(documents)} IT support tickets")

# ============================================================================
# --- 2. Set up TWO retrievers ---
#
# BM25Retriever (keyword search):
#   - Exact token matching with BM25 scoring (TF saturation + length normalization)
#   - Finds "ERR-404" by matching the exact string character by character
#   - Fast, no embeddings needed, runs locally
#   - Fails when user describes the problem without using exact terms
#
# Chroma vector retriever (semantic search):
#   - Embeds query + documents, finds by cosine similarity
#   - Understands "page not found" ≈ "404 error" ≈ "missing endpoint"
#   - Fails on exact codes, acronyms, and numbers
# ============================================================================

# BM25 — keyword search
bm25_retriever = BM25Retriever.from_documents(documents, k=3)

# Chroma — vector search
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ============================================================================
# --- 3. Combine with EnsembleRetriever ---
#
# EnsembleRetriever runs BOTH retrievers, then fuses results using
# Reciprocal Rank Fusion (RRF):
#
#   RRF score = sum( 1 / (rank_in_list + k) ) across all lists
#
#   Example: Doc appears rank 1 in BM25 and rank 3 in vector:
#     score = 1/(1+60) + 1/(3+60) = 0.0164 + 0.0159 = 0.0323
#
#   Doc appears rank 1 in BM25 only:
#     score = 1/(1+60) = 0.0164  (lower — only one list found it)
#
#   Docs found by BOTH retrievers naturally rank higher.
#
# weights=[0.5, 0.5] means both retrievers contribute equally.
# Adjust to [0.7, 0.3] to favor BM25 for exact-match-heavy use cases,
# or [0.3, 0.7] to favor vector for semantic-heavy queries.
# ============================================================================
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],
)

# --- 4. Run queries that show the difference ---
queries = [
    # Query A: exact error code — BM25 should dominate
    "ERR-404",
    # Query B: semantic description, no exact terms — vector should dominate
    "the app is running very slowly for users",
    # Query C: mix of exact + semantic — needs both
    "ERR-502 nginx keeps crashing",
]

for query in queries:
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}")

    bm25_results = bm25_retriever.invoke(query)
    vector_results = vector_retriever.invoke(query)
    ensemble_results = ensemble_retriever.invoke(query)

    print(f"\n  --- BM25 (keyword) ---")
    for i, doc in enumerate(bm25_results):
        print(f"    {i+1}. [{doc.metadata['id']}] {doc.page_content[:90]}...")

    print(f"\n  --- Vector (semantic) ---")
    for i, doc in enumerate(vector_results):
        print(f"    {i+1}. [{doc.metadata['id']}] {doc.page_content[:90]}...")

    print(f"\n  --- Ensemble (hybrid) ---")
    for i, doc in enumerate(ensemble_results):
        # Check which retriever(s) found this doc
        in_bm25 = any(d.metadata["id"] == doc.metadata["id"] for d in bm25_results)
        in_vector = any(d.metadata["id"] == doc.metadata["id"] for d in vector_results)
        source = "BOTH" if in_bm25 and in_vector else ("BM25" if in_bm25 else "VECTOR")
        print(f"    {i+1}. [{doc.metadata['id']}] ({source}) {doc.page_content[:80]}...")
