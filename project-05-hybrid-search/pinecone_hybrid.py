from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# --- 1. Same IT support ticket data as 05_hybrid_search.py ---
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

texts = [t["text"] for t in tickets]
ids = [t["id"] for t in tickets]
print(f"Loaded {len(tickets)} IT support tickets")

# ============================================================================
# --- 2. YOU do the encoding (Pinecone doesn't do this for you) ---
#
# Dense vectors: capture semantic meaning
#   "page not found" and "missing endpoint" → similar vectors
#   Created by an embedding model (all-MiniLM-L6-v2, 384 dimensions)
#
# Sparse vectors: capture exact keyword matches (like BM25)
#   "ERR-404" → {token_id_for_ERR: 2.1, token_id_for_404: 1.8}
#   Created by BM25Encoder from pinecone-text library
#   Only non-zero entries are stored (hence "sparse")
#
# Pinecone just STORES and SEARCHES these — you compute them.
# ============================================================================
print("\nEncoding documents...")

# Dense encoder (semantic)
dense_model = SentenceTransformer("all-MiniLM-L6-v2")
dense_vectors = dense_model.encode(texts).tolist()
print(f"  Dense vectors: {len(dense_vectors)} x {len(dense_vectors[0])} dimensions")

# Sparse encoder (keyword/BM25)
# fit() learns word frequencies from the corpus (like BM25's IDF component)
bm25 = BM25Encoder()
bm25.fit(texts)
sparse_vectors = bm25.encode_documents(texts)
print(f"  Sparse vectors: {len(sparse_vectors)} documents encoded")

# ============================================================================
# --- 3. Connect to Pinecone and upsert ---
#
# Pinecone index must be pre-created on the website with:
#   - Name: "hybrid-search"
#   - Dimensions: 384 (matches all-MiniLM-L6-v2)
#   - Metric: "dotproduct" (required for sparse-dense hybrid search)
# ============================================================================
print("\nConnecting to Pinecone...")
pc = Pinecone()  # reads PINECONE_API_KEY from environment
index = pc.Index("hybrid-search")

# Upsert: send id + dense vector + sparse vector + metadata
vectors_to_upsert = []
for i in range(len(tickets)):
    vectors_to_upsert.append({
        "id": ids[i],
        "values": dense_vectors[i],                  # dense (you computed)
        "sparse_values": sparse_vectors[i],           # sparse (you computed)
        "metadata": {"text": texts[i], "id": ids[i]}, # metadata for retrieval
    })

index.upsert(vectors=vectors_to_upsert)
print(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone")

# ============================================================================
# --- 4. Hybrid search ---
#
# alpha controls the balance between dense and sparse:
#   alpha=1.0 → pure vector (semantic only)
#   alpha=0.0 → pure sparse (keyword/BM25 only)
#   alpha=0.5 → equal weight (hybrid)
#
# Compare with 05_hybrid_search.py where we used:
#   EnsembleRetriever with weights=[0.5, 0.5]
# Same concept, but here Pinecone does the fusion internally.
# ============================================================================

def hybrid_search(query, alpha=0.5, top_k=3):
    """Search with tunable balance between semantic and keyword."""
    # YOU encode the query (Pinecone doesn't)
    dense_query = dense_model.encode(query).tolist()
    sparse_query = bm25.encode_queries([query])[0]

    # Pinecone does the search + fusion
    results = index.query(
        vector=dense_query,
        sparse_vector=sparse_query,
        top_k=top_k,
        include_metadata=True,
    )
    return results["matches"]


# --- 5. Run the same queries as 05_hybrid_search.py ---
queries = [
    "ERR-404",
    "the app is running very slowly for users",
    "ERR-502 nginx keeps crashing",
]

for query in queries:
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}")

    # Compare all three modes
    for mode_name, alpha in [("BM25 only", 0.0), ("Vector only", 1.0), ("Hybrid", 0.5)]:
        results = hybrid_search(query, alpha=alpha)
        print(f"\n  --- {mode_name} (alpha={alpha}) ---")
        for i, match in enumerate(results):
            text_preview = match["metadata"]["text"][:80]
            print(f"    {i+1}. [{match['id']}] score={match['score']:.4f} | {text_preview}...")
