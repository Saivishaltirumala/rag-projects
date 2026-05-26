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

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# --- 1. Load document ---
loader = TextLoader("sample.txt")
documents = loader.load()
print(f"Loaded {len(documents)} document(s)")

# --- 2. Split into chunks ---
# KNOWN LIMITATION: Fixed character-length splitting breaks logical sections (e.g. a 10-row
# points table gets split across 3 chunks). This causes retrieval failures:
#
#   Q: "Which teams finished with 8 points?"
#   A: Only finds LSG (Chunk 3), misses MI (Chunk 2)
#
# Why it fails — a chain of 3 factors:
#   1. chunk_size (root cause): 500-char cut splits the table mid-way, separating related rows
#   2. embedding (amplifier): Chunk 2 has "8 points" diluted among 15,14,13,12 → weaker match.
#      Chunk 3 is short (98 chars) and dominated by "8 points + Eliminated" → strong match.
#      Embeddings capture average meaning of the chunk, not individual lines.
#   3. k=3 (final gate): Chunk 2 ranks 4th (distance 1.2096) just behind Chunk 4 (1.2062),
#      so it gets cut off. k=4 would catch it, but that's brute force not a real fix.
#
# Fix: use semantic/structure-aware splitting that keeps logical sections together.
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# --- 3. Embed & store in Chroma ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
print("Vector store created")

# --- 4. Build RAG chain ---
# k=3 -> retrieve top 3 most similar chunks by cosine distance (lower distance = more similar)
# lower k = faster/cheaper but may miss context, higher k = more context but risks noise
# NOTE: increasing k is a band-aid — the real fix is better chunking so related data
# stays together and doesn't need multiple retrievals to reconstruct
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}"""
)

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

# chain_type options (old API): "stuff" | "map_reduce" | "refine" | "map_rerank"
# This LCEL chain is equivalent to "stuff" — all retrieved chunks combined into one prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LCEL (LangChain Expression Language) chain — the pipe "|" operator chains steps like Unix pipes.
#
# Trace flow for: rag_chain.invoke("Who has the Orange Cap?")
#
#   "Who has the Orange Cap?"
#       │
#       ▼
#   ┌── RunnableParallel (both paths run on same input simultaneously) ──┐
#   │                                                                     │
#   │  "context" path:                    "question" path:                │
#   │    "Who has the Orange Cap?"          "Who has the Orange Cap?"     │
#   │        │                                   │                        │
#   │        ▼ retriever                         ▼ RunnablePassthrough()  │
#   │    [Doc("Sai Sudharsan..638"),       "Who has the Orange Cap?"     │
#   │     Doc("Orange Cap Race.."),         (passes input unchanged,     │
#   │     Doc("Gujarat Titans..")]          needed to preserve raw       │
#   │        │                              query for the prompt)        │
#   │        ▼ format_docs                                               │
#   │    "Sai Sudharsan..638\n\n                                         │
#   │     Orange Cap Race..\n\n                                          │
#   │     Gujarat Titans.."                                              │
#   └─────────────────────────────────────────────────────────────────────┘
#       │
#       ▼ output dict fills prompt template's {context} and {question}
#   "Answer the question based only on the following context:
#    Sai Sudharsan..638 runs...
#    Question: Who has the Orange Cap?"
#       │
#       ▼ llm (Claude)
#   AIMessage("Sai Sudharsan has the Orange Cap with 638 runs")
#       │
#       ▼ StrOutputParser (extracts text from AIMessage)
#   "Sai Sudharsan has the Orange Cap with 638 runs"
#
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 5. Interactive Q&A loop ---
print("\n--- RAG Q&A (type 'quit' to exit) ---\n")
while True:
    question = input("You: ").strip()
    if question.lower() in ("quit", "exit", "q"):
        break
    if not question:
        continue
    answer = rag_chain.invoke(question)
    print(f"\nBot: {answer}\n")
