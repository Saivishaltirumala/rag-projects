# NOTE: This project uses its own venv because ragas 0.4.3 needs
# langchain-community <0.4.0 which conflicts with other projects.
# Activate with: source venv/bin/activate

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
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas import evaluate
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# ============================================================================
# STAGE 1: Generate Synthetic Test Dataset
# ============================================================================
# RAGAS TestsetGenerator reads the document, understands its content, and
# automatically creates diverse question-answer pairs. This eliminates the
# need to manually write test cases.
# ============================================================================
print("=" * 70)
print("STAGE 1: Generating Synthetic Test Dataset")
print("=" * 70)

# Load document
loader = TextLoader("tax_guidelines_2026.txt")
documents = loader.load()
print(f"Loaded {len(documents)} document(s)")

# Initialize LLM and embeddings — same stack as all previous projects
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Wrap for RAGAS — RAGAS has its own LLM/embedding interface, these wrappers
# bridge LangChain objects to RAGAS's internal format
ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# Generate synthetic test cases
generator = TestsetGenerator(llm=ragas_llm, embedding_model=ragas_embeddings)
testset = generator.generate_with_langchain_docs(documents, testset_size=5)

# Convert to DataFrame and display
df = testset.to_pandas()
print(f"\nGenerated {len(df)} synthetic test cases:\n")
for i, row in df.iterrows():
    print(f"  Q{i+1}: {row['user_input']}")
    print(f"  Ground Truth: {row['reference']}")
    print()

# ============================================================================
# STAGE 2: Run RAG Pipeline on Generated Questions
# ============================================================================
# Build a basic RAG chain (same as project-01) and run every generated
# question through it. Collect the RAG answer + retrieved contexts for
# RAGAS evaluation.
# ============================================================================
print("=" * 70)
print("STAGE 2: Running RAG Pipeline on Generated Questions")
print("=" * 70)

# Clean up stale Chroma DB from previous runs to avoid duplicate embeddings
import shutil
if Path("./chroma_db").exists():
    shutil.rmtree("./chroma_db")

# Chunk and embed the document
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print(f"Split into {len(chunks)} chunks, embedded in Chroma")
print("Running questions through RAG chain...\n")

# Build RAG chain (reused from project-01)
prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run each question through the RAG pipeline and collect results
eval_samples = []
for i, row in df.iterrows():
    question = row["user_input"]
    ground_truth = row["reference"]

    # Get RAG answer
    print(f"  Processing Q{i+1}/{len(df)}...", end=" ", flush=True)
    answer = rag_chain.invoke(question)

    # Get retrieved contexts separately (for RAGAS evaluation)
    retrieved_docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in retrieved_docs]

    eval_samples.append(
        SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        )
    )

    print(f"Done")
    print(f"    Q: {question[:80]}...")
    print(f"    Answer: {answer[:120]}...")
    print()

# ============================================================================
# STAGE 3: Evaluate with RAGAS
# ============================================================================
# RAGAS uses an LLM as a judge to score each answer on 4 metrics.
# Each metric is scored 0.0 → 1.0 (higher is better).
#
#   Faithfulness:       Is the answer supported by the retrieved context?
#   Answer Relevancy:   Does the answer actually address the question?
#   Context Precision:  Are the retrieved chunks relevant and well-ranked?
#   Context Recall:     Did we retrieve ALL the info needed to answer?
# ============================================================================
print("=" * 70)
print("STAGE 3: RAGAS Evaluation")
print("=" * 70)

eval_dataset = EvaluationDataset(samples=eval_samples)

metrics = [
    Faithfulness(llm=ragas_llm),
    AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
    ContextPrecision(llm=ragas_llm),
    ContextRecall(llm=ragas_llm),
]

results = evaluate(
    dataset=eval_dataset,
    metrics=metrics,
)

# Print per-question scores
results_df = results.to_pandas()
print(f"\n--- Per-Question Scores ---")
metric_cols = [c for c in results_df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference", "reference_contexts")]
for i, row in results_df.iterrows():
    print(f"\n  Q{i+1}: {row['user_input'][:80]}...")
    for col in metric_cols:
        val = row[col]
        if isinstance(val, float):
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"    {col:25s} {bar} {val:.4f}")

# Print aggregate scores
print(f"\n{'='*70}")
print("RAGAS EVALUATION RESULTS (Averages)")
print(f"{'='*70}")
for col in metric_cols:
    avg = results_df[col].mean()
    if isinstance(avg, float):
        bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
        print(f"  {col:25s} {bar} {avg:.4f}")
print(f"{'='*70}")
