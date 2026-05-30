import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# ============================================================================
# STAGE 1: Extract entities & relationships from text using LLM
# ============================================================================
# The LLM reads each chunk and outputs structured JSON with:
#   - entities: nodes in the graph (Player, Team, Award, etc.)
#   - relationships: edges connecting nodes (plays_for, won, scored, etc.)
#
# This is the most expensive step — every chunk needs an LLM call.
# We use Groq (Llama 3, free tier) to keep costs at zero.
# ============================================================================
print("=" * 70)
print("STAGE 1: Extracting Entities & Relationships from Text")
print("=" * 70)

# Load and chunk the same IPL 2026 data used in previous projects
loader = TextLoader(Path(__file__).resolve().parent.parent / "sample.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Loaded and split into {len(chunks)} chunks")

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

EXTRACTION_PROMPT = """Extract entities and relationships from the text below.
Return ONLY valid JSON with this exact format, no other text:
{{
  "entities": [
    {{"name": "entity name", "type": "Player|Team|Award|Tournament|Match", "properties": {{}}}}
  ],
  "relationships": [
    {{"source": "entity name", "target": "entity name", "type": "plays_for|won|scored|qualified|eliminated|captains|holds_award|match_result"}}
  ]
}}

Rules:
- Entity names must be consistent (use full names, not abbreviations)
- Include numeric properties like runs, wickets, points, NRR where available
- Each relationship must reference entities that exist in the entities list

Text: {text}"""

all_entities = []
all_relationships = []

for i, chunk in enumerate(chunks):
    print(f"  Processing chunk {i+1}/{len(chunks)}...", end=" ", flush=True)
    try:
        response = llm.invoke(EXTRACTION_PROMPT.format(text=chunk.page_content))
        data = json.loads(response.content)
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        all_entities.extend(entities)
        all_relationships.extend(relationships)
        print(f"Found {len(entities)} entities, {len(relationships)} relationships")
    except (json.JSONDecodeError, Exception) as e:
        print(f"Skipped (parse error: {e})")
    time.sleep(2)  # Groq free tier rate limit: ~6000 tokens/min

print(f"\nTotal extracted: {len(all_entities)} entities, {len(all_relationships)} relationships")

# ============================================================================
# STAGE 2: Store in Neo4j
# ============================================================================
# Entities become NODES with labels (Player, Team, etc.) and properties.
# Relationships become EDGES connecting nodes.
#
# MERGE is used instead of CREATE to avoid duplicates — if a node with
# the same name already exists, it updates properties instead of creating
# a duplicate.
# ============================================================================
print(f"\n{'='*70}")
print("STAGE 2: Storing in Neo4j")
print("=" * 70)

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
)
driver.verify_connectivity()
print("Connected to Neo4j")

# Clear existing data for a clean run
with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
    print("Cleared existing graph data")

# Insert entities as nodes
def sanitize(s):
    """Remove spaces and special chars from Cypher identifiers."""
    return s.replace(" ", "_").replace("-", "_").replace(".", "")

with driver.session() as session:
    for entity in all_entities:
        name = entity["name"]
        label = sanitize(entity.get("type", "Entity"))
        props = entity.get("properties", {})
        # Use parameterized properties to avoid injection and special char issues
        params = {"name": name}
        set_parts = []
        for k, v in props.items():
            safe_key = sanitize(k)
            params[safe_key] = str(v)
            set_parts.append(f"n.{safe_key} = ${safe_key}")
        set_clause = f"SET {', '.join(set_parts)}" if set_parts else ""
        query = f"MERGE (n:{label} {{name: $name}}) {set_clause}"
        try:
            session.run(query, **params)
        except Exception:
            pass

    print(f"Inserted {len(all_entities)} entity nodes")

# Insert relationships as edges
rel_count = 0
with driver.session() as session:
    for rel in all_relationships:
        try:
            rel_type = sanitize(rel["type"]).upper()
            query = f"""
                MATCH (a {{name: $source}}), (b {{name: $target}})
                MERGE (a)-[r:{rel_type}]->(b)
            """
            session.run(query, source=rel["source"], target=rel["target"])
            rel_count += 1
        except Exception:
            pass

    print(f"Inserted {rel_count} relationships")

# ============================================================================
# STAGE 3: Print graph structure
# ============================================================================
print(f"\n{'='*70}")
print("STAGE 3: Graph Structure")
print("=" * 70)

with driver.session() as session:
    # Count nodes by label
    result = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC")
    print("\n  Node counts by type:")
    for record in result:
        print(f"    {record['label']:20s} {record['count']}")

    # Count relationships by type
    result = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY count DESC")
    print("\n  Relationship counts by type:")
    for record in result:
        print(f"    {record['type']:20s} {record['count']}")

    # Show sample relationships
    result = session.run("MATCH (a)-[r]->(b) RETURN a.name AS from_node, type(r) AS rel, b.name AS to_node LIMIT 15")
    print("\n  Sample relationships:")
    for record in result:
        print(f"    {record['from_node']} --{record['rel']}--> {record['to_node']}")

# ============================================================================
# STAGE 4: GraphRAG — Natural language → Cypher → Answer
# ============================================================================
# The LLM converts English questions into Cypher queries.
# This is what makes GraphRAG powerful — users don't need to know Cypher.
#
# Flow: User question → LLM generates Cypher → execute on Neo4j
#       → results → LLM generates natural language answer
# ============================================================================
print(f"\n{'='*70}")
print("STAGE 4: GraphRAG Q&A")
print("=" * 70)

# First, get the graph schema so the LLM knows what's available
with driver.session() as session:
    nodes_result = session.run("MATCH (n) RETURN DISTINCT labels(n)[0] AS label, keys(n) AS props LIMIT 20")
    rels_result = session.run("MATCH (a)-[r]->(b) RETURN DISTINCT labels(a)[0] AS from_label, type(r) AS rel, labels(b)[0] AS to_label LIMIT 20")

    node_info = [f"(:{r['label']} {{{', '.join(r['props'])}}})" for r in nodes_result]
    rel_info = [f"(:{r['from_label']})-[:{r['rel']}]->(:{r['to_label']})" for r in rels_result]
    schema = f"Nodes: {', '.join(set(node_info))}\nRelationships: {', '.join(set(rel_info))}"

CYPHER_PROMPT = """You are a Neo4j Cypher expert. Convert the user's question into a Cypher query.

Graph schema:
{schema}

Rules:
- Return ONLY the Cypher query, no explanation, no markdown
- Use MATCH and RETURN
- Property access: use n.name, n.runs, etc.
- Always RETURN meaningful columns with aliases

Question: {question}"""

ANSWER_PROMPT = """Answer the user's question based on the database results below.
Be concise and direct. If no results found, say so.

Question: {question}
Database results: {results}"""

# Demo queries + interactive loop
demo_queries = [
    "Which players play for qualified teams?",
    "Who has the most runs among Gujarat Titans players?",
    "Which team has both an Orange Cap and Purple Cap contender?",
]

print("\n--- Running demo queries ---\n")
for question in demo_queries:
    print(f"  Q: {question}")
    try:
        # Step 1: Generate Cypher
        cypher_response = llm.invoke(CYPHER_PROMPT.format(schema=schema, question=question))
        cypher = cypher_response.content.strip().strip("`").replace("cypher\n", "")
        print(f"  Cypher: {cypher}")

        # Step 2: Execute on Neo4j
        with driver.session() as session:
            result = session.run(cypher)
            records = [dict(r) for r in result]

        # Step 3: Generate answer
        answer_response = llm.invoke(ANSWER_PROMPT.format(question=question, results=records))
        print(f"  Answer: {answer_response.content}")
    except Exception as e:
        print(f"  Error: {e}")
    print()
    time.sleep(3)  # Groq rate limit buffer

# Interactive loop
print("--- Interactive Q&A (type 'quit' to exit) ---\n")
while True:
    question = input("You: ").strip()
    if question.lower() in ("quit", "exit", "q"):
        break
    if not question:
        continue
    try:
        cypher_response = llm.invoke(CYPHER_PROMPT.format(schema=schema, question=question))
        cypher = cypher_response.content.strip().strip("`").replace("cypher\n", "")
        print(f"Cypher: {cypher}")

        with driver.session() as session:
            result = session.run(cypher)
            records = [dict(r) for r in result]

        answer_response = llm.invoke(ANSWER_PROMPT.format(question=question, results=records))
        print(f"\nBot: {answer_response.content}\n")
    except Exception as e:
        print(f"\nError: {e}\n")

driver.close()
