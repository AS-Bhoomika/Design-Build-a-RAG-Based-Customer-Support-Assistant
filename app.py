from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from typing import TypedDict

# ---------------------------------------------------
# Load PDF Knowledge Base
# ---------------------------------------------------
loader = PyPDFLoader("support_kb.pdf")
docs = loader.load()

# ---------------------------------------------------
# Split PDF into Chunks
# ---------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)

# ---------------------------------------------------
# Create Embeddings
# ---------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------
# Store in ChromaDB
# ---------------------------------------------------
vectordb = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="data"
)

# ---------------------------------------------------
# Create Retriever
# ---------------------------------------------------
retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ---------------------------------------------------
# Graph State
# ---------------------------------------------------
class GraphState(TypedDict):
    query: str
    context: str
    answer: str
    escalate: bool


# ---------------------------------------------------
# Process Node
# ---------------------------------------------------
def process_node(state):
    query = state["query"].strip()
    query_lower = query.lower()

    # HITL Escalation Conditions
    hitl_words = [
        "angry",
        "complaint",
        "legal",
        "agent",
        "manager",
        "bad service",
        "refund not received",
        "not happy",
        "issue unresolved",
        "speak to human",
        "frustrated",
        "very bad",
        "need support"
    ]

    if any(word in query_lower for word in hitl_words):
        return {
            "context": "",
            "answer": "",
            "escalate": True
        }

    # Retrieve Relevant Content
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs]).strip()

    # No Data Found
    if context == "":
        return {
            "context": "",
            "answer": "No relevant information found in the PDF knowledge base.",
            "escalate": False
        }

    # Remove Duplicate Lines
    unique_lines = list(dict.fromkeys(context.splitlines()))
    lines = [line.strip() for line in unique_lines if line.strip()]

    # Match Best Lines
    matched = []

    # Priority 1: Exact Phrase Match
    for line in lines:
        if query_lower in line.lower():
            matched.append(line)

    # Priority 2: Keyword Match
    if not matched:
        for line in lines:
            if any(word in line.lower() for word in query_lower.split()):
                matched.append(line)

    # Final Output
    final_answer = "\n".join(matched) if matched else "\n".join(lines)

    return {
        "context": context,
        "answer": f"Based on uploaded PDF:\n{final_answer}",
        "escalate": False
    }


# ---------------------------------------------------
# Output Node
# ---------------------------------------------------
def output_node(state):
    if state["escalate"]:
        return {
            "answer": "Escalated to Human Agent. Please wait."
        }

    return {
        "answer": state["answer"]
    }


# ---------------------------------------------------
# Build LangGraph Workflow
# ---------------------------------------------------
builder = StateGraph(GraphState)

builder.add_node("process", process_node)
builder.add_node("output", output_node)

builder.set_entry_point("process")
builder.add_edge("process", "output")
builder.add_edge("output", END)

app = builder.compile()


# ---------------------------------------------------
# Chat Loop
# ---------------------------------------------------
while True:
    q = input("Ask Question (type exit): ").strip()

    if q == "":
        print("Please type a question.")
        print("-" * 50)
        continue

    if q.lower() == "exit":
        break

    result = app.invoke({
        "query": q,
        "context": "",
        "answer": "",
        "escalate": False
    })

    print("\nAnswer:", result["answer"])
    print("-" * 50)