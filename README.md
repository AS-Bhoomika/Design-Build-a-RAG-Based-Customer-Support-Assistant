# RAG-Based Customer Support Assistant using LangGraph & HITL

## Project Overview
This project is an AI-powered **Retrieval-Augmented Generation (RAG)** based customer support assistant designed to answer customer queries from a PDF knowledge base. It retrieves relevant information using embeddings, responds contextually, uses graph-based workflow logic, and supports Human-in-the-Loop (HITL) escalation for sensitive issues.

This project demonstrates real-world AI system design using document retrieval, vector databases, workflow orchestration, and intelligent routing.

---

## Project Objective
Design and implement a RAG system that:
- Processes a PDF knowledge base
- Retrieves relevant information using embeddings
- Answers user queries contextually
- Uses LangGraph for workflow control
- Routes responses based on intent
- Supports Human-in-the-Loop escalation

---

##  Features
-  PDF Knowledge Base Processing
-  Chunking using RecursiveCharacterTextSplitter
-  Embeddings using HuggingFace
-  Vector Storage with ChromaDB
-  Semantic Retrieval
-  LangGraph Workflow
-  Conditional Query Routing
-  Human Agent Escalation (HITL)
-  CLI Chatbot Interface

---

## Architecture

### Query Flow
User Query → Input Layer → LangGraph → Retrieval / HITL → Final Response

### Document Flow
PDF → Loader → Chunking → Embeddings → ChromaDB

---

## Components
- **Document Loader** – Loads PDF file
- **Chunking Module** – Splits text into chunks
- **Embedding Module** – Converts text to vectors
- **Vector Store** – Stores embeddings
- **Retriever** – Finds relevant chunks
- **LangGraph Engine** – Controls workflow
- **Routing Layer** – Decides answer or escalation
- **HITL Module** – Human support handoff

---

##  Tech Stack
- Python
- LangChain
- LangGraph
- ChromaDB
- HuggingFace Embeddings
- PyPDFLoader
- VS Code
- GitHub

---

## Project Structure

rag-support-assistant/
│── app.py
│── support_kb.pdf
│── requirements.txt
│── data/
│── README.md

---

## Installation & Setup

### 1. Clone Repository
git clone <your-github-link>
cd rag-support-assistant

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run Project
python app.py

---

## Sample Queries
Refund Policy  
Delivery Policy  
Support Hours  
Cancellation  
I am angry  
Need manager  
Complaint about service  

---

## Sample Output

### Normal Query
Answer: Refund Policy: Refunds are processed within 5 to 7 business days.

### Escalation Query
Answer: Escalated to Human Agent. Please wait.

---

## Workflow Logic

### Nodes
- Processing Node
- Output Node

### Flow
Start → Process → Output → End

### State Object
query → context → answer → escalate

---

## Human-in-the-Loop (HITL)

The system escalates queries when users express:
- Anger
- Complaint
- Legal issue
- Need manager
- Unresolved problem
- Refund not received

This ensures sensitive issues are handled by a human support agent.

---

##  Testing
Tested with:
- Normal policy questions
- Complaint scenarios
- Empty input
- Unknown queries

---

##  Future Enhancements
- Multi-document support
- Streamlit Web UI
- Memory integration
- Feedback collection
- Better LLM summarization
- API deployment
- Cloud hosting

---

##  Learning Outcomes
- Understanding RAG systems
- Working with embeddings
- Using vector databases
- Building LangGraph workflows
- Implementing HITL logic
- Designing scalable AI systems

---

## Acknowledgement
Special thanks to **Innomatics Research Labs** for guidance, mentorship, and learning support.

---

##  Contact
Feel free to connect and share feedback.
