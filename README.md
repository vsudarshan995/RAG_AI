# InsuranceRAG: Multi-Agent Policy Auditor & Claim Evaluator

InsuranceRAG is an advanced Agentic Retrieval-Augmented Generation (RAG) system. It utilizes a **Multi-Agent Orchestration** layer built on **LangGraph** to automate the complex process of auditing insurance claims against master policy documents and historical data.

## 🤖 Multi-Agent Intelligence

Unlike standard RAG systems that perform a single search, InsuranceRAG uses a state-machine workflow to coordinate specialized agents:

* **Initialization Agent**: Sets up the investigation instance and initializes a permanent audit trail.
* **Policy Selector Agent**: Uses semantic search to retrieve specific clauses from the `policy_master_collection`.
* **History Investigator Agent**: Analyzes the client's past claims in the `claims_collection` to identify behavioral patterns or potential risks.
* **Compliance Evaluator Agent**: Applies organizational Standard Operating Procedures (SOPs), such as police report requirements and submission limits.
* **Synthesis Orchestrator**: Evaluates the findings from all agents to provide a final **APPROVED** or **DENIED** verdict.
* **Archive Agent**: Commits the final decision and full reasoning to the `evaluation_audit_log` for legal compliance.

## 🚀 Key Features

* **Agentic Workflow**: Managed by LangGraph for structured, multi-step decision-making.
* **Dual-Collection Architecture**: Strict separation of immutable Master Policies from Client Claims to prevent data contamination.
* **Semantic Ingestion**: Uses a `SemanticChunker` to split documents based on meaning and topic shifts rather than fixed character counts.
* **Real-Time Progress Visualization**: A dedicated FastAPI endpoint (`/preview/flow/{id}`) renders a live Mermaid.js chart of the agentic process.
* **Automated Watcher**: A background service that monitors storage folders, manages Windows file-locks, and triggers AI-based classification upon file detection.

## 📂 Project Structure

```text
InsuranceRAG/
├── main.py              # FastAPI Server (Async Multi-Agent Endpoints)
├── agent_orchestrator.py # LangGraph State Machine & Agent Nodes
├── ai_service.py        # Modular AI Logic & Vector DB Collections
├── processor.py         # Background Watcher & Semantic Ingestion
├── local_db/            # Persistent ChromaDB storage
├── storage/             # Landing zone for Policies and Claims PDFs
└── requirements.txt     # Project Dependencies
