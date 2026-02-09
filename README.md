# InsuranceRAG: AI-Powered Policy Auditor & Claim Evaluator

InsuranceRAG is a specialized Retrieval-Augmented Generation (RAG) system designed to automate the cross-referencing of insurance claims against master policy documents. It features a dual-collection architecture, automated AI classification of documents, and a modular backend capable of swapping between local (Ollama) and cloud (OpenAI) LLMs.

## 🚀 Key Features

* **Dual-Collection Architecture**: Separates immutable Master Policies from Client Claims to prevent data contamination.
* **Cognitive Intake (AI Classification)**: Automatically identifies claim types (Motor, Life, Medical) by comparing new submissions against existing master policy definitions.
* **Modular LLM Core**: Easily swap between local models like **Llama 3 (via Ollama)** and cloud models like **GPT-4o**.
* **Automated Watcher**: A background processing service that monitors storage folders, handles file-locking on Windows, and organizes indexed files into `processed/` subfolders.
* **Strict Temporal Filtering**: Filters claim evaluations by `client_id` and `submission_date` to ensure contextually accurate audits.

## 📂 Project Structure

```text
InsuranceRAG/
├── main.py              # FastAPI Server (Uploads & DB Management)
├── processor.py         # Background Watcher & Vector Ingestion
├── ai_service.py        # Modular AI Logic & RAG Chains
├── inspector.ipynb      # Database & Metadata Verification Utility
├── requirements.txt     # Python Dependencies
├── .gitignore           # Excludes DB and Storage from Version Control
└── storage/             # (Local Only) Landing zone for PDFs


## 🚀 How to Run the System

To get the system running, follow these steps in order. You will need **three** active terminal windows or background processes.

### 1. Prerequisites
* **Python 3.10+**
* **Ollama**: [Download and Install Ollama](https://ollama.ai/)
* **Local LLM Model**: Pull the required model via terminal:
  ```bash
  ollama pull llama3:8b-instruct-q2_K

### 2. Environment Setup
# Clone the repository
git clone <your-repo-url>
cd InsuranceRAG

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt

### 3. Execution Order
Step A: Start Ollama
Ensure the Ollama application is running in your system tray so the local API (typically at http://localhost:11434) is accessible.

Step B: Start the Document Processor (Terminal 1)
The processor monitors the storage/ folder. It handles AI-based document classification, manages Windows file-locking, and indexes data into the vector database.

Bash
python processor.py
Leave this running. It will log "FILE DETECTED" and "SUCCESS" as you upload documents.

Step C: Start the FastAPI Server (Terminal 2)
The FastAPI server handles the web interface for file uploads and AI queries.

Bash
uvicorn main:app --reload
The API will be available at http://127.0.0.1:8000.


🛠️ Testing the Workflow
Access Swagger UI: Open http://127.0.0.1:8000/docs in your browser.

Upload a Policy:

Use POST /upload/policy.

Enter a category (e.g., Motor).

Upload a PDF.

Check the Processor: Look at your Terminal 1. You should see the AI identifying the policy and moving it to the processed/ folder.

Evaluate a Claim:

Use POST /ask/evaluate-claim.

Input the client_id and submission_date associated with a claim file.

The AI will compare the claim context against the master policy.
