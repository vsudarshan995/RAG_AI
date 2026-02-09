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
