import os
import logging
import time
import shutil
from watchdog.observers.read_directory_changes import WindowsApiObserver as Observer
from watchdog.events import PatternMatchingEventHandler
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM

# 1. Configuration for dual output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.FileHandler("processor_debug.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 2. Initialize Components
logger.info("⚙️ Initializing RAG Components...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = OllamaLLM(model="llama3:8b-instruct-q2_K", num_ctx=2048, temperature=0)

policy_db = Chroma(persist_directory=r"D:\pY\InsuranceRAG\local_db", embedding_function=embeddings, collection_name="policy_master_collection")
claims_db = Chroma(persist_directory=r"D:\pY\InsuranceRAG\local_db", embedding_function=embeddings, collection_name="claims_collection")

class IngestionHandler(PatternMatchingEventHandler): # UPDATED PARENT CLASS
    def __init__(self):
        # Configure the handler to ignore 'processed' folders and non-PDFs
        super().__init__(
            patterns=["*.pdf"],
            ignore_patterns=["*\\processed\\*", "*/processed/*"], # Excludes nested processed folders
            ignore_directories=True
        )
        self.processed_cache = {}
        logger.info("🏠 Handler initialized with strict folder exclusion.")

    # Patterns now handle the filtering, so we only need to catch the events
    def on_created(self, event): self.handle_event(event)
    def on_modified(self, event): self.handle_event(event)

    def handle_event(self, event):
        # We no longer need the 'if pdf' or 'if processed' checks here 
        # because PatternMatchingEventHandler filters them automatically!

        now = time.time()
        if event.src_path in self.processed_cache and now - self.processed_cache[event.src_path] < 5: 
            return 
            
        self.processed_cache[event.src_path] = now
        logger.info(f"FILE DETECTED: {os.path.basename(event.src_path)}")
        self.process_pdf(event.src_path)

    def classify_claim_type(self, sample_text, filename):
        policy_info = policy_db.similarity_search("Claim categories", k=3)
        context = "\n".join([d.page_content for d in policy_info])
        prompt = f"Context: {context}\n\nClaim: {sample_text}\n\nOutput ONLY the category name:"
        return llm.invoke(prompt).strip()

    def wait_for_file_release(self, file_path, retries=10, delay=1):
        """Wait for the OS to finish writing and release the file handle."""
        for i in range(retries):
            try:
                # Try to rename the file to itself. If it fails, the file is locked.
                os.rename(file_path, file_path)
                return True
            except OSError:
                logger.warning(f"⏳ File '{os.path.basename(file_path)}' is busy. Retrying {i+1}/{retries}...")
                time.sleep(delay)
        return False

    def process_pdf(self, file_path):
        filename = os.path.basename(file_path)
        
        # Wait for main.py/Windows to release the file
        if not self.wait_for_file_release(file_path):
            logger.error(f"❌ Could not access {filename}. OS still holding lock.")
            return

        temp_path = file_path + ".ingesting"
        try:
            os.rename(file_path, temp_path)
            logger.info(f"🚀 Processing: {filename}")
            
            parts = os.path.normpath(file_path).split(os.sep)
            is_claim = "claims" in parts
            
            loader = PyMuPDFLoader(temp_path)
            docs = loader.load()
            
            # Classification
            if is_claim:
                category = self.classify_claim_type(docs[0].page_content[:1500] if docs else "", filename)
            else:
                category = parts[-3] 
            
            meta_data = {
                "source_type": "Claim" if is_claim else "Policy",
                "document_category": category,
                "client_id": parts[-3] if is_claim else "Company",
                "submission_date": parts[-2] if is_claim else "N/A"
            }

            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
            for chunk in chunks: chunk.metadata.update(meta_data)
            
            db = claims_db if is_claim else policy_db
            db.add_documents(chunks)
            
            # Move to processed
            processed_dir = os.path.join(os.path.dirname(file_path), "processed")
            os.makedirs(processed_dir, exist_ok=True)
            shutil.move(temp_path, os.path.join(processed_dir, filename))
            logger.info(f"✅ Indexed and moved to processed: {filename}")
            
        except Exception as e: 
            logger.error(f"❌ Error processing {filename}:", exc_info=True)
            if os.path.exists(temp_path): os.rename(temp_path, file_path)

if __name__ == "__main__":
    WATCH_PATH = r"D:\pY\InsuranceRAG\storage"
    observer = Observer()
    observer.schedule(IngestionHandler(), path=WATCH_PATH, recursive=True)
    logger.info(f"🚀 Processor monitoring: {WATCH_PATH}")
    observer.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: observer.stop()
    observer.join()