import os
import shutil
import chromadb
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException

# 1. NEW IMPORT: Connect the AI service logic
from ai_service import router as ai_router

# 2. The main app object
app = FastAPI(title="Insurance RAG API")

# 3. Include the AI Router (This activates the /ask endpoints)
app.include_router(ai_router)

# 4. Define ABSOLUTE storage paths
BASE_DIR = Path(r"D:\pY\InsuranceRAG\storage")
CHROMA_PATH = r"D:\pY\InsuranceRAG\local_db"
BASE_DIR.mkdir(parents=True, exist_ok=True)

# 5. Initialize Chroma Client for database management
client = chromadb.PersistentClient(path=CHROMA_PATH)

@app.get("/")
def read_root():
    return {"status": "System Online", "docs_url": "/docs"}

# --- STORAGE ENDPOINTS ---

@app.post("/upload/policy", tags=["Storage Management"])
async def upload_policy(
    category: str,  # NEW: Mandatory category parameter
    file: UploadFile = File(...)
):
    """
    Upload a policy document. Mandatory 'category' is required (e.g., Motor, Life, Medical).
    """
    # 1. Validate category isn't empty
    category = category.strip().capitalize()
    if not category:
        raise HTTPException(status_code=400, detail="Document classification category is mandatory.")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    
    # 2. Update folder structure to include the category
    # This structure is used by processor.py to extract metadata
    target_folder = BASE_DIR / "policies" / category / timestamp
    target_folder.mkdir(parents=True, exist_ok=True)
    
    # 3. Save the file with the category prefix for easy identification
    file_path = target_folder / file.filename
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
            
    return {
        "message": "Policy uploaded and classified", 
        "category": category,
        "path": str(file_path)
    }

@app.post("/upload/claim/{client_id}/{submission_type}", tags=["Storage Management"])
async def upload_claim(client_id: str, submission_type: str, file: UploadFile = File(...)):
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder_name = f"{date_str}_{submission_type}"
    target_folder = BASE_DIR / "claims" / client_id / folder_name
    target_folder.mkdir(parents=True, exist_ok=True)
    
    file_path = target_folder / file.filename
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
            
    return {"message": "Claim submitted", "path": str(file_path)}

# --- DATABASE & CLEANUP ENDPOINTS ---

@app.delete("/clear-all", tags=["Database Management"])
async def clear_all_data():
    """
    Wipes both dual collections and physical storage.
    """
    try:
        # 1. Clear both dual collections
        for name in ["policy_master_collection", "claims_collection"]:
            try:
                client.delete_collection(name=name)
            except:
                pass # Collection might not exist yet
            client.create_collection(name=name)
        
        # 2. Clear Physical Files
        if BASE_DIR.exists():
            shutil.rmtree(BASE_DIR)
            BASE_DIR.mkdir(parents=True, exist_ok=True)
            
        return {"message": "All dual database collections and physical files have been deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete-client/{client_id}", tags=["Database Management"])
async def delete_client_data(client_id: str):
    """
    Removes specific client data from the claims collection.
    """
    try:
        # 1. Delete from Claims Collection
        collection = client.get_collection(name="claims_collection")
        collection.delete(where={"client_id": client_id})
        
        # 2. Delete Physical Folder
        client_folder = BASE_DIR / "claims" / client_id
        if client_folder.exists():
            shutil.rmtree(client_folder)
            
        return {"message": f"Data and files for {client_id} removed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))