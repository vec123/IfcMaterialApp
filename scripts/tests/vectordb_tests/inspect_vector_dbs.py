import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Resolve path mappings seamlessly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../" if "src" in current_dir else ".")) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.load_conf import load_config
from src.RAG.vectorstore import VectorStoreManager

# Isolated Absolute Database Directory Routes
CANONICAL_DB_PATH = os.environ.get("VECTOR_DB_PATH")
IFC_DB_PATH =  os.environ.get("IFC_VECTOR_DB_PATH")

def force_dump_collection(db_path: str, collection_name: str, label: str):
    """
    Spins up an isolated VectorStoreManager targeting the absolute disk path
    to ensure we extract 'materials_v1' from the correct database instance.
    """
    print("\n" + "="*80)
    print(f"🎯 CONSOLE INSPECTOR: {label}")
    print(f"📂 DIRECTORY PATH:    {db_path}")
    print(f"📦 COLLECTION NAME:   '{collection_name}'")
    print("="*80)
    
    if not os.path.exists(db_path):
        print(f"🚨 ERROR: The directory path '{db_path}' does not exist on disk!")
        return

    try:
        # Explicit initialization to defeat any internal cached singleton cross-talk
        vsm = VectorStoreManager(db_path=db_path, collection_name=collection_name)
        
        # Pull records directly from this specific Chroma collection payload
        chroma_collection = vsm.collection
        raw_data = chroma_collection.get(include=["metadatas", "documents"])

        ids = raw_data.get("ids", []) if raw_data else []
        metadatas = raw_data.get("metadatas", []) if raw_data else []
        documents = raw_data.get("documents", []) if raw_data else []

        total_elements = len(ids)
        print(f"📊 Total Elements Discovered: {total_elements}")
        print("-" * 80)

        if total_elements == 0:
            print(f"🚨 Warning: This database folder contains 0 elements inside '{collection_name}'.")
            return

        for i in range(total_elements):
            element_id = ids[i]
            element_meta = metadatas[i] if i < len(metadatas) else {}
            element_doc = documents[i] if i < len(documents) else "No document text stored."
            
            print(f"\n🔥 Element [{i + 1}/{total_elements}]")
            print(f"🆔 ID / Key:       {element_id}")
            print(f"📋 Metadata:      {element_meta}")
            print(f"📄 Page Content:  {str(element_doc).strip()}")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ Failed processing database at '{db_path}' due to error: {e}")

if __name__ == "__main__":
    # Force collection name to 'materials_v1' based on your configuration architecture
    TARGET_COLLECTION = "materials_v1"
    
    # Run the isolated directory dumps
    force_dump_collection(
        db_path=CANONICAL_DB_PATH, 
        collection_name=TARGET_COLLECTION, 
        label="CANONICAL DATABASE LIBRARY"
    )
    
    force_dump_collection(
        db_path=IFC_DB_PATH, 
        collection_name=TARGET_COLLECTION, 
        label="EXTRACTED PROJECT IFC MATERIALS"
    )
    
    print("\n🏁 Inspection complete.")