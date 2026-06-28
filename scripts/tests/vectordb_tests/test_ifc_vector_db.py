import os
import yaml
import json
import logging
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VectorDBTester")

def load_config_from_env() -> Dict[str, Any]:
    """Loads environment variables and parses the config.yaml file via absolute paths."""
    load_dotenv()
    
    config_path = os.getenv("CONFIG_PATH")
    if not config_path:
        raise ValueError("❌ CONFIG_PATH not specified in the environment variables/.env file.")
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Configuration file designated by .env was not found at: {config_path}")
        
    logger.info(f"📋 Configuration successfully mapped via .env to: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_database_diagnostics_and_text_export():
    logger.info("🚀 Initializing Vector Database Verification & Text/Metadata Export Pipeline...")
    
    # 1. Load config and resolve core project space pathing
    config = load_config_from_env()
    project_root = os.getenv("PROJECT_PATH", "")
    
    # Target the exact configuration key for your IFC vector database
    yaml_db_path = config['paths']['ifc_vector_db_path']
    absolute_db_path = os.path.abspath(os.path.join(project_root, yaml_db_path))
    output_dir = os.path.abspath(os.path.join(project_root, "data_out"))
    
    vs_config = config['vector_store']
    collections_to_test = [
        vs_config['typology_collection_name'],
        vs_config['material_collection_name'],
        vs_config['ifc_element_collection_name']
    ]
    
    # 2. Instantiate Local Embedding Engine (Enforcing CPU)
    model_name = vs_config['embedding_model']
    device = "cpu"
    logger.info(f"🧠 Loading Shared Embedding Engine: {model_name} onto [{device}]...")
    embedding_engine = SentenceTransformer(model_name, device=device)
    
    # 3. Initialize Persistent Database Client
    logger.info(f"💾 Connecting to vector storage instance at: {absolute_db_path}")
    chroma_client = chromadb.PersistentClient(path=absolute_db_path)
    
    print("\n" + "="*60)
    print("        VECTOR DATABASE INTEGRITY METRICS & TEXT CHUNK EXPORTS")
    print("="*60)
    
    # 4. Iterate, run query tests, and pull records (docs and metadata only)
    for collection_name in collections_to_test:
        try:
            # Check collection existence and record count
            collection = chroma_client.get_collection(name=collection_name)
            record_count = collection.count()
            print(f"📦 Collection: {collection_name:<25} | Status: ONLINE | Records: {record_count}")
            
            if record_count == 0:
                logger.warning(f"Collection '{collection_name}' is empty. Skipping processing.")
                continue
                
            # ── A. RUN SEMANTIC SEARCH TEST CHUNKS ───────────────────────────
            if "typology" in collection_name:
                test_query = "B433_01_Fusta pillar element"
            elif "material" in collection_name:
                test_query = "Hormigón armado estructural"
            else:
                test_query = "B433_01_Fusta pillar element"
                
            query_embedding = embedding_engine.encode(test_query).tolist()
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(2, record_count),
                include=["documents", "metadatas", "distances"]
            )
            
            print(f"   🔍 Test Semantic Query: '{test_query}'")
            if results and results['documents'] and results['documents'][0]:
                for idx, doc in enumerate(results['documents'][0]):
                    distance = results['distances'][0][idx]
                    print(f"      🔹 Match [{idx+1}]: {doc[:60]}... (Distance: {distance:.4f})")
            
            # ── B. EXTRACT CHUNKS WITHOUT EMBEDDINGS VECTOR ARRAY ───────────
            all_records = collection.get(
                include=["documents", "metadatas"]
            )
            
            export_chunks = []
            for i in range(len(all_records['ids'])):
                chunk_id = all_records['ids'][i]
                
                document_content = all_records['documents'][i] if all_records.get('documents') is not None else None
                metadata = all_records['metadatas'][i] if all_records.get('metadatas') is not None else {}
                
                # Assemble the human-readable text packet for this chunk
                chunk_data = {
                    "chunk_id": chunk_id,
                    "metadata": metadata,
                    "document_content": document_content
                }
                export_chunks.append(chunk_data)
                
            # Construct destination path
            export_filename = f"db_export_{collection_name}.json"
            export_filepath = os.path.join(output_dir, export_filename)
            
            with open(export_filepath, 'w', encoding='utf-8') as json_file:
                json.dump({
                    "collection": collection_name,
                    "total_records": record_count,
                    "chunks": export_chunks
                }, json_file, indent=4, ensure_ascii=False)
                
            print(f"   💾 Clean text JSON dump saved to: data_out/{export_filename}")
                
        except Exception as ce:
            print(f"❌ Collection: {collection_name:<25} | Error: {str(ce)}")
            
    print("="*60 + "\n")

if __name__ == "__main__":
    # FIX: Aligned perfectly with definition name above
    run_database_diagnostics_and_text_export()