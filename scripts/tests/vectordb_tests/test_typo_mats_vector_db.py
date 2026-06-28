import os
import sys
import json
import logging
from sentence_transformers import SentenceTransformer
import chromadb

# ── 1. PROJECT PATH RESOLUTION & CONFIG INITIALIZATION ─────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.load_conf import build_cfg  # Import your native custom config framework

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GraphRAGTester")

def verify_graph_edges(edges_path: str):
    """Validates the integrity and structural schema of the exported graph edges."""
    print("\n" + "─"*60)
    print(" 🛠️  VALIDATING GRAPH RELATIONSHIPS (EDGES)")
    print("─"*60)
    
    if not edges_path or not os.path.exists(edges_path):
        print(f"⚠️  Graph Edges file missing or not generated yet at: {edges_path}")
        return

    try:
        with open(edges_path, "r", encoding="utf-8") as f:
            edges_data = json.load(f)
            
        total_edges = len(edges_data)
        print(f"🔗 File Found: {os.path.basename(edges_path)} | Status: ONLINE | Total Edges: {total_edges}")
        
        if total_edges > 0:
            print("   📋 Sample Edge Schema Verification:")
            # Display up to 2 sample structural edges for validation
            for idx, edge in enumerate(edges_data[:2]):
                print(f"      🔹 Edge [{idx+1}]: Typology '{edge.get('typology_id')}' ──(layer: {edge.get('layer_position')})──> Material '{edge.get('material_id')}' [Thickness: {edge.get('thickness')}m]")
        else:
            logger.warning("The graph edges file is an empty JSON array.")
            
    except Exception as e:
        print(f"❌ Graph Edges Validation Failed | Error: {str(e)}")

def run_graph_database_diagnostics():
    logger.info("🚀 Initializing Graph-RAG Vector Database & Schema Integrity Diagnostics...")
    
    # Load configuration using your native workspace builder
    cfg = build_cfg()
    
    # Extract absolute paths resolved by build_cfg()
    absolute_db_path = cfg['paths'].get('vector_db_path')
    graph_edges_path = cfg['paths'].get('graph_edges_path')
    output_dir       = os.path.abspath(os.path.join(project_root, "data_out"))
    
    vs_config = cfg['vector_store']
    collections_to_test = [
        vs_config['typology_collection_name'],
        vs_config['material_collection_name']
    ]
    
    # Enforce CPU execution for diagnostics testing to save developer resources
    model_name = vs_config['embedding_model']
    device = "cpu"
    logger.info(f"🧠 Loading Local Embedding Engine: {model_name} onto [{device}]...")
    embedding_engine = SentenceTransformer(model_name, device=device)
    
    # Connect to persistent Chroma store instance
    logger.info(f"💾 Connecting to local vector store at: {absolute_db_path}")
    chroma_client = chromadb.PersistentClient(path=absolute_db_path)
    
    print("\n" + "="*60)
    print("        VECTOR DATABASE INTEGRITY METRICS & TEXT EXPORTS")
    print("="*60)
    
    # Iterate and run test queries on both Typology and Material Collections
    for collection_name in collections_to_test:
        try:
            collection = chroma_client.get_collection(name=collection_name)
            record_count = collection.count()
            print(f"📦 Collection: {collection_name:<25} | Status: ONLINE | Records: {record_count}")
            
            if record_count == 0:
                logger.warning(f"Collection '{collection_name}' contains no vectors. Skipping query check.")
                continue
                
            # ── A. RUN SEMANTIC SEARCH QUERY TEST ───────────────────────────
            if "typo" in collection_name.lower():
                test_query = "Cubierta Plana Invertida"
            else:
                test_query = "Hormigón estructural armado"
                
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
                    print(f"      🔹 Match [{idx+1}]: {doc[:65]}... (Distance: {distance:.4f})")
            
            # ── B. EXTRACT CLEAN TEXT PACKETS WITHOUT 1024-VEC OVERHEAD ────
            all_records = collection.get(
                include=["documents", "metadatas"]
            )
            
            export_chunks = []
            for i in range(len(all_records['ids'])):
                chunk_data = {
                    "chunk_id": all_records['ids'][i],
                    "metadata": all_records['metadatas'][i] if all_records.get('metadatas') is not None else {},
                    "document_content": all_records['documents'][i] if all_records.get('documents') is not None else None
                }
                export_chunks.append(chunk_data)
                
            # Write out clean human-readable JSON payload
            os.makedirs(output_dir, exist_ok=True)
            export_filename = f"db_export_{collection_name}.json"
            export_filepath = os.path.join(output_dir, export_filename)
            
            with open(export_filepath, 'w', encoding='utf-8') as json_file:
                json.dump({
                    "collection": collection_name,
                    "total_records": record_count,
                    "chunks": export_chunks
                }, json_file, indent=4, ensure_ascii=False)
                
            print(f"   💾 Human-readable text dump saved to: data_out/{export_filename}")
                
        except Exception as ce:
            print(f"❌ Collection: {collection_name:<25} | Error: {str(ce)}")
            
    print("="*60)
    
    # Run the edge structural map check
    verify_graph_edges(graph_edges_path)

if __name__ == "__main__":
    run_graph_database_diagnostics()