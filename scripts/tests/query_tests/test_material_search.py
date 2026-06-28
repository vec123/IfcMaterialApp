import json
import os
from datetime import datetime
from dotenv import load_dotenv

from src.RAG.vectorstore import VectorStoreManager
from src.RAG.searchengine import ConstructionSearchService
from src.agents.query.query_mats.query_mat_agent import SearchMaterialAgent
from src.utils.load_conf import load_config

load_dotenv()

DB_PATH = os.getenv("VECTOR_DB_PATH") or None

def main():
    cfg_env = os.getenv("CONFIG_PATH") or None
    if not cfg_env:
        raise ValueError("CONFIG_PATH environment variable not set.")
    cfg = load_config(cfg_env)
    
    if not DB_PATH or not os.path.exists(DB_PATH):
        raise ValueError("VECTOR_DB_PATH variable does not exist or path is invalid.")
    
    # 1. Initialize DB and Search Service for Materials Collection
    vs_material = VectorStoreManager(
        db_path=DB_PATH,
        collection_name=cfg['vector_store']['material_collection_name'],
        model_name=cfg['vector_store']['model_name']
    )
    material_search_service = ConstructionSearchService(vs_material)
    
    # 2. Initialize the dedicated Track A Orchestrator Agent
    material_agent = SearchMaterialAgent(config=cfg, search_service=material_search_service, verbose=True)
    
    # 3. Define a query with cross-lingual and descriptive nuances
    user_query = "I need rigid thermal insulation panels with high density"
    print(f"🚀 Running Material Search Pipeline for: '{user_query}'\n")
    
    # 4. Run the automated pipeline (Extract -> Vector Search -> LLM Cross-Encode Rank)
    ranked_results = material_agent.search(user_query, top_k=5)
    
    # 5. Format and Save Outputs
    output_data = {
        "metadata": {
            "query": user_query,
            "timestamp": datetime.now().isoformat(),
            "pipeline_type": "Material Search & LLM Re-ranking (Track A)"
        },
        "ranked_materials": ranked_results
    }
    
    os.makedirs("test_results", exist_ok=True)
    output_path = "test_results/test_material_search_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Finished processing material query.")
    print(f"Results outputted to: {output_path}")

if __name__ == "__main__":
    main()