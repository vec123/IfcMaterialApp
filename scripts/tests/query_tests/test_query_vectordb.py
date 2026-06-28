import json
import os
from datetime import datetime
from src.RAG.vectorstore import VectorStoreManager
from src.RAG.searchengine import ConstructionSearchService
from src.graph.navigator import GraphNavigationService
from src.agents.query.query_typologies.agents import ExtractionAgent, CleaningAgent
from src.utils.load_conf import load_config
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("VECTOR_DB_PATH") or None
GRAPH_EDGES_PATH = os.getenv("GRAPH_EDGES_PATH") or "./data_out/graph_relationships.json"

if None in [DB_PATH, GRAPH_EDGES_PATH]: 
    raise ValueError("Environment Paths incomplete")

def main():
    cfg_env = os.getenv("CONFIG_PATH") or None
    if not cfg_env:
        raise ValueError("CONFIG_PATH environment variable not set. Please set it to the path of your config.yaml")
    cfg = load_config(cfg_env)
    
    if not DB_PATH or not os.path.exists(DB_PATH):
        raise ValueError("VECTOR_DB_PATH variable does not exist or path is invalid")
    
    # 1. Instantiate Managers
    vs_material = VectorStoreManager(
        db_path=DB_PATH,
        collection_name=cfg['vector_store']['material_collection_name'],
        model_name=cfg['vector_store']['model_name']
    )
    
    vs_typology = VectorStoreManager(
        db_path=DB_PATH,
        collection_name=cfg['vector_store']['typology_collection_name'],
        model_name=cfg['vector_store']['model_name']
    )
    
    # Initialize Core Search & Graph Services
    material_search_service = ConstructionSearchService(vs_material)
    graph_service = GraphNavigationService(vs_typology, GRAPH_EDGES_PATH)
    
    # Initialize Extraction Agents
    extractor = ExtractionAgent(cfg)
    cleaner = CleaningAgent(cfg) 

    # 2. Process Query
    user_input = "I'm looking for high density gypsum boards and good thermal insulation."
    
    print("Calling ExtractionAgent with user input:", user_input)
    raw_extracted = extractor.process(user_input)
    
    target_schema = {
        "materials": [],
        "thickness": None,
        "layer_count": None
    }

    extracted_data = cleaner.clean(str(raw_extracted), target_schema=target_schema)
    materials = extracted_data.get("materials", [])
    properties = {
        "thickness": extracted_data.get("thickness"),
        "layer_count": extracted_data.get("layer_count")
    }

    output_data = {
        "metadata": {
            "original_query": user_input,
            "timestamp": datetime.now().isoformat(),
            "extracted_properties": properties,
            "extracted_materials": materials
        },
        "search_results": []
    }

    if not materials:
        materials = [user_input]

    # 5. EXECUTE THE GRAPH-RAG HYBRID SEARCH LOOP
    for material in materials:
        print(f"\n🔍 Searching for material node matching: '{material}'")
        
        # Search the material database collection explicitly
        hits = material_search_service.search(material, limit=2)
        
        material_entry = {
            "target_material": material,
            "matched_materials": [],
            "connected_typologies": []
        }
        
        for h in hits:
            canonical_material_id = h["id"]  # Spanish name string used as ID
            
            material_entry["matched_materials"].append({
                "canonical_name": canonical_material_id,
                "score": round(h.get("score", 0), 4),
                "properties": h.get("metadata", {})
            })
            
            # --- FIXED: Direct clean graph traversal call ---
            typologies = graph_service.get_connected_typologies(canonical_material_id)
            material_entry["connected_typologies"].extend(typologies)
                
        output_data["search_results"].append(material_entry)

    # 6. Save Unified Graph Output Results
    os.makedirs("test_results", exist_ok=True)
    with open("test_results/test_query_vectordb_results.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Finished processing query. Graph results outputted to test_results/test_query_vectordb_results.json")

if __name__ == "__main__":
    main()