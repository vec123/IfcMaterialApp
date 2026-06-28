import json
import re
from datetime import datetime
from src.RAG.vectorstore import VectorStoreManager
from src.RAG.searchengine import ConstructionSearchService
from src.agents.query.query_agent import QueryAgent
from utils.load_conf import load_config

def technical_clean(ifc_name: str) -> str:
    """
    Strips IFC prefixes and dimensions to isolate the material name.
    Example: 'FAC_Rasilla_5cm' -> 'Rasilla'
    """
    # 1. Remove known IFC prefixes at the start
    prefixes = r'^(DIV|FAC|TRA|EST|FON|REV|EST-V|EST-H)_'
    clean_name = re.sub(prefixes, '', ifc_name)
    
    # 2. Remove trailing dimensions (e.g., _5cm, _150mm)
    clean_name = re.sub(r'_\d+(cm|mm)$', '', clean_name)
    
    # 3. Replace remaining underscores with spaces
    return clean_name.replace("_", " ")

def main():
    # 1. Load Config and Data
    cfg = load_config("conf/config.yaml")
    with open('data_in/test_walls.json', 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)

    # 2. Initialize Services
    vs_manager = VectorStoreManager(
        db_path=cfg['vector_store']['db_path'],
        collection_name=cfg['vector_store']['collection_name'],
        model_name=cfg['vector_store']['model_name']
    )
    search_service = ConstructionSearchService(vs_manager)
    agent = QueryAgent(cfg)

    # 3. Storage for final results
    final_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "source_file": "ifc_mapping.json"
        },
        "mappings": []
    }

    # 4. Process each element
    for entry in mapping_data["IFC_Mapping"]:
        raw_name = entry["element_name"]
        typology_id = entry["typology_id"]
        raw_name = technical_clean(raw_name)  # Clean the IFC name to isolate material info
        print(f"🛠️  Processing: {raw_name} ({typology_id})")

        # Optional: Replace underscores with spaces to help the Agent/LLM
        sanitized_query = raw_name.replace("_", " ")
        
        # Extract materials via Agent
        extracted = agent.process_query(sanitized_query)
        materials = extracted.get("materials", [])

        element_match_data = {
            "ifc_element": raw_name,
            "typology_id": typology_id,
            "extracted_materials": materials,
            "vector_matches": []
        }

        # Query Vector DB for each extracted material
        for material in materials:
            hits = search_service.search(material, limit=5)
            element_match_data["vector_matches"].append({
                "search_term": material,
                "hits": [
                    {
                        "material_id": h["id"],
                        "score": round(h["score"], 4),
                        "properties": h["metadata"]
                    } for h in hits
                ]
            })

        final_results["mappings"].append(element_match_data)

    # 5. Export results
    output_file = "ifc_query_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Done! Results saved to {output_file}")

if __name__ == "__main__":
    main()