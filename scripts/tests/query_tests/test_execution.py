import json
import os
import logging
from datetime import datetime
from agents.query.agents import ExecutionAgent
from utils.load_conf import load_config
from src.RAG.vectorstore import VectorStoreManager
from src.RAG.searchengine import ConstructionSearchService

# Setup logging to see detailed filter matches
logging.basicConfig(level=logging.INFO)

def main():
    # 1. Setup Infrastructure & Config
    cfg = load_config("conf/config.yaml")
    
    # 2. Load the REAL Typology Database
    db_path = cfg.get('database', {}).get('path', 'data_in/constructions.json')
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            real_db = json.load(f)
        print(f"✅ Loaded Real DB: {len(real_db)} constructions found.")
    except Exception as e:
        print(f"❌ Failed to load real DB at {db_path}: {e}")
        return

    # 3. Setup Semantic Search Service
    vs_manager = VectorStoreManager(
        db_path=cfg['vector_store']['db_path'],
        collection_name=cfg['vector_store']['collection_name'],
        model_name=cfg['vector_store']['model_name']
    )
    search_service = ConstructionSearchService(vs_manager)

    # 4. Initialize Execution Agent with the REAL database
    executor = ExecutionAgent(cfg, real_db)

    # 5. Define Real-World Test Scenarios
    scenarios = [
        {
            "name": "REAL: Specific Insulation Match",
            "intent": "layer_search",
            "params": {"materials": ["lana mineral"], "thickness": 0.04}
        },
        {
            "name": "REAL: Total Thickness Filter",
            "intent": "assembly_search",
            "params": {"thickness": 0.30}
        },
        {
            "name": "REAL: Complexity Check",
            "intent": "complexity_search",
            "params": {"layer_count": 3}
        },
        {
            "name": "REAL: Semantic Vector Search",
            "intent": "semantic_search",
            "params": {"materials": ["placa de yeso laminado"]}
        }
    ]

    results = []

    print(f"\n{'='*60}")
    print("RUNNING EXECUTION TESTS AGAINST PRODUCTION DATABASE")
    print(f"{'='*60}\n")

    for test in scenarios:
        print(f"🚀 Running: {test['name']}")
        
        test_entry = {
            "test_name": test["name"],
            "intent": test["intent"],
            "params": test["params"],
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
            "result_count": 0,
            "sample_matches": []
        }
        
        try:
            if test['intent'] == "semantic_search":
                execution_results = {}
                for mat in test['params'].get("materials", []):
                    hits = search_service.search(mat, limit=3)
                    execution_results[mat] = hits
                # For semantic search, the count is the total number of hits across all materials
                count = sum(len(hits) for hits in execution_results.values())
                sample_ids = []
                for mat_hits in execution_results.values():
                    sample_ids.extend([h.get("id") for h in mat_hits if h.get("id")])
            else:
                execution_results = executor.execute(test['intent'], test['params'])
                count = len(execution_results)
                sample_ids = list(execution_results.keys())[:3]

            print(f"  📊 Status: Found {count} matches")
            test_entry["status"] = "success"
            test_entry["result_count"] = count
            test_entry["sample_matches"] = sample_ids
            
        except Exception as e:
            print(f"  ❌ Error executing {test['intent']}: {e}")
            test_entry["status"] = "error"
            test_entry["error_message"] = str(e)
        
        results.append(test_entry)
        print("-" * 40)

    # 6. Save logs for audit
    output_path = "test_execution_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"--- Test Complete. Results saved to {output_path} ---")

if __name__ == "__main__":
    main()
