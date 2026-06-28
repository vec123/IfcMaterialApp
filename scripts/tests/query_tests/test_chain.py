import json
from datetime import datetime
from src.agents.query.agents import (
    IntentAgent, 
    FunctionInputAgent, 
    CleaningAgent,
    RefiningAgent,
    ExecutionAgent
)
from src.utils.load_conf import load_config


from src.RAG.vectorstore import VectorStoreManager
from src.RAG.searchengine import ConstructionSearchService

def main():
    # 1. Setup
    cfg = load_config("conf/config.yaml")
    intender = IntentAgent(cfg)
    param_extractor = FunctionInputAgent(cfg)
    cleaner = CleaningAgent(cfg)
    refiner = RefiningAgent(cfg)

    construction_db_path = "./data_in/constructions.json"  
    with open(construction_db_path, 'r', encoding='utf-8') as f:
        construction_db = json.load(f)
    full_length = len(construction_db)
    executor = ExecutionAgent(database=construction_db)

    vs_manager = VectorStoreManager(
        db_path=cfg['vector_store']['db_path'],
        collection_name=cfg['vector_store']['collection_name'],
        model_name=cfg['vector_store']['model_name']
    )
    search_service = ConstructionSearchService(vector_store=vs_manager) 

    intent_schema = {"intent": "string"}
    # Schema for parameter validation
    param_schema = {
        "materials": "list",
        "thickness": "float",
        "mode": "string",
        "layer_count": "integer",
        "operator": "string",
        "tolerance": "float",
        "include_eat": "boolean",
        "include_nan": "boolean"
    }

    # 2. Test Scenarios specifically designed to trigger different parameters
    test_queries = [
        "Find me a wall with mineral wool",       # layer_property_search / 0.04
        "I need a 20cm concrete assembly",               # assembly_search / 0.2
        "Show me constructions with more than 5 layers", # complexity_search / > 5
        "Search for assemblies with brick and wood",     
        "Search for assemblies with concrete",   
        "Search for typoogies with gypsum",   
        "Search for typoogies with ventilated facade",     
         "Search for typoogies with ventilated air chamber",    
      #  "Find typologies for this description TRA_GuixLaminat_5cm ",             # assembly_search / 0.04 / 0.01
       #  "Find typologies for this description DIV_MaoCalat_15cm ",   
       #  "Find typologies for this description REV_RevestimentGuixLaminat_65mm " ,
       #  "Find typologies for this description FON_MurPantallaFormigoArmat_500mm " ,
    ]




    results = []

    print(f"--- Starting End-to-End Chain Test ---")

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # STEP Detect Intent
        raw_intent = intender.determine(query)
        print(f"  [1] Raw Intent: {raw_intent}")
        cleaned_intent_dict = cleaner.clean(str(raw_intent), target_schema=intent_schema)
        print(f"  [1] Cleaned Intent Dict: {cleaned_intent_dict}")
        intent_value = cleaned_intent_dict.get("intent", "general_filter")
        
        print(f"  [1] Intent: {intent_value}")

        # STEP  Extract Parameters (The new FunctionInputAgent)
        current_schema = param_extractor.get_schema(intent_value)
        extracted_params = param_extractor.determine(intent_value, query)
        print(f"  [2] Extracted Parameters: {extracted_params}")
        cleaned_params = extracted_params
       #cleaned_params = cleaner.clean(str(extracted_params), target_schema=current_schema)
        
        mapped_params = search_service.map_params_materials(cleaned_params)
        print(f"  [2] Mapped_params Parameters: {mapped_params}")

        final_constructions = executor.execute(intent_value, mapped_params)
        
        results.append({
            "query": query,
            "intent": intent_value,
            "parameters": extracted_params,
            "mapped_params": mapped_params,
            "full_length": full_length,
            "matches_found": len(final_constructions),
            "timestamp": datetime.now().isoformat()
        })

    #  Save logs
    output_path = "test_chain_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n--- Test Complete. Chain results saved to {output_path} ---")

if __name__ == "__main__":
    main()