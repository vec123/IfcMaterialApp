import json
import os
from datetime import datetime
from src.agents.query.agents import IntentAgent, CleaningAgent
from src.utils.load_conf import load_config

def main():
    # 1. Setup
    cfg = load_config("conf/config.yaml")
    intender = IntentAgent(cfg)
    cleaner = CleaningAgent(cfg)
    
    # Define the target schema for the intent response
    # This ensures we always get a dict with the 'intent' key
    intent_schema = {"intent": "string"}

    # 2. Test Scenarios
    test_queries = [
        "Find me a wall with a 4cm layer of insulation",       # Expected: layer_search
        "I need a 20cm concrete assembly",             # Expected: assembly_search
        "Give me high density gypsum boards",          # Expected: semantic_search
        "Show me constructions with exactly 5 layers", # Expected: complexity_search
        "Tell me about sustainable bricks"             # Expected: semantic_search
    ]

    results = []

    print(f"--- Starting Intent Classification Test ---")

    for query in test_queries:
        # Get raw response from IntentAgent
        print(f"Query: {query}")
        raw_intent = intender.determine(query)
        print(f"-----------> Raw intent response for query '{query}': {raw_intent}")
        # Use Cleaner to ensure we have a valid dictionary
        # We cast to string because cleaner expects a string input to regex search
        cleaned_data = cleaner.clean(str(raw_intent), target_schema=intent_schema)
        print(f"-----------> Cleaned intent data for query '{query}': {cleaned_data}")
        intent_value = cleaned_data.get("intent", "general_filter")
        print(f"-----------> Determined intent for query '{query}': {intent_value}")
        
        results.append({
            "query": query,
            "detected_intent": intent_value,
            "raw_response": raw_intent,
            "timestamp": datetime.now().isoformat()
        })

    # 3. Save logs for audit
    output_path = "test_intent_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"--- Test Complete. Results saved to {output_path} ---")

if __name__ == "__main__":
    main()