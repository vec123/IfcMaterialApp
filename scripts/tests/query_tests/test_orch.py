import logging
from src.utils.load_conf import load_config
from src.RAG.vectorstore import VectorStoreManager
from src.RAG.searchengine import ConstructionSearchService
from src.agents.query.orchestrator import ConstructionOrchestrator

def run_integrated_test():
    # A. Load Real Infrastructure
    cfg = load_config("conf/config.yaml")
    vs_manager = VectorStoreManager(
        db_path=cfg['vector_store']['db_path'],
        collection_name=cfg['vector_store']['collection_name'],
        model_name=cfg['vector_store']['model_name']
    )
    search_service = ConstructionSearchService(vs_manager)
    
    # B. Load Mock Logic Database
    mock_db = {
        "W_BRICK_40": {"layers": [{"material": "Face Brick", "thickness": 0.11}, {"material": "Lana Mineral", "thickness": 0.04}]}
    }

    # C. Initialize Orchestrator
    orchestrator = ConstructionOrchestrator(
        database=mock_db,
        search_service=search_service,
        config=cfg
    )

    # D. Test Scenarios
    queries = [
        "Find me a 4cm layer of lana mineral", # Should trigger ExecutionAgent (Hard Logic)
        "high density gypsum boards"           # Should trigger VectorSearch (Semantic)
    ]

    for query in queries:
        print(f"\nProcessing: {query}")
        result = orchestrator.handle_query(query)
        print(f"Detected Intent: {result['intent']}")
        print(f"Results Found: {len(result['data'])}")

if __name__ == "__main__":
    run_integrated_test()