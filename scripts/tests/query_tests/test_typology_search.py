import json
import os
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv

from src.RAG.vectorstore import VectorStoreManager
from src.RAG.searchengine import ConstructionSearchService
from src.agents.query.query_typologies.agents import (
    IntentAgent,
    FunctionInputAgent,
    CleaningAgent,
    StructureResponseAgent,
)
from src.agents.query.query_typologies.execution import ExecutionAgent
from src.agents.query.query_typologies.orchestrator import PipelineOrchestrator
from src.utils.load_conf import load_config

load_dotenv()

DB_PATH = os.getenv("VECTOR_DB_PATH") or None
# Path to your structured typologies database dictionary file
TYPOLOGIES_DB_PATH = os.getenv("TYPOLOGIES_DB_PATH") or "./data_in/typologies.json"


def main():
    cfg_env = os.getenv("CONFIG_PATH") or None
    if not cfg_env:
        raise ValueError("CONFIG_PATH environment variable not set.")
    cfg = load_config(cfg_env)

    if not DB_PATH or not os.path.exists(DB_PATH):
        raise ValueError("VECTOR_DB_PATH variable does not exist or path is invalid.")
    if not os.path.exists(TYPOLOGIES_DB_PATH):
        raise ValueError(
            f"Typologies database dictionary file not found at: {TYPOLOGIES_DB_PATH}"
        )

    # 1. Initialize Underlying Data Collections
    vs_material = VectorStoreManager(
        db_path=DB_PATH,
        collection_name=cfg["vector_store"]["material_collection_name"],
        model_name=cfg["vector_store"]["model_name"],
    )
    material_search_service = ConstructionSearchService(vs_material)

    # Load physical structures catalog dict
    with open(TYPOLOGIES_DB_PATH, "r", encoding="utf-8") as f:
        constructions_mock_db = json.load(f)

    # 2. Build Multi-Agent Core Registry
    agents_registry = {
        "intent": IntentAgent(cfg),
        "param_extractor": FunctionInputAgent(cfg),
        "cleaner": CleaningAgent(cfg),
        "structurer": StructureResponseAgent(cfg),
    }

    # 3. Bind the Execution Target Filters
    executor = ExecutionAgent(database=constructions_mock_db)

    # 4. Instantiate the Main Orchestrator
    orchestrator = PipelineOrchestrator(
        agents=agents_registry,
        search_service=material_search_service,
        executor=executor,
    )

    # 5. Define a complex multi-criteria query targeting both Category and Layer Type
    user_query = "Find partitions containing high density gypsum board panels"
    print(f"🚀 Submitting multi-intent query to Pipeline Orchestrator: '{user_query}'\n")

    # 6. Execute progressive multi-filter RAG workflow
    # Pipeline now returns aggregated values across all processed specialist steps
    intents, extracted_params, mapped_params, final_results = orchestrator.run(
        user_query, top_k=3
    )

    # 7. Package Payload Results and Diagnostic Logs
    output_data = {
        "metadata": {
            "query": user_query,
            "timestamp": datetime.now().isoformat(),
            "pipeline_type": "Multi-Intent Multi-Filter Chain Execution (Track B - 20B Optimized)",
        },
        "orchestrator_diagnostics": {
            "classified_intents_executed": intents,
            "extracted_raw_parameters_accumulated": extracted_params,
            "vector_mapped_parameters_accumulated": mapped_params,
        },
        "results_count": len(final_results),
        "matched_typologies": final_results,
    }

    os.makedirs("test_results", exist_ok=True)
    output_path = "test_results/test_typology_pipeline_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Finished processing multi-filter intent query.")
    print(f"Executed Intents Sequence: {intents}")
    print(f"Total matching elements isolated: {len(final_results)}")
    print(f"Results outputted to: {output_path}")


if __name__ == "__main__":
    main()