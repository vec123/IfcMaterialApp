# src/query/typology_orchestrator.py

import logging
from typing import Tuple, Dict, Any, List
import src.agents.ifc.query_typology.typology_filters as tf

logger = logging.getLogger(__name__)

class TypologyPipelineOrchestrator:
    def __init__(self, agents: Dict[str, Any], search_service: Any, typology_database: Dict[str, Any]):
        """
        Coordinates complex multi-intent filtering queries over a relational typology schema dataset.
        """
        self.intender = agents.get("intent")
        self.extractor = agents.get("extractor")
        self.search_service = search_service  # Vector store lookup service for material naming alignment
        self.database = typology_database

        # Deterministic reduction map linking intents to functions
        self.tool_map = {
            "find_by_typology_id":       tf.filter_by_typology_id,
            "filter_by_element_class":   tf.filter_by_element_class,
            "filter_by_layer_count":     tf.filter_by_layer_count,
            "filter_by_material_mix":    tf.filter_by_material_mix,
            "filter_by_total_thickness": tf.filter_by_total_thickness,
        }

    def run(self, query: str, vector_limit: int = 5) -> Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]]:
        # 1. Determine all implicit constraints present in the query
        intents = self.intender.determine(query) or ["filter_by_material_mix"]
        
        # Define sequential order: filter out metadata blocks before running material scans
        execution_order = [
            "find_by_typology_id", "filter_by_element_class", 
            "filter_by_layer_count", "filter_by_total_thickness", "filter_by_material_mix"
        ]
        active_ordered_intents = [i for i in execution_order if i in intents] or intents
        
        current_state = self.database.copy()
        accumulated_params = {}

        # 2. Sequential Reduction Engine
        for intent in active_ordered_intents:
            extracted = self.extractor.determine(intent, query)
            if not extracted:
                continue

            # Route natural language material terms through vector expansion
            if intent == "filter_by_material_mix" and "materials" in extracted:
                raw_materials = extracted.get("materials", [])
                aligned_materials = []
                
                for mat in raw_materials:
                    # Leverage the existing VectorStore mapping infrastructure
                    hits = self.search_service.search(mat, limit=1)
                    if hits:
                        hit_id = getattr(hits[0], "id", None) or hits[0].get("id")
                        aligned_materials.append(hit_id)
                    else:
                        aligned_materials.append(mat)
                        
                extracted["materials"] = aligned_materials

            accumulated_params[intent] = extracted

            # Apply execution step filters
            filter_func = self.tool_map.get(intent)
            if filter_func:
                try:
                    current_state = filter_func(typologies=current_state, **extracted)
                except Exception as e:
                    logger.error(f"Typology reduction stage failed processing '{intent}': {e}")

        # Convert back to a standardized list output for rendering
        final_results = list(current_state.values())
        return active_ordered_intents, accumulated_params, final_results