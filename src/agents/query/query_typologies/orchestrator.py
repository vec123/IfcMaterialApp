import logging
from typing import Tuple, Any, Dict, List

# Initialize module-level logger
logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    def __init__(self, agents: Dict[str, Any], search_service: Any, executor: Any):
        """
        Coordinates a progressive, multi-intent filter chain optimized 
        for highly focused smaller open-source LLMs.
        """
        # Safe lookups to match the exact keys returned by your loader
        self.intender = agents.get('intent') or agents.get('query_intent') or agents.get('intent_agent')
        self.extractor = agents.get('param_extractor') or agents.get('query_extractor') or agents.get('param_agent')
        self.cleaner = agents.get('cleaner')
        self.structurer = agents.get('structurer')
        
        self.search_service = search_service
        self.executor = executor
        
        # Hardened safety assertion
        if not self.intender or not self.extractor:
            raise KeyError(
                f"PipelineOrchestrator initialization failed. Required agent keys missing. "
                f"Provided: {list(agents.keys())}"
            )

    def run(self, query: str, top_k: int = 5) -> Tuple[List[str], Dict[str, Any], Dict[str, Any], List[Any]]:
        """
        Pipes data through extracted criteria sequentially using progressive dataset attrition.
        """
        # 1. Detect multi-intent presence tokens
        intents = self.intender.determine(query) or ["find_by_layer_type"]

        # Enforce deterministic optimization execution order (e.g. category drop bounds before vector math)
        execution_order = [
            "find_by_id", "find_by_category", "find_by_layer_type", 
            "filter_by_thickness", "filter_by_specific_layer", "filter_by_number_of_layers"
        ]
        active_ordered_intents = [intent for intent in execution_order if intent in intents] or intents

        accumulated_raw_params = {}
        accumulated_mapped_params = {}
        current_dataset_state = self.executor.database.copy()

        # 2. Sequential Filter Chain Processing Loop
        for active_intent in active_ordered_intents:
            raw_extracted = self.extractor.determine(active_intent, query)
            if not raw_extracted:
                continue

            accumulated_raw_params.update(raw_extracted)

            # Isolate semantic search vector space canonical expansions early
            if active_intent in ("find_by_layer_type", "filter_by_specific_layer"):
                mapped_subset = self.search_service.map_params_materials(raw_extracted.copy(), top_k=top_k)
            else:
                mapped_subset = raw_extracted.copy()

            accumulated_mapped_params.update(mapped_subset)

            # Execute step filter over the continually shrinking dataset state
            filter_func = self.executor.tool_map.get(active_intent)
            if filter_func:
                try:
                    current_dataset_state = self._apply_filter(
                        intent=active_intent,
                        filter_func=filter_func,
                        dataset=current_dataset_state,
                        params=mapped_subset
                    )
                except Exception as e:
                    logger.error(f"Execution error encountered processing filter stage '{active_intent}': {e}")

        # 3. Normalize dataset output footprint to standard list structures
        final_list = (
            list(current_dataset_state.values()) 
            if isinstance(current_dataset_state, dict) 
            else current_dataset_state
        )

        # 4. Final Layout Post-Ranking Structuring Pass
        primary_intent = "find_by_layer_type" if "find_by_layer_type" in intents else active_ordered_intents[0]
        final_results = self.structurer.structure(query, primary_intent, final_list)

        return active_ordered_intents, accumulated_raw_params, accumulated_mapped_params, final_results

    def _apply_filter(self, intent: str, filter_func: Any, dataset: Any, params: Dict[str, Any]) -> Any:
        """Dispatches parameters matching the precise historic signatures of tool components."""
        if intent == "find_by_id":
            return filter_func(constructions=dataset, ids=params.get("ids", []))
            
        elif intent == "find_by_category":
            return filter_func(constructions=dataset, category=params.get("category", ""))
            
        elif intent == "find_by_layer_type":
            return filter_func(material_names=params.get("materials", []), constructions_data=dataset)
            
        elif intent == "filter_by_thickness":
            return filter_func(
                constructions=dataset, 
                target_thickness=params.get("thickness", 0.0), 
                mode=params.get("mode", "total"), 
                tolerance=params.get("tolerance", 0.005)
            )
            
        elif intent == "filter_by_number_of_layers":
            return filter_func(
                constructions=dataset, 
                count=params.get("layer_count", 0), 
                operator=params.get("operator", "==")
            )
            
        elif intent == "filter_by_specific_layer":
            mat = params.get("material")
            if isinstance(mat, list):
                mat = mat[0] if mat else ""
            return filter_func(
                constructions=dataset, 
                target_material=mat, 
                target_thickness=params.get("target_thickness", 0.0), 
                tolerance=params.get("tolerance", 0.005)
            )
            
        return dataset