from typing import List, Dict, Any, Optional, Tuple, Union
from src.agents.core.core import BaseLLMAgent, logger
import src.query.filters as cl
import json

class IntentAgent(BaseLLMAgent):
    """Classifies all distinct things the user wants to do in a single phrase."""

    def determine(self, user_input: str) -> List[str]:
        prompt = f"""
        Analyze the user query and identify ALL intents present.
        If the user specifies multiple constraints, return ALL matching intent keys separated by commas.

        Options:
        "find_by_id"               - user asks for a specific construction by its ID (e.g. S01).
        "find_by_category"         - user wants a specific architectural category (Fachada, Cubierta, Suelo, Tabiquería, Partición Vertical).
        "filter_by_thickness"      - user specifies a thickness value.
        "filter_by_specific_layer" - user specifies a material AND a thickness in the same layer.
        "filter_by_number_of_layers" - user specifies a count/number of layers.
        "find_by_layer_type"       - user wants constructions containing a specific material or layer type.

        User Input: "{user_input}"

        Return ONLY a JSON object containing an array of strings:
        {{"intents": ["intent_key_1", "intent_key_2"]}}
        """
        try:
            raw = self._call_llm(prompt, is_json=True)
            clean_raw = self._extract_json_string(raw)
            data = json.loads(clean_raw)
            intents = data.get("intents", [])
            return [intents] if isinstance(intents, str) else intents
        except Exception:
            return ["find_by_layer_type"]  # Safe baseline fallback

class FunctionInputAgent(BaseLLMAgent):
    def __init__(self, config):
        super().__init__(config)
        self.specialists = {
            "find_by_id":                IDInputAgent(config),
            "find_by_category":          CategoryInputAgent(config),
            "filter_by_thickness":       ThicknessInputAgent(config),
            "filter_by_specific_layer":  SpecificLayerInputAgent(config),
            "filter_by_number_of_layers": LayerCountInputAgent(config),
            "find_by_layer_type":        MaterialsInputAgent(config),
            "find_by_multiple_layer_type": MaterialsInputAgent(config),
        }

    def get_schema(self, intent: str) -> Dict[str, str]:
        """Retrieves the expected schema for the chosen intent."""
        agent = self.specialists.get(intent)
        return agent.schema if agent else {}

    def determine(self, intent: str, user_input: str) -> Dict[str, Any]:
        agent = self.specialists.get(intent)
        if not agent:
            logger.warning(f"No specialist for intent: {intent}")
            return {}
        return agent.determine(user_input)

class ThicknessInputAgent(BaseLLMAgent):
    schema = {
        "thickness": "float", 
        "mode": "string", 
        "tolerance": "float"
    }

    def determine(self, user_input: str) -> Dict[str, Any]:
        prompt = f"""Extract parameters for: "{user_input}"
        Return ONLY JSON:
        {{
            "thickness": float (meters), 
            "mode": "total" | "individual", 
            "tolerance": float (default 0.005)
        }}"""
        return self._call_and_parse(prompt)

class SpecificLayerInputAgent(BaseLLMAgent):
    schema = {
        "material": "string", 
        "target_thickness": "float", 
        "tolerance": "float", 
        "include_eat": "boolean", 
        "include_nan": "boolean"
    }

    def determine(self, user_input: str) -> Dict[str, Any]:
        prompt = f"""Extract specific layer details for: "{user_input}"
        Return ONLY JSON:
        {{
            "material": str, 
            "target_thickness": float (meters), 
            "tolerance": float, 
            "include_eat": bool, 
            "include_nan": bool
        }}"""
        return self._call_and_parse(prompt)

class LayerCountInputAgent(BaseLLMAgent):
    schema = {
        "layer_count": "integer", 
        "operator": "string"
    }

    def determine(self, user_input: str) -> Dict[str, Any]:
        prompt = f"""Extract layer count details for: "{user_input}"
        Return ONLY JSON:
        {{
            "layer_count": int, 
            "operator": "==" | ">" | "<" | ">=" | "<="
        }}"""
        return self._call_and_parse(prompt)

class MaterialsInputAgent(BaseLLMAgent):
    schema = {
        "layertype": "string"
    }

    def determine(self, user_input: str) -> Dict[str, Any]:
        prompt = f"""
        Extract the layer-type and or material from: "{user_input}"
        Arragne them as a comma-separated strings. 
        If it is a single material, return a single string.
        OUTPUT FORMAT: Return ONLY a JSON object. No prose.
        EXAMPLE: {{"materials": string}} 
        where string is either a single material/type or a comma-separated list of materials/types.
        """
        return self._call_and_parse(prompt)


class IDInputAgent(BaseLLMAgent):
    schema = {"ids": "list of strings"}

    def determine(self, user_input: str) -> Dict[str, Any]:
        prompt = f"""Extract the construction ID(s) from: "{user_input}"
        IDs follow the pattern "S" followed by digits, e.g. "S01", "S123".
        Return ONLY JSON: {{"ids": ["S01", ...]}}"""
        return self._call_and_parse(prompt)


class CategoryInputAgent(BaseLLMAgent):
    schema = {"category": "string"}
    KNOWN = ["Cubierta", "Fachada", "Partición Vertical", "Suelo", "Tabiquería"]

    def determine(self, user_input: str) -> Dict[str, Any]:
        prompt = f"""Identify the construction category from: "{user_input}"
        Known categories: {self.KNOWN}
        Common mappings: wall/facade→Fachada, roof→Cubierta, floor/slab→Suelo,
        partition→Partición Vertical, light partition/drywall→Tabiquería.
        Return ONLY JSON: {{"category": "exact category name from the list above"}}"""
        return self._call_and_parse(prompt)


class CleaningAgent(BaseLLMAgent):
    def clean(self, raw_data: str, target_schema: Dict[str, Any]) -> Dict[str, Any]:
        cleaned_str = self._extract_json_string(raw_data)

        try:
            return json.loads(cleaned_str)
        except json.JSONDecodeError:
            # The repair prompt needs to be much stricter
            repair_prompt = f"""
            CRITICAL: The following string contains valuable data but is not valid JSON. 
            Fix the formatting so it matches this schema: {json.dumps(target_schema)}
            
            Original String: {cleaned_str}
            Return ONLY the valid JSON.
            """
            repaired_raw = self._call_llm(repair_prompt, is_json=True)
            try:
                return json.loads(self._extract_json_string(repaired_raw))
            except:
                print("CleaningAgent: Critical failure. Returning default schema.")
                # Fallback: if repair fails, return the schema with empty values 
                # but at least the script doesn't crash.
                return {k: ([] if v == "list" else None) for k, v in target_schema.items()}
            


class StructureResponseAgent(BaseLLMAgent):
    """
    Super-agent that dispatches result-structuring to an intent-specific
    subagent, mirroring how FunctionInputAgent dispatches param extraction.

    The subagent decides how to reorder the executor's output for that intent.
    Intents without a registered specialist return their input unchanged.
    """

    def __init__(self, config):
        super().__init__(config)
        self.specialists = {
            "find_by_layer_type": MaterialStructureSubagent(config),
        }

    def structure(
        self,
        user_input: str,
        intent: str,
        results: Union[Dict[str, Any], List[Any]],
    ) -> Union[Dict[str, Any], List[Any]]:
        if not results or len(results) < 2:
            return results
        specialist = self.specialists.get(intent)
        if specialist is None:
            return results
        return specialist.structure(user_input, results)


class MaterialStructureSubagent(BaseLLMAgent):
    """
    Structuring subagent for the `find_by_layer_type` intent.

    Strategy:
    1. Collect every unique material name appearing in any layer of the matched
       typologies.
    2. Ask the LLM which subset of those materials is most relevant to the user's
       prompt (semantic / synonym match against the user's wording).
    3. Reorder so typologies containing at least one relevant material come
       first, and the rest come last. Original relative order is preserved
       inside each group.
    """

    def structure(
        self,
        user_input: str,
        results: Union[Dict[str, Any], List[Any]],
    ) -> Union[Dict[str, Any], List[Any]]:
        is_dict = isinstance(results, dict)
        items: List[Tuple[Any, Dict[str, Any]]] = (
            list(results.items()) if is_dict else list(enumerate(results))
        )

        unique_materials = sorted({
            str(layer.get("material")).strip()
            for _, info in items
            for layer in (info.get("layers", []) if isinstance(info, dict) else [])
            if layer.get("material") is not None
            and str(layer.get("material")).strip().lower() not in ("", "nan")
        })

        if not unique_materials:
            return results

        prompt = f"""
        From this list of materials present across the matched construction
        typologies, identify which materials are MOST RELEVANT to the user's
        request. Match by meaning / synonym, not just by exact string.

        User Query: "{user_input}"

        Available materials:
        {json.dumps(unique_materials, ensure_ascii=False)}

        Return ONLY JSON of the form:
        {{"relevant_materials": ["<material from the list above>", ...]}}

        If none of the available materials match the user's request, return an
        empty list.
        """

        raw = self._call_llm(prompt, is_json=True)
        try:
            data = json.loads(self._extract_json_string(raw))
            relevant = data.get("relevant_materials", [])
        except Exception as e:
            logger.error(f"MaterialStructureSubagent parse failed: {e}")
            return results

        if not isinstance(relevant, list) or not relevant:
            return results

        relevant_lower = {str(m).strip().lower() for m in relevant if m}

        def contains_relevant(info: Any) -> bool:
            if not isinstance(info, dict):
                return False
            for layer in info.get("layers", []):
                mat = str(layer.get("material") or "").strip().lower()
                if mat in relevant_lower:
                    return True
            return False

        front_items: List[Tuple[Any, Dict[str, Any]]] = []
        back_items: List[Tuple[Any, Dict[str, Any]]] = []
        for key, info in items:
            (front_items if contains_relevant(info) else back_items).append((key, info))

        ordered = front_items + back_items
        if is_dict:
            return {k: v for k, v in ordered}
        return [v for _, v in ordered]


class ExtractionAgent(BaseLLMAgent):
    """Extracts construction parameters from raw text."""
    
    def process(self, user_input: str) -> str:
        # Note: We return the string here so the CleaningAgent can handle it
        prompt = f"""
        Extract construction parameters from: "{user_input}"
        Materials: list of names.
        Thickness: float in meters (e.g. 4cm -> 0.04).
        
        Return JSON structure: 
        {{"materials": [list of strings]}}
        """
        return self._call_llm(prompt, is_json=True)
    


class RefiningAgent(BaseLLMAgent):
    """
    Acts as a validation layer to ensure discrete values, units, 
    and data types strictly follow the ExecutionAgent's requirements.
    """
    def refine(self, intent: str, extracted_params: Dict[str, Any]) -> Dict[str, Any]:
        # Define the strict discrete boundaries
        allowed_values = {
            "mode": ["total", "individual"],
            "operator": ["==", ">", "<", ">=", "<="],
            "booleans": [True, False]
        }
        prompt = f"""
        Refine the following JSON for the intent "{intent}".
        
        STRICT VALIDATION RULES:
        1. "mode" must be EXACTLY one of: {allowed_values['mode']}.
        2. "operator" must be EXACTLY one of: {allowed_values['operator']}.
        3. "thickness" and "tolerance" MUST be floats (meters).
        4. "layer_count" MUST be an integer.
        5. "include_eat" and "include_nan" MUST be booleans.
        
        Input Data to fix:
        {json.dumps(extracted_params)}

        If a value is invalid or missing, substitute with the logical default for that intent.
        Return ONLY the corrected JSON.
        """
        
        raw = self._call_llm(prompt, is_json=True)
        try:
            clean_raw = self._extract_json_string(raw)
            refined_data = json.loads(clean_raw)
            
            # Final Python-side safety check for discrete values
            if "mode" in refined_data and refined_data["mode"] not in allowed_values["mode"]:
                refined_data["mode"] = "total"
                
            if "operator" in refined_data and refined_data["operator"] not in allowed_values["operator"]:
                refined_data["operator"] = "=="

            return refined_data
        except Exception as e:
            logger.error(f"RefiningAgent critical failure: {e}")
            return extracted_params