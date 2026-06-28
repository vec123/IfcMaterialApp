
from typing import Any, Dict, List
import json
import os
import re
from src.agents.core.core import BaseLLMAgent, logger


class MaterialExtensionAgent(BaseLLMAgent):
    """
    Extracts, cleans, and translates material names into multilingual search variations,
    anchoring results against a predefined dictionary of IFC architectural materials.
    """
    def __init__(self, config: dict, anchor_file_path: str = None, verbose: bool = False):
        super().__init__(config, verbose=verbose)

    def _normalize_key(self, text: str) -> str:
        """Cleans strings to ensure reliable dictionary lookups (lowercase, alphanumeric only)."""
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return " ".join(text.split())

    def _load_anchors(self, file_path: str) -> Dict[str, Any]:
        """Loads the localized anchor strings from a JSON file and pre-normalizes keys."""
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"Anchor file not found at '{file_path}'. Operating without anchors.")
            return {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                raw_anchors = data.get("anchors", {})
                # Normalize keys immediately to enable robust validation later
                return {self._normalize_key(k): v for k, v in raw_anchors.items()}
        except Exception as e:
            logger.error(f"Failed to load anchor file: {e}")
            return {}

    def extract(self, user_input: str) -> Dict[str, Any]:
        # 1. Pre-check: If user input directly maps to an anchor, short-circuit immediately
       # normalized_input = self._normalize_key(user_input)
       # if normalized_input in self.anchors:
       #     es_name = anchor_data.get("es", {}).get("name", user_input)
       #     en_name = anchor_data.get("en", {}).get("name", user_input)
       #     return {
       #         "name": es_name,
       #         "variations": [es_name, en_name]
       #     }

        # 2. Defensively engineered prompt to prevent conversation & systemic hallucinations
        prompt = f"""
        Analyze the following IFC/BIM material identifier string: "{user_input}"
        
        CRITICAL LINGUISTIC CONTEXT:
        - The string may contain Catalan construction terms.

        Task: 
        - Remove numerical identifiers and strings identifiers without semantic meaning
        - Extract the actual core materials present in the input text.
        - Generate exactly 6 architectural variations, synonyms, or technical translations: 2 strictly in Spanish, 2 strictly in Catalan, 2 strictly in English. 
        Do not add any specifications that are not explicit in the IFC/BIM material identifier string.
        - Do NOT invent fake brands, unrelated layers, or fallback materials (like insulation) if they aren't in the  IFC/BIM material identifier string.
        - add descriptions of the element in spanish and english

        Anchors: LLana mineral is mineral wool, formigo is concrete, acer is steel, mao means brick
        
        CRITICAL: Return ONLY a raw valid JSON object. Do not include introductory text, explanations, markdown fences or language identifiers.
        {{
            "name": "Cleaned Base Name String",
            "variations": ["material variation 1", "material variation 2", " material variation 3", "material variation 4", "material variation 5", "material variation 6"],
            "spanish description": "spanish description oo the construction element",
            "english description": "english description of the construction element"
        }}
        """
        
        print("input ", prompt)
        # Using a safer custom execution routine to protect against conversational bloat
        output = self._safe_call_and_parse(prompt, user_input)
        print("output: ", output)
        
        return output

    def _safe_call_and_parse(self, prompt: str, fallback_input: str) -> Dict[str, Any]:
        """Executes the LLM call and rigidly strips non-JSON boilerplate out of local engine responses."""
        try:
            # Fallback to base agent engine execution
            raw_output = self._call_and_parse(prompt)
            
            # If the base class returned a dict already, validate it
            if isinstance(raw_output, dict):
                return self._sanitize_variations_list(raw_output)
                
            if isinstance(raw_output, str):
                # Use Regex to isolate the raw JSON block if the model included chatty text
                json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    return self._sanitize_variations_list(parsed)
                    
        except Exception as e:
            logger.error(f"Failed to parse LLM structured response safely: {e}")
            
        return {"name": fallback_input, "variations": [fallback_input]}

    def _sanitize_variations_list(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures variations data is structurally sound and safely formatted."""
        if "variations" not in output or not isinstance(output["variations"], list):
            output["variations"] = []
        # Filter out accidental empty string updates or systemic parsing artifacts
        output["variations"] = [str(v).strip() for v in output["variations"] if v]
        return output