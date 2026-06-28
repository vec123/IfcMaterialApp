# src/agents/typology/intent_agents.py

import json
from typing import List, Dict, Any, Union
from src.agents.core.core import BaseLLMAgent, logger
from src.agents.ifc.query_typology.specialists import (
TypoIDInputAgent,
ElementClassInputAgent,
TypoLayerCountInputAgent,
MaterialMixInputAgent,
TypoThicknessInputAgent

)
class TypologyIntentAgent(BaseLLMAgent):
    """Classifies complex architectural and relational constraints within an IFC Typology space query."""

    def determine(self, user_input: str) -> List[str]:
        prompt = f"""
        Analyze the structural architectural query and identify ALL intents present.
        If the user specifies multiple constraints, return ALL matching intent keys separated by commas.

        Options:
        "find_by_typology_id"       - user explicitly asks for a unique typology set code (e.g., T01, WallType_04).
        "filter_by_element_class"    - user references an IFC functional element class (e.g., Slab, Forjat, Cubierta, IfcWall, Columns).
        "filter_by_layer_count"      - user constraints the structural layer configuration quantity (e.g., more than 3 layers, exactly 1 layer).
        "filter_by_material_mix"     - user specifies one or multiple required materials (e.g., Concrete AND Gypsum, Ladrillo).
        "filter_by_total_thickness"  - user specifies total compound structural depth requirements (e.g., thinner than 150mm, > 0.2m).

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
            return ["filter_by_material_mix"]  # Safe fallback matching legacy components


class TypologyFunctionInputAgent(BaseLLMAgent):
    """Router agent that matches intents to discrete parameter extraction specialists."""
    
    def __init__(self, config):
        super().__init__(config)
        self.specialists = {
            "find_by_typology_id":      TypoIDInputAgent(config),
            "filter_by_element_class":   ElementClassInputAgent(config),
            "filter_by_layer_count":     TypoLayerCountInputAgent(config),
            "filter_by_material_mix":    MaterialMixInputAgent(config),
            "filter_by_total_thickness": TypoThicknessInputAgent(config),
        }

    def determine(self, intent: str, user_input: str) -> Dict[str, Any]:
        agent = self.specialists.get(intent)
        if not agent:
            logger.warning(f"No specialist mapped for intent: {intent}")
            return {}
        return agent.determine(user_input)