import json
from typing import Dict, Any
from src.agents.core.core import BaseLLMAgent

class ElementClassInputAgent(BaseLLMAgent):
    schema = {"ifc_class": "string"}
    
    def determine(self, user_input: str) -> Dict[str, Any]:
        prompt = f"""Parse this query for structural components or IFC functional classes: "{user_input}"
        Map synonyms to native IFC structural schema elements:
        - Wall / Facade / Tabique / Mur -> "IfcWall" or "IfcWallStandardCase"
        - Slab / Floor / Forjat / Solera -> "IfcSlab"
        - Roof / Cubierta / Tejado -> "IfcRoof"
        - Door / Puerta -> "IfcDoor"
        - Window / Ventana -> "IfcWindow"

        Return ONLY valid JSON:
        {{
            "ifc_class": "The resolved Ifc Class name string (e.g. IfcSlab, IfcWall) or null"
        }}"""
        return self._call_and_parse(prompt)


class TypoLayerCountInputAgent(BaseLLMAgent):
    # Expanded schema signature to accept polymorphic structures
    schema = {"layer_count": "integer", "operator": "string", "min_layers": "integer", "max_layers": "integer"}

    def determine(self, user_input: str) -> Dict[str, Any]:
        prompt = f"""Extract layer boundary criteria from this user phrase: "{user_input}"
        
        CRITICAL INSTRUCTIONS:
        1. If the user defines an explicit RANGE (e.g., "between 4 and 8", "from 3 to 5"):
           - Populate "min_layers" and "max_layers" as integers.
           - Set "layer_count" and "operator" to null.
        2. If the user defines a SINGLE boundary condition (e.g., "less than 5", "at least 2", "exactly 3"):
           - Populate "layer_count" and "operator" ("==", ">", "<", ">=", "<=").
           - Set "min_layers" and "max_layers" to null.

        Return ONLY valid JSON:
        {{
            "layer_count": int or null,
            "operator": "==" | ">" | "<" | ">=" | "<=" or null,
            "min_layers": int or null,
            "max_layers": int or null
        }}"""
        return self._call_and_parse(prompt)


class MaterialMixInputAgent(BaseLLMAgent):
    schema = {"materials": "list of strings", "matching_mode": "string"}

    def determine(self, user_input: str) -> Dict[str, Any]:
        prompt = f"""Extract required structural layer materials from: "{user_input}"
        Deduce if the user needs ALL listed materials together (AND), or any single hit (OR).
        Return ONLY valid JSON:
        {{
            "materials": ["concrete", "insulation", ...],
            "matching_mode": "AND" | "OR"
        }}"""
        return self._call_and_parse(prompt)


class TypoThicknessInputAgent(BaseLLMAgent):
    # Expanded schema signature to handle metric ranges
    schema = {"thickness_m": "float", "operator": "string", "tolerance": "float", "min_thickness": "float", "max_thickness": "float"}

    def determine(self, user_input: str) -> Dict[str, Any]:
        prompt = f"""Extract structural thickness ranges or values from this user phrase: "{user_input}"
        Always convert millimeters (mm) or centimeters (cm) to absolute meters (m). (e.g., 20cm -> 0.20, 150mm -> 0.15)
        
        CRITICAL INSTRUCTIONS:
        1. If the user defines an explicit thickness RANGE (e.g., "between 10cm and 30cm", "from 150mm to 200mm"):
           - Populate "min_thickness" and "max_thickness" values as floats.
           - Set "thickness_m" and "operator" to null.
        2. If the user defines a SINGLE thickness boundary (e.g., "thicker than 20cm", "under 0.3m"):
           - Populate "thickness_m" and "operator" ("==", ">", "<", ">=", "<=").
           - Set "min_thickness" and "max_thickness" to null.

        Return ONLY valid JSON:
        {{
            "thickness_m": float or null,
            "operator": "==" | ">" | "<" | ">=" | "<=" or null,
            "tolerance": float (default is 0.005),
            "min_thickness": float or null,
            "max_thickness": float or null
        }}"""
        return self._call_and_parse(prompt)


class TypoIDInputAgent(BaseLLMAgent):
    schema = {"typology_ids": "list of strings"}

    def determine(self, user_input: str) -> Dict[str, Any]:
        prompt = f"""Extract explicit typology tracking identifiers from: "{user_input}"
        Return ONLY valid JSON:
        {{
            "typology_ids": ["T01", "Wall_Type_A", ...]
        }}"""
        return self._call_and_parse(prompt)