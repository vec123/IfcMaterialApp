from typing import Any, Dict, List
import json
import os
from src.agents.core.core import BaseLLMAgent, logger



class MaterialExtractionAgent(BaseLLMAgent):
    """
    Extracts a material definition from user input.
    """
    # ... (keep schema as it is)

    def extract(self, user_input: str) -> Dict[str, Any]:
            prompt = f"""
            Your role is identifying materials in a string.
            Extract and clean the material name from the following identifier: "{user_input}"

            Return ONLY a JSON object with this exact key:
            {{
                "name": string or null
            }}
            
            CRITICAL RULES FOR "name":
            1. Extract the core material name from the text (preserve it in Spanish if that is how it appears).
            2. Clean the string by stripping away alphanumeric codes, prefix/suffix tags, project identifiers, 
             and trailing generic numbers (e.g., "B051_01_Morter" should become "Morter").
            3. If no recognizable material name can be deduced from the text, set "name" to null.
            4. Return ONLy the name of the identified material
            """
            print("input ", user_input)
            output = self._call_and_parse(prompt)
            print("output: ", output)
            
            # If the LLM failed or returned None, safely fallback to an empty schema
            if output is None:
                return {"name": None}
            
            return output