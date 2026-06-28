from typing import Any, Dict, List
import json
import os
from src.agents.core.core import BaseLLMAgent, logger


class AddMaterialIntentAgent(BaseLLMAgent):
    """Classifies whether the user wants to add or update a material."""

    def determine(self, user_input: str) -> str:
        prompt = f"""
        Classify the intent for: "{user_input}"
        Options:
        "add_new_material"          - user wants to add a new material to the database,
        "update_existing_material"  - user wants to update an existing material's properties,
        "other"                     - not a material add/update operation.
        Return ONLY JSON: {{"intent": "category"}}
        """
        return self._call_llm(prompt, is_json=True)

class MaterialExtractionAgent(BaseLLMAgent):
    """
    Extracts a material definition from user input.
    """
    # ... (keep schema as it is)

    def extract(self, user_input: str) -> Dict[str, Any]:
        prompt = f"""
        Extract material thermal properties from: "{user_input}"

        Return ONLY JSON:
        {{
            "name": string or null,
            "density": float (kg/m³) or null,
            "thermal_conductivity": float (W/m·K) or null,
            "specific_heat": float (J/kg·K) or null
        }}
        
        CRITICAL RULES FOR "name":
        1. Copy the exact material name from the text (preferably in Spanish if present).
        2. Strip away trailing generic words like "material", "producto", "capa", "aislante", "board", "insulation". 
           (e.g., "veritasium material" -> "veritasium", "aerogel board" -> "aerogel").
        3. If no specific material name appears in the text, set "name" to null.
        4. Do NOT use generic fallback placeholders like "Unknown".
        """
        return self._call_and_parse(prompt)


class MaterialPropertiesValidatorAgent(BaseLLMAgent):
    """Checks that all required thermal properties are present."""

    REQUIRED = ["name", "density", "thermal_conductivity", "specific_heat"]

    def validate(self, material: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[str] = []
        for field in self.REQUIRED:
            if material.get(field) is None:
                issues.append(f"Missing required property: '{field}'.")
        return {"valid": len(issues) == 0, "issues": issues}


class MaterialNameValidatorAgent(BaseLLMAgent):
    """
    Catches generic placeholder names hallucinated by the LLM when no material
    name was present in the user's text (e.g. "Material", "Unknown", "Unnamed").
    Rule-based — no LLM call needed.
    """

    PLACEHOLDERS = {
        "material", "unknown", "unnamed", "n/a", "none", "nombre",
        "nombre material", "material desconocido", "material agregado",
        "sin nombre", "unspecified",
    }

    def validate(self, name: Any) -> Dict[str, Any]:
        if not name:
            return {"valid": False, "issue": "Material name is missing."}
        if str(name).strip().lower() in self.PLACEHOLDERS:
            return {"valid": False, "issue": f"'{name}' is a generic placeholder, not a material name."}
        if len(str(name).strip()) < 3:
            return {"valid": False, "issue": f"'{name}' is too short to be a valid material name."}
        return {"valid": True, "issue": None}


class MaterialDuplicateCheckAgent(BaseLLMAgent):
    """
    Checks whether a material already exists and surfaces semantically similar ones.

    Two independent checks:
    1. Exact key lookup in materials.json (case-insensitive) → blocks add_new if matched.
    2. Vector similarity search → returns top-K similar materials as suggestions.
       Currently informational only; intended for future "did you mean?" UX.
    """

    SIMILARITY_TOP_K = 3

    def __init__(self, materials_db_path: str):
        
        with open(materials_db_path, encoding="utf-8") as f:
             self.materials_db = json.load(f)
       

    def check(self, material_name: str, search_service: Any) -> Dict[str, Any]:
        if not material_name:
            return {"exists": False, "db_match": None, "similar_materials": []}

        # 1. Exact lookup
        exact_match = None
        if material_name in self.materials_db:
            exact_match = material_name
        else:
            lower = material_name.lower()
            for key in self.materials_db:
                if key.lower() == lower:
                    exact_match = key
                    break

        # 2. Vector search for similar materials (informational)
        hits = search_service.search(material_name, limit=self.SIMILARITY_TOP_K)
        similar = [
            {"name": h["metadata"].get("material"), "score": h["score"]}
            for h in hits
            if h["metadata"].get("material") != exact_match
        ]

        return {
            "exists": exact_match is not None,
            "db_match": exact_match,
            "similar_materials": similar,  # future: present to user before saving
        }


class MaterialPersistenceAgent(BaseLLMAgent):
    """
    Upserts a validated material into:
      1. materials.json (exact key store)
      2. added_materials/{name}.json (individual record log)
      3. The vector DB (for semantic search in query and add_typologies pipelines)
    """

    def __init__(self, materials_db_path: str):
        self.materials_db_path = materials_db_path
        self._added_dir = os.path.join(os.path.dirname(materials_db_path), "added_materials")

    def save(
        self,
        material: Dict[str, Any],
        vs_manager: Any,
        materials_db: Dict[str, Any],
    ) -> None:
        name = material["name"]
        density = material.get("density", 0.0)
        tc = material.get("thermal_conductivity", 0.0)
        sh = material.get("specific_heat", 0.0)

        record = {"density": density, "thermal_conductivity": tc, "specific_heat": sh}

        # 1. Write to materials.json
        materials_db[name] = record
        with open(self.materials_db_path, "w", encoding="utf-8") as f:
            json.dump(materials_db, f, indent=2, ensure_ascii=False)

        # 2. Write individual record to added_materials/
        os.makedirs(self._added_dir, exist_ok=True)
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()[:80]
        added_path = os.path.join(self._added_dir, f"{safe_name}.json")
        with open(added_path, "w", encoding="utf-8") as f:
            json.dump({name: record}, f, indent=2, ensure_ascii=False)

        # 3. Upsert to vector DB
        doc = f"{name}. Density: {density}, TC: {tc}"
        vs_manager.upsert_materials(
            [name],
            [doc],
            [{"material": name, "density": density, "tc": tc,
              "source": "user_added", "material_translations": ""}],
        )
        logger.info(f"Material '{name}' saved to materials.json, {self._added_dir}, and vector DB.")
