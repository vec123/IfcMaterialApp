from typing import Any, Dict, List, Optional, Tuple
from src.agents.core.core import BaseLLMAgent, logger
import json
import os


class AddTypologyIntentAgent(BaseLLMAgent):
    """Classifies whether the user wants to add or update a construction typology."""

    def determine(self, user_input: str) -> str:
        prompt = f"""
        Classify the intent for: "{user_input}"
        Options:
        "add_new_typology" - user wants to add a new construction typology,
        "update_existing"  - user wants to modify an existing typology,
        "other"            - not a typology add/update operation.
        Return ONLY JSON: {{"intent": "category"}}
        """
        return self._call_llm(prompt, is_json=True)


class TypologyExtractionAgent(BaseLLMAgent):
    """
    Parses user input into a construction typology.
    Extracts only structural data: id, category, and layers (material + thickness).
    Thermal properties are NOT handled here — that is add_mats responsibility.
    """

    schema = {
        "id": "string or null",
        "category": "string",
        "cte_code": "string or null",
        "u_formula": "string or null",
        "layers": "list",
    }

    def extract(self, user_input: str) -> Dict[str, Any]:
        prompt = f"""
        Extract a building construction typology from: "{user_input}"

        Return ONLY JSON:
        {{
            "id": string or null,
            "category": string (e.g. "Fachada", "Division", "Forjado"),
            "cte_code": string or null,
            "u_formula": string or null,
            "layers": [
                {{
                    "material": string,
                    "thickness": float (meters) or "eAT" or null
                }}
            ]
        }}
        Thickness: 10cm → 0.10, 115mm → 0.115.
        Use "eAT" for variable air gap or insulation with formula-based thickness.
        Set unmentioned fields to null.
        """
        return self._call_and_parse(prompt)


class IDResolutionAgent:
    """
    Determines the construction ID:
    - Uses the user-suggested ID if free; flags it if already taken.
    - Auto-generates the next sequential ID when none is provided.
    """

    def resolve(
        self, extracted_id: Optional[str], existing_ids: List[str]
    ) -> Tuple[Optional[str], List[str]]:
        if extracted_id:
            if extracted_id in existing_ids:
                return None, [f"ID '{extracted_id}' already exists in the database."]
            return extracted_id, []
        return self._generate(existing_ids), []

    def _generate(self, existing_ids: List[str]) -> str:
        prefixes: Dict[str, List[int]] = {}
        for eid in existing_ids:
            prefix = "".join(c for c in eid if c.isalpha())
            num_str = "".join(c for c in eid if c.isdigit())
            if prefix and num_str:
                prefixes.setdefault(prefix, []).append(int(num_str))
        prefix = "S" if "S" in prefixes else next(iter(prefixes), "S")
        next_num = max(prefixes.get(prefix, [0])) + 1
        return f"{prefix}{next_num:03d}"


class StructureValidatorAgent:
    """Checks required fields and that every layer has a material and thickness."""

    def validate(self, construction: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[str] = []
        if not construction.get("category"):
            issues.append("Missing required field: 'category'.")
        layers = construction.get("layers") or []
        if not layers:
            issues.append("Construction must have at least one layer.")
        else:
            for i, layer in enumerate(layers):
                if not layer.get("material"):
                    issues.append(f"Layer {i + 1}: missing 'material'.")
                if layer.get("thickness") is None:
                    issues.append(f"Layer {i + 1}: missing 'thickness'.")
        return {"valid": len(issues) == 0, "issues": issues}


class LayerMaterialExistenceAgent:
    """
    Checks whether a layer's material exists in materials.json.
    Does NOT validate thermal properties — that belongs to add_mats.
    If not found, the user must add the material first via add_mats.

    Lookup strategy:
    1. Exact key match in materials.json (case-insensitive).
    2. Vector search fallback for language/accent variations
       (threshold 0.60 corrects for ChromaDB L2 on unit-norm embeddings).
    """

    VECTOR_THRESHOLD = 0.60

    def __init__(self, materials_db: Dict[str, Any]):
        self.materials_db = materials_db

    def validate(
        self,
        layer: Dict[str, Any],
        search_service: Any,
        typology_context: str = "",
    ) -> Dict[str, Any]:
        """
        Check whether a layer's material exists in the DB.

        typology_context — category/name of the typology being added (e.g. "Fachada").
        When provided, it is appended to the vector query to improve disambiguation
        among materials with similar names that are used in different contexts.
        """
        material = layer.get("material", "")

        if not material:
            return {"layer": layer, "material_found_in_db": False, "db_match": None, "valid": False}

        # 1. Exact key lookup in materials.json
        if material in self.materials_db:
            return {"layer": layer, "material_found_in_db": True, "db_match": material, "valid": True}

        lower = material.lower()
        for key in self.materials_db:
            if key.lower() == lower:
                return {"layer": layer, "material_found_in_db": True, "db_match": key, "valid": True}

        # 2. Vector search — enrich query with typology context when available
        query = f"{material} {typology_context}".strip() if typology_context else material
        hits = search_service.search(query, limit=1)
        found = bool(hits and hits[0]["score"] >= self.VECTOR_THRESHOLD)
        return {
            "layer": layer,
            "material_found_in_db": found,
            "db_match": hits[0]["metadata"].get("material") if found else None,
            "valid": found,
            "query_used": query,
        }


class TypologyPersistenceAgent:
    """
    Writes a validated construction typology to:
      1. The main constructions.json database.
      2. added_typologies/{new_id}.json (individual record log).
      3. The typology vector index (when search_service is provided).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._added_dir = os.path.join(os.path.dirname(db_path), "added_typologies")

    def save(
        self,
        new_id: str,
        construction: Dict[str, Any],
        db: Dict[str, Any],
        search_service: Any = None,
    ) -> None:
        db[new_id] = construction
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, ensure_ascii=False)

        os.makedirs(self._added_dir, exist_ok=True)
        added_path = os.path.join(self._added_dir, f"{new_id}.json")
        with open(added_path, "w", encoding="utf-8") as f:
            json.dump({new_id: construction}, f, indent=2, ensure_ascii=False)

        if search_service is not None:
            search_service.upsert_typology(new_id, construction)

        logger.info(f"Typology '{new_id}' written to {self.db_path} and {self._added_dir}.")
