import logging
from typing import Any, Dict, List, Optional, Tuple

from src.agents.query.agents import CleaningAgent

logger = logging.getLogger("AddTypologyOrchestrator")


class AddTypologyOrchestrator:
    """
    Orchestrates the add-typology pipeline:

        user input
            → AddTypologyIntentAgent      (intent detection)
            → TypologyExtractionAgent     (structural extraction: id, category, layers)
            → StructureValidatorAgent     (required fields)
            → IDResolutionAgent           (ID uniqueness)
            → LayerMaterialExistenceAgent × N  (material present in vector DB?)
            → TypologyPersistenceAgent    (write to JSON, only if no issues)

    If a material is missing from the DB the pipeline surfaces it as an issue
    and instructs the caller to add it first via the add_mats pipeline.

    Returns:
        intent      – detected intent string
        extracted   – raw extracted typology dict
        issues      – list of validation problems (empty = success)
        saved_id    – key under which the typology was saved, or None
    """

    def __init__(
        self,
        agents: Dict[str, Any],
        search_service: Any,
        persistence_agent: Any,
        db: Dict[str, Any],
    ):
        self.intender = agents["intent"]
        self.extractor = agents["extractor"]
        self.id_resolver = agents["id_resolver"]
        self.structure_validator = agents["structure_validator"]
        self.material_existence = agents["material_existence"]
        self.cleaner = agents["cleaner"]
        self.search_service = search_service
        self.persistence = persistence_agent
        self.db = db

    def run(self, user_input: str) -> Tuple[str, Dict, List[str], Optional[str]]:
        # 1. Intent
        raw_intent = self.intender.determine(user_input)
        intent_data = self.cleaner.clean(
            str(raw_intent), target_schema={"intent": "string"}
        )
        intent = intent_data.get("intent", "add_new_typology")

        if intent not in ("add_new_typology", "update_existing"):
            return intent, {}, ["Intent not recognised as a typology add/update operation."], None

        # 2. Extract
        extracted = self.extractor.extract(user_input)

        # 3. Structure validation
        struct_result = self.structure_validator.validate(extracted)
        issues: List[str] = struct_result["issues"][:]

        # 4. ID validation
        resolved_id, id_issues = self.id_resolver.resolve(
            extracted.get("id"), list(self.db.keys())
        )
        issues.extend(id_issues)

        # 5. Material existence check per layer (enriched with typology category context)
        typology_context = extracted.get("category", "")
        for i, layer in enumerate(extracted.get("layers") or []):
            result = self.material_existence.validate(
                layer, self.search_service, typology_context=typology_context
            )
            if not result["valid"]:
                mat = layer.get("material") or f"layer {i + 1}"
                issues.append(
                    f"Material '{mat}' not found in the database. "
                    "Add it first using the add_mats pipeline."
                )

        # 6. Persist only when fully valid
        saved_id: Optional[str] = None
        if not issues:
            construction = {"category": extracted["category"]}
            if extracted.get("cte_code"):
                construction["cte_code"] = extracted["cte_code"]
            if extracted.get("u_formula"):
                construction["u_formula"] = extracted["u_formula"]
            construction["layers"] = extracted.get("layers", [])
            self.persistence.save(resolved_id, construction, self.db, self.search_service)
            saved_id = resolved_id

        return intent, extracted, issues, saved_id
