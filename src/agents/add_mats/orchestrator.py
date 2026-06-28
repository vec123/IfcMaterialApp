import logging
from typing import Any, Dict, List, Tuple


logger = logging.getLogger("AddMaterialOrchestrator")


class AddMaterialOrchestrator:
    """
    Orchestrates the add-material pipeline:

        user input
            → AddMaterialIntentAgent          (intent detection)
            → MaterialExtractionAgent         (name + thermal properties)
            → MaterialPropertiesValidatorAgent (all four fields required)
            → MaterialDuplicateCheckAgent      (warn if already in DB)
            → MaterialPersistenceAgent         (upsert to vector DB)

    Returns:
        intent    – detected intent string
        extracted – raw extracted material dict
        issues    – list of validation problems (empty = success)
        saved     – True if the material was persisted
    """

    def __init__(
        self,
        agents: Dict[str, Any],
        search_service: Any,
        vs_manager: Any,
        materials_db: Dict[str, Any],
    ):
        self.intender = agents["intent"]
        self.extractor = agents["extractor"]
        self.name_validator = agents["name_validator"]
        self.properties_validator = agents["properties_validator"]
        self.duplicate_check = agents["duplicate_check"]
        self.persistence = agents["persistence"]
        self.cleaner = agents["cleaner"]
        self.search_service = search_service
        self.vs_manager = vs_manager
        self.materials_db = materials_db

    def run(self, user_input: str) -> Tuple[str, Dict, List[str], bool]:
        # 1. Intent
        raw_intent = self.intender.determine(user_input)
        intent_data = self.cleaner.clean(
            str(raw_intent), target_schema={"intent": "string"}
        )
        intent = intent_data.get("intent", "add_new_material")

        if intent not in ("add_new_material", "update_existing_material"):
            return intent, {}, ["Intent not recognised as a material add/update operation."], False

        # 2. Extract
        extracted = self.extractor.extract(user_input)

        # 3a. Validate name is not a placeholder
        name_result = self.name_validator.validate(extracted.get("name"))
        issues: List[str] = []
        if not name_result["valid"]:
            issues.append(name_result["issue"])

        # 3b. Validate all required thermal properties present
        prop_result = self.properties_validator.validate(extracted)
        issues.extend(prop_result["issues"])

        # 4. Duplicate check — exact match blocks add_new; similar materials logged only
        dup_result = self.duplicate_check.check(extracted.get("name", ""), self.search_service)
        if dup_result["exists"] and intent == "add_new_material":
            issues.append(
                f"Material '{extracted.get('name')}' already exists "
                f"(matched: '{dup_result['db_match']}'). "
                "Use update intent to overwrite."
            )

        # 5. Persist if no issues
        saved = False
        if not issues:
            self.persistence.save(extracted, self.vs_manager, self.materials_db)
            saved = True

        return intent, extracted, issues, saved
