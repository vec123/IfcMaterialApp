"""
End-to-end test for the add_mats pipeline.
Each agent step is called individually so intermediate results are captured.

Run from RAGTest root:
    python scripts/add_tests/test_add_mats.py
"""

import json, os, sys
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.add_mats.agents import (
    AddMaterialIntentAgent,
    MaterialDuplicateCheckAgent,
    MaterialExtractionAgent,
    MaterialNameValidatorAgent,
    MaterialPersistenceAgent,
    MaterialPropertiesValidatorAgent,
)
from src.agents.query.agents import CleaningAgent
from src.RAG.searchengine import ConstructionSearchService
from src.RAG.vectorstore import VectorStoreManager
from src.utils.load_conf import load_config


TEST_CASES = [
    {
        "label": "New material — all properties provided → save succeeds",
        "input": (
            "Add material aerogel insulation board: "
            "density 120 kg/m3, thermal conductivity 0.015 W/mK, specific heat 1000 J/kgK."
        ),
        "expect_success": True,
    },
    {
        "label": "Missing thermal_conductivity → validation fails",
        "input": "Add material perlite plaster: density 900, specific_heat 840.",
        "expect_success": False,
    },
    {
        "label": "Missing name → validation fails",
        "input": "Add a material with density 500, thermal_conductivity 0.12, specific_heat 900.",
        "expect_success": False,
    },
    {
        "label": "Existing material with add_new intent → duplicate blocked",
        "input": (
            "Add new material lana mineral: "
            "density 40, thermal_conductivity 0.037, specific_heat 840."
        ),
        "expect_success": False,
    },
    {
        "label": "Existing material with update intent → save succeeds",
        "input": (
            "Update material lana mineral: "
            "density 45, thermal_conductivity 0.036, specific_heat 840."
        ),
        "expect_success": True,
    },
]


def main():
    cfg = load_config(os.path.join(project_root, "conf/config.yaml"))

    mats_path = os.path.join(project_root, cfg["ingestion"]["materials_file"])
    with open(mats_path, "r", encoding="utf-8") as f:
        materials_db = json.load(f)

    vs = VectorStoreManager(
        db_path=cfg["vector_store"]["db_path"],
        collection_name=cfg["vector_store"]["collection_name"],
        model_name=cfg["vector_store"]["model_name"],
    )
    search_service = ConstructionSearchService(vector_store=vs)

    intender       = AddMaterialIntentAgent(cfg)
    extractor      = MaterialExtractionAgent(cfg)
    cleaner        = CleaningAgent(cfg)
    name_validator = MaterialNameValidatorAgent()
    prop_validator = MaterialPropertiesValidatorAgent()
    dup_check      = MaterialDuplicateCheckAgent(materials_db)
    persistence    = MaterialPersistenceAgent(mats_path)

    results = []
    print("─" * 60)
    print("  ADD MATERIALS — end-to-end test")
    print("─" * 60)

    for tc in TEST_CASES:
        query = tc["input"]
        print(f"\n[TEST] {tc['label']}")
        print(f"  Query: {query[:80]}...")

        # Step 1 — Intent
        raw_intent = intender.determine(query)
        print(f"  [1] Raw intent    : {raw_intent}")
        intent_dict = cleaner.clean(str(raw_intent), target_schema={"intent": "string"})
        intent = intent_dict.get("intent", "add_new_material")
        print(f"  [1] Cleaned intent: {intent}")

        # Step 2 — Extract
        extracted = extractor.extract(query)
        print(f"  [2] Extracted     : {extracted}")

        # Step 3a — Name validation
        name_result = name_validator.validate(extracted.get("name"))
        print(f"  [3a] Name         : {name_result}")

        # Step 3b — Properties validation
        prop_result = prop_validator.validate(extracted)
        print(f"  [3b] Properties   : {prop_result}")

        # Step 4 — Duplicate check: exact key + vector similarity suggestions
        dup_result = dup_check.check(extracted.get("name", ""), search_service)
        print(f"  [4] Exact match   : exists={dup_result['exists']}  db_match={dup_result['db_match']}")
        if dup_result["similar_materials"]:
            print(f"  [4] Similar (informational, future 'did you mean?'):")
            for s in dup_result["similar_materials"]:
                print(f"        • {s['name']}  (score={s['score']})")

        # Collect issues
        issues = ([] if name_result["valid"] else [name_result["issue"]]) + prop_result["issues"][:]
        if dup_result["exists"] and intent == "add_new_material":
            issues.append(
                f"Material '{extracted.get('name')}' already exists "
                f"(matched: '{dup_result['db_match']}'). "
                "Use update intent to overwrite."
            )

        # Step 5 — Persist
        saved = False
        if not issues and intent in ("add_new_material", "update_existing_material"):
            persistence.save(extracted, vs, materials_db)
            saved = True

        status = "PASS" if saved == tc["expect_success"] else "FAIL"
        print(f"  Issues : {issues or '—'}")
        print(f"  Saved  : {saved}")
        print(f"  [{status}]")

        results.append({
            "query": query,
            "label": tc["label"],
            "intent_raw": str(raw_intent),
            "intent_cleaned": intent,
            "extracted": extracted,
            "name_validation": name_result,
            "properties_validation": prop_result,
            "duplicate_check": dup_result,
            "issues": issues,
            "saved": saved,
            "test_passed": status == "PASS",
            "timestamp": datetime.now().isoformat(),
        })

    out = os.path.join(project_root, "test_add_mats_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    passed = sum(r["test_passed"] for r in results)
    print(f"\n{'─' * 60}")
    print(f"  {passed}/{len(results)} tests passed — results → {out}")
    print("─" * 60)


if __name__ == "__main__":
    main()
