"""
End-to-end test for the add_typologies pipeline.
Each agent step is called individually so intermediate results are captured.

Run from RAGTest root:
    python scripts/add_tests/test_add_typologies.py
"""

import json, os, sys
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.add_typologies.agents import (
    AddTypologyIntentAgent,
    IDResolutionAgent,
    LayerMaterialExistenceAgent,
    StructureValidatorAgent,
    TypologyExtractionAgent,
    TypologyPersistenceAgent,
)
from src.agents.query.agents import CleaningAgent
from src.RAG.searchengine import ConstructionSearchService
from src.RAG.vectorstore import VectorStoreManager
from src.utils.load_conf import load_config


TEST_CASES = [
    {
        "label": "All materials known in DB → save succeeds",
        "input": (
            "Add a facade wall (Fachada) with three layers: "
            "fábrica de ladrillo cerámico 0.115m, lana mineral 0.04m, "
            "fábrica de ladrillo hueco 0.07m."
        ),
        "expect_success": True,
    },
    {
        "label": "Unknown material → must add via add_mats first",
        "input": (
            "Add a wall (Fachada) with one layer: "
            "aerogel insulation board, thickness 0.03m."
        ),
        "expect_success": False,
    },
    {
        "label": "Missing category → structure validation fails",
        "input": "Add something with one layer: lana mineral 0.04m.",
        "expect_success": False,
    },
    {
        "label": "Explicit ID that is free → save succeeds",
        "input": (
            "Add construction S999, category Fachada, layers: "
            "revestimiento intermedio 0.01m, lana mineral 0.04m."
        ),
        "expect_success": True,
    },
    {
        "label": "Duplicate ID → save blocked",
        "input": "Add construction S01, category Fachada, layers: lana mineral 0.04m.",
        "expect_success": False,
    },
]


def main():
    cfg = load_config(os.path.join(project_root, "conf/config.yaml"))

    real_db   = os.path.join(project_root, cfg["ingestion"]["input_file"])
    mats_path = os.path.join(project_root, cfg["ingestion"]["materials_file"])
    tmp_db    = os.path.join(project_root, "data_in/tmp_test_typologies.json")

    with open(real_db, "r", encoding="utf-8") as f:
        db = json.load(f)
    with open(mats_path, "r", encoding="utf-8") as f:
        materials_db = json.load(f)
    with open(tmp_db, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    vs = VectorStoreManager(
        db_path=cfg["vector_store"]["db_path"],
        collection_name=cfg["vector_store"]["collection_name"],
        model_name=cfg["vector_store"]["model_name"],
    )
    search_service = ConstructionSearchService(vector_store=vs)

    intender       = AddTypologyIntentAgent(cfg)
    extractor      = TypologyExtractionAgent(cfg)
    cleaner        = CleaningAgent(cfg)
    id_resolver    = IDResolutionAgent()
    struct_val     = StructureValidatorAgent()
    mat_existence  = LayerMaterialExistenceAgent(materials_db)
    persistence    = TypologyPersistenceAgent(tmp_db)

    results = []
    print("─" * 60)
    print("  ADD TYPOLOGIES — end-to-end test")
    print("─" * 60)

    for tc in TEST_CASES:
        query = tc["input"]
        print(f"\n[TEST] {tc['label']}")
        print(f"  Query: {query[:80]}...")

        # Step 1 — Intent
        raw_intent = intender.determine(query)
        print(f"  [1] Raw intent    : {raw_intent}")
        intent_dict = cleaner.clean(str(raw_intent), target_schema={"intent": "string"})
        intent = intent_dict.get("intent", "add_new_typology")
        print(f"  [1] Cleaned intent: {intent}")

        # Step 2 — Extract
        extracted = extractor.extract(query)
        print(f"  [2] Extracted     : {extracted}")

        # Step 3 — Structure validation
        struct_result = struct_val.validate(extracted)
        print(f"  [3] Structure     : {struct_result}")

        # Step 4 — ID resolution
        resolved_id, id_issues = id_resolver.resolve(extracted.get("id"), list(db.keys()))
        print(f"  [4] ID resolved   : {resolved_id}  issues={id_issues}")

        # Step 5 — Per-layer material existence (exact lookup in materials.json)
        layer_results = []
        for i, layer in enumerate(extracted.get("layers") or []):
            lr = mat_existence.validate(layer, search_service)
            print(f"  [5] Layer {i+1} '{layer.get('material')}': found={lr['material_found_in_db']} match={lr.get('db_match')}")
            layer_results.append({
                "material": layer.get("material"),
                "thickness": layer.get("thickness"),
                "found_in_db": lr["material_found_in_db"],
                "db_match": lr.get("db_match"),
                "valid": lr["valid"],
            })

        # Collect all issues
        issues = struct_result["issues"] + id_issues
        for lr in layer_results:
            if not lr["valid"]:
                issues.append(
                    f"Material '{lr['material']}' not found in the database. "
                    "Add it first using the add_mats pipeline."
                )

        # Step 6 — Persist
        saved_id = None
        if not issues and intent in ("add_new_typology", "update_existing"):
            construction = {"category": extracted["category"]}
            if extracted.get("cte_code"):
                construction["cte_code"] = extracted["cte_code"]
            if extracted.get("u_formula"):
                construction["u_formula"] = extracted["u_formula"]
            construction["layers"] = extracted.get("layers", [])
            persistence.save(resolved_id, construction, db)
            saved_id = resolved_id

        success = saved_id is not None
        status  = "PASS" if success == tc["expect_success"] else "FAIL"
        print(f"  Issues : {issues or '—'}")
        print(f"  Saved  : {saved_id or '(not saved)'}")
        print(f"  [{status}]")

        results.append({
            "query": query,
            "label": tc["label"],
            "intent_raw": str(raw_intent),
            "intent_cleaned": intent,
            "extracted": extracted,
            "structure_validation": struct_result,
            "id_resolved": resolved_id,
            "id_issues": id_issues,
            "layer_validations": layer_results,
            "issues": issues,
            "saved_id": saved_id,
            "test_passed": status == "PASS",
            "timestamp": datetime.now().isoformat(),
        })

    out = os.path.join(project_root, "test_add_typologies_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    passed = sum(r["test_passed"] for r in results)
    print(f"\n{'─' * 60}")
    print(f"  {passed}/{len(results)} tests passed — results → {out}")
    print("─" * 60)
    os.remove(tmp_db)


if __name__ == "__main__":
    main()
