import os
import sys
import json
from tqdm import tqdm
from dotenv import load_dotenv

# ── 1. ENVIRONMENT & PATH SETTING ──────────────────────────────────────────
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../" if "src" in current_dir else ".")) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.load_conf import load_config
from src.RAG.vectorstore import VectorStoreManager
from src.RAG.searchengine import ConstructionSearchService

# ── 2. ISOLATED DATABASE ROUTING ───────────────────────────────────────────
CANONICAL_DB_PATH =  os.environ.get("IFC_VECTOR_DB_PATH")
IFC_DB_PATH = os.environ.get("IFC_VECTOR_DB_PATH")

# Match your specific configuration architecture labels
TARGET_COLLECTION = "materials_v1"
CFG_PATH = os.environ.get("CONFIG_PATH", os.path.join(project_root, "conf/config.yaml"))

print(f"📂 Loading Configuration from: {CFG_PATH}")
cfg = load_config(CFG_PATH)
model_name = cfg['vector_store']['model_name']

print(f"🔗 Base Vector Engine initialized with model: {model_name}")

# ── 3. EXTRACTOR HELPERS ───────────────────────────────────────────────────
def extract_metadata(hit):
    """Safely extracts a metadata dictionary object from a search match hit."""
    if isinstance(hit, dict):
        return hit.get("metadata") or hit.get("metas") or hit.get("payload") or {}
    return getattr(hit, "metadata", None) or getattr(hit, "payload", None) or getattr(hit, "metas", {})

def extract_id_and_score(hit):
    """Safely extracts structural ID strings and raw compliance match numbers."""
    raw_id = hit.get("id") if isinstance(hit, dict) else getattr(hit, "id", "Unknown")
    if isinstance(hit, dict):
        score = hit.get("score") or hit.get("distance") or hit.get("score_match", 0.0)
    else:
        score = getattr(hit, "score", None) or getattr(hit, "distance", 0.0)
    return str(raw_id), score

# ── 4. CROSS MATCHING ENGINE MATRICES ──────────────────────────────────────
def run_matrix_generation(output_filename="matching_matrix/material_matching_matrix.json"):
    print(f"\n⚡ Step 1: Initializing Target Query Service from: {IFC_DB_PATH}")
    ifc_vsm = VectorStoreManager(db_path=IFC_DB_PATH, collection_name=TARGET_COLLECTION, model_name=model_name)
    search_service = ConstructionSearchService(vector_store=ifc_vsm)

    print(f"⚡ Step 2: Extracting Source Docs directly from Canonical Store: {CANONICAL_DB_PATH}")
    canonical_vsm = VectorStoreManager(db_path=CANONICAL_DB_PATH, collection_name=TARGET_COLLECTION, model_name=model_name)
    
    # Mirroring the precise extraction syntax proven functional in your inspector
    chroma_collection = canonical_vsm.collection
    raw_data = chroma_collection.get(include=["metadatas", "documents"])

    ids = raw_data.get("ids", []) if raw_data else []
    metadatas = raw_data.get("metadatas", []) if raw_data else []
    documents = raw_data.get("documents", []) if raw_data else []

    total_source_items = len(ids)
    print(f"📊 Elements collected to process: {total_source_items}")
    
    if total_source_items == 0:
        print("🚨 Execution aborted: The Source Canonical database returned zero assets.")
        return

    matching_matrix = {}

    # Step 3: Run queries sequentially utilizing individual texts
    print("\n🚀 Commencing cross-matching loop...")
    for i in tqdm(range(total_source_items), desc="Cross-referencing databases"):
        source_id = ids[i]
        source_text = documents[i] if i < len(documents) else ""
        source_meta = metadatas[i] if i < len(metadatas) else {}

        if not source_text:
            continue

        try:
            # Query the target database utilizing a high limit count threshold 
            hits = search_service.search(query=str(source_text).strip(), limit=50000)
            
            ranked_matches = []
            seen_ifc_materials = set()

            for hit in hits:
                raw_target_id, score = extract_id_and_score(hit)
                target_meta = extract_metadata(hit)

                # Parse structural variants split markers (e.g., base_id##var_x)
                ifc_material_id = target_meta.get("material_id")
                if not ifc_material_id and "##" in str(raw_target_id):
                    ifc_material_id = str(raw_target_id).split("##")[0]
                elif not ifc_material_id:
                    ifc_material_id = raw_target_id

                # Apply structural array deduplication
                if ifc_material_id in seen_ifc_materials:
                    continue
                seen_ifc_materials.add(ifc_material_id)

                ranked_matches.append({
                    "ifc_material_id": ifc_material_id,
                    "score": score,
                    "metadata": target_meta
                })

            # Append the mapping structure to the matching matrix dictionary
            matching_matrix[source_id] = {
                "source_text": source_text,
                "source_metadata": source_meta,
                "matches": ranked_matches
            }

        except Exception as e:
            print(f"\n⚠️  Skipped processing item ID '{source_id}' due to exception: {e}")
            continue

    # ── 5. SAVE MATRIX DATA METADATA OUT ──────────────────────────────────
    output_path = os.path.join(project_root, output_filename)
    
    # Extract the target directory path and verify its existence recursively
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"📁 Creating destination tree directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n💾 Writing structured match data output directly to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(matching_matrix, out_file, ensure_ascii=False, indent=2)

    print("✨ Matrix processing strategy finished successfully.")

if __name__ == "__main__":
    run_matrix_generation()