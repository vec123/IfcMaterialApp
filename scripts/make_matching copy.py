import os
import sys
import json
from typing import Dict, Any, List, Optional
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
CANONICAL_DB_PATH = os.environ.get("VECTOR_DB_PATH")
IFC_DB_PATH = os.environ.get("IFC_VECTOR_DB_PATH")
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER")
if MODEL_PROVIDER == "google":
    API_KEY = os.environ.get("GOOGLE_API_KEY")
elif MODEL_PROVIDER == "groq":
    API_KEY = os.environ.get("GROQ_API_KEY")
else:
    API_KEY = None
TARGET_COLLECTION = "materials_v1"

CFG_PATH = os.environ.get("CONFIG_PATH", os.path.join(project_root, "conf/config.yaml"))

print(f"📂 Loading Configuration from: {CFG_PATH}")
cfg = load_config(CFG_PATH)
model_name = cfg['vector_store']['model_name']

print(f"🔗 Base Vector Engine initialized with model: {model_name}")

# ── 3. EXTRACTOR HELPERS ───────────────────────────────────────────────────
def extract_metadata(hit: Any) -> Dict[str, Any]:
    """Safely extracts a metadata dictionary object from a search match hit."""
    if isinstance(hit, dict):
        return hit.get("metadata") or hit.get("metas") or hit.get("payload") or {}
    return getattr(hit, "metadata", None) or getattr(hit, "payload", None) or getattr(hit, "metas", {})

def extract_id_and_score(hit: Any) -> tuple[str, float]:
    """Safely extracts structural ID strings and true cosine similarity match numbers."""
    raw_id = hit.get("id") if isinstance(hit, dict) else getattr(hit, "id", "Unknown")
    
    if isinstance(hit, dict):
        score = hit.get("score") if hit.get("score") is not None else hit.get("distance", 0.0)
    else:
        score = getattr(hit, "score", None) if getattr(hit, "score", None) is not None else getattr(hit, "distance", 0.0)
        
    return str(raw_id), float(score)

# ── 4. OPTIMIZED CROSS MATCHING ENGINE (NO DUPLICATE MATERIAL IDS) ────────
def run_matrix_generation(output_filename="matching_matrix/material_matching_matrix.json"):
    print(f"\n⚡ Step 1: Initializing Target Query Service from Canonical Library: {CANONICAL_DB_PATH}")
    canonical_vsm = VectorStoreManager(db_path=CANONICAL_DB_PATH, collection_name=TARGET_COLLECTION, model_name=model_name)
    search_service = ConstructionSearchService(vector_store=canonical_vsm)

    print(f"⚡ Step 2: Extracting Source Docs directly from IFC Project Store: {IFC_DB_PATH}")
    ifc_vsm = VectorStoreManager(db_path=IFC_DB_PATH, collection_name=TARGET_COLLECTION, model_name=model_name)
    
    chroma_collection = ifc_vsm.collection
    raw_data = chroma_collection.get(include=["metadatas", "documents"])

    ifc_raw_ids = raw_data.get("ids", []) if raw_data else []
    ifc_metadatas = raw_data.get("metadatas", []) if raw_data else []
    ifc_documents = raw_data.get("documents", []) if raw_data else []

    total_chunks = len(ifc_raw_ids)
    print(f"📊 Total raw vector vectors collected from IFC DB: {total_chunks}")
    
    if total_chunks == 0:
        print("🚨 Execution aborted: The Source IFC database returned zero assets.")
        return

    # Master aggregation: maps source_base_id -> dictionary of properties
    aggregated_ifc_matrix = {}

    print("\n🚀 Commencing deduplicated IFC -> Canonical vector queries...")
    for i in tqdm(range(total_chunks), desc="Searching Canonical for IFC vectors"):
        raw_id = ifc_raw_ids[i]
        source_text = ifc_documents[i] if i < len(ifc_documents) else ""
        source_meta = ifc_metadatas[i] if i < len(ifc_metadatas) else {}

        if not source_text or not str(source_text).strip():
            continue

        # Deduplicate Source: Extract base source material ID
        ifc_material_id = source_meta.get("material_id")
        if not ifc_material_id and "##" in str(raw_id):
            ifc_material_id = str(raw_id).split("##")[0]
        elif not ifc_material_id:
            ifc_material_id = raw_id

        try:
            # Query the Canonical DB
            hits = search_service.search(query=str(source_text).strip(), limit=50)
            
            if ifc_material_id not in aggregated_ifc_matrix:
                aggregated_ifc_matrix[ifc_material_id] = {
                    "source_text": source_text,
                    "source_metadata": source_meta,
                    "canonical_matches_map": {}  # Maps target_base_id -> match data object
                }

            current_matches_map = aggregated_ifc_matrix[ifc_material_id]["canonical_matches_map"]

            # Process Target Hits with deduplication on target base IDs
            for hit in hits:
                raw_canonical_id, score = extract_id_and_score(hit)
                target_meta = extract_metadata(hit)

                # Deduplicate Target: Extract base target material ID (stripping ##var, ##desc, etc.)
                canonical_base_id = target_meta.get("material_id")
                if not canonical_base_id and "##" in str(raw_canonical_id):
                    canonical_base_id = str(raw_canonical_id).split("##")[0]
                elif not canonical_base_id:
                    canonical_base_id = raw_canonical_id

                # If this base target ID was already seen, only update if the new chunk has a higher score
                if canonical_base_id in current_matches_map:
                    if score > current_matches_map[canonical_base_id]["score"]:
                        current_matches_map[canonical_base_id]["score"] = score
                        # Optional: track which exact chunk variant won
                        current_matches_map[canonical_base_id]["winning_chunk_id"] = raw_canonical_id 
                else:
                    current_matches_map[canonical_base_id] = {
                        "canonical_material_id": canonical_base_id,
                        "winning_chunk_id": raw_canonical_id,
                        "score": score,
                        "metadata": target_meta
                    }

        except Exception as e:
            print(f"\n⚠️ Skipped processing IFC Item Vector '{raw_id}' due to exception: {e}")
            continue

    # Finalize structure: Convert the target maps to sorted lists for clean output
    final_matching_matrix = {}
    for ifc_id, data in aggregated_ifc_matrix.items():
        ranked_list = list(data["canonical_matches_map"].values())
        
        # Sort matches descending by top score
        ranked_list.sort(key=lambda x: x["score"], reverse=True)

        final_matching_matrix[ifc_id] = {
            "source_text": data["source_text"],
            "source_metadata": data["source_metadata"],
            "matches": ranked_list
        }

    # ── 5. SAVE MATRIX DATA METADATA OUT ──────────────────────────────────
    output_path = os.path.join(project_root, output_filename)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"📁 Creating destination tree directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n💾 Writing true clean IFC -> Canonical matrix output directly to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(final_matching_matrix, out_file, ensure_ascii=False, indent=2)

    print("✨ Matrix processing strategy finished successfully with zero duplicate material IDs.")

if __name__ == "__main__":
    run_matrix_generation()