import os
import sys
import json  # Added for pretty printing metadata
from dotenv import load_dotenv

# ── 1. ENVIRONMENT & PATH SETTING ──────────────────────────────────────────
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
# Adjust the nesting depth depending on where you place this script
project_root = os.path.abspath(os.path.join(current_dir, "../../..")) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.load_conf import load_config
from src.RAG.vectorstore import VectorStoreManager
from src.RAG.searchengine import ConstructionSearchService

# ── 2. INITIALIZATION ──────────────────────────────────────────────────────
VECTOR_DB_PATH = os.environ.get("VECTOR_DB_PATH")
IFC_VECTOR_DB_PATH = os.environ.get("IFC_VECTOR_DB_PATH")
CFG_PATH = os.environ.get("CONFIG_PATH", os.path.join(project_root, "conf/config.yaml"))

if not IFC_VECTOR_DB_PATH:
    print("❌ Error: IFC_VECTOR_DB_PATH environment variable not found.")
    sys.exit(1)

print(f"📂 Loading Configuration from: {CFG_PATH}")
cfg = load_config(CFG_PATH)

material_collection = cfg['vector_store'].get('material_collection_name', 'ifc_materials_v1')
model_name = cfg['vector_store']['model_name']

print(f"📦 Connecting to Vector Store at: {IFC_VECTOR_DB_PATH}")
print(f"🏷️ Target Collection: {material_collection}")
print(f"🤖 Embedding Model: {model_name}\n")

# Initialize the manager and engine
vs_manager = VectorStoreManager(
    db_path=VECTOR_DB_PATH, 
    collection_name=material_collection, 
    model_name=model_name
)
search_service = ConstructionSearchService(vector_store=vs_manager)

# ── 3. DIAGNOSTIC EXPERIMENTS ──────────────────────────────────────────────

def print_results(hits):
    if not hits:
        print("   No matches found.")
        return
    
    seen_material_ids = set()
    printed_count = 0
    
    for hit in hits:
        # 1. Parse ID and Score
        raw_id = hit.get("id") if isinstance(hit, dict) else getattr(hit, "id", "Unknown")
        
        score = "N/A"
        if isinstance(hit, dict):
            score = hit.get("score") or hit.get("distance") or hit.get("score_match", "N/A")
        else:
            score = getattr(hit, "score", None) or getattr(hit, "distance", "N/A")
            
        # 2. Extract Metadata
        metadata = {}
        if isinstance(hit, dict):
            metadata = hit.get("metadata") or hit.get("metas") or hit.get("payload") or {}
        else:
            metadata = getattr(hit, "metadata", None) or getattr(hit, "payload", None) or getattr(hit, "metas", {})

        # 3. DEDUPLICATION LOGIC
        # Extract the base material ID from metadata
        material_id = metadata.get("material_id")
        
        # Fallback: If metadata lacks material_id, try parsing it from the raw vector ID (before the '##var')
        if not material_id and "##" in str(raw_id):
            material_id = str(raw_id).split("##")[0]
        elif not material_id:
            material_id = raw_id

        # Skip if we've already printed this core material
        if material_id in seen_material_ids:
            continue
            
        seen_material_ids.add(material_id)
        printed_count += 1

        # 4. Print Unique Result
        print(f"   [{printed_count}] Material ID: {material_id:<45} | Top Match Score: {score}")
        
        if metadata:
            import json
            meta_str = json.dumps(metadata, ensure_ascii=False, indent=2)
            print("metadata: ", meta_str)
            indented_meta = "\n".join(f"       {line}" for line in meta_str.splitlines())
            # print(f"       📄 Metadata:\n{indented_meta}")
        else:
            print("       📄 Metadata: No metadata found.")
        print("-" * 50)
        
        # Limit to the top 5 *unique* items
        if printed_count >= 50:
            break

print("\n" + "=" * 80)
print("🔬 TEST 2: Query ('mineral wool')")
print("=" * 80)
try:
    # REMOVE the explicit "query: " prefix here!
    pipe_query = "mineral wool" 
    pipe_hits = search_service.search(query=pipe_query, limit=50000)
    print_results(pipe_hits)
except Exception as e:
    print(f"❌ Test 2 Failed: {e}")

print("\n" + "=" * 80)
print("🔬 TEST 2:")
print("=" * 80)
try:
    pipe_query = "query: wood"
    pipe_hits = search_service.search(query=pipe_query, limit=50000)
    print_results(pipe_hits)
except Exception as e:
    print(f"❌ Test 2 Failed: {e}")

print("\n" + "=" * 80)
print("🏁 DIAGNOSTIC ANALYSIS")
print("=" * 80)
print("👉 If TEST 1 shows varied scores (e.g., 0.84, 0.61, 0.32) but TEST 2 shows")
print("   the exact same flat numbers (e.g., 0.1176), your application query logic")
print("   MUST be modified to iterate through translation lists instead of joining them.")
print("=" * 80)