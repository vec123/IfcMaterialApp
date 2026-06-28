import os
import sys
import json
import streamlit as st

# ── 1. PROJECT PATH RESOLUTION & CONFIG INITIALIZATION ─────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.load_conf import build_cfg  # Import your unified config compiler
from src.RAG.vectorstore import VectorStoreManager
from src.RAG.searchengine import ConstructionSearchService
from src.agents.helpers.material_id import MaterialExtractionAgent
from app.loader.loader import clear_cuda_runtime, get_app_config, get_material_RAG_search_services

# Clean fragmented VRAM early before initialization sequences
#clear_cuda_runtime()

# Load the dynamic configuration object (Single Source of Truth)
cfg = st.session_state.cfg

# ── 3. HELPER UTILITY FOR DEDUPLICATION ────────────────────────────────────

def process_hits(hits):
    """Processes, deduplicates, and structures hits identically to diagnostic tools."""
    unique_results = []
    seen_material_ids = set()
    
    for hit in hits:
        raw_id = hit.get("id") if isinstance(hit, dict) else getattr(hit, "id", "Unknown")
        
        score = "N/A"
        if isinstance(hit, dict):
            score = hit.get("score") or hit.get("distance") or hit.get("score_match", "N/A")
        else:
            score = getattr(hit, "score", None) or getattr(hit, "distance", "N/A")
            
        metadata = {}
        if isinstance(hit, dict):
            metadata = hit.get("metadata") or hit.get("metas") or hit.get("payload") or {}
        else:
            metadata = getattr(hit, "metadata", None) or getattr(hit, "payload", None) or getattr(hit, "metas", {})

        material_id = metadata.get("material_id")
        if not material_id and "##" in str(raw_id):
            material_id = str(raw_id).split("##")[0]
        elif not material_id:
            material_id = raw_id

        if material_id in seen_material_ids:
            continue
            
        seen_material_ids.add(material_id)
        unique_results.append({
            "material_id": material_id,
            "score": score,
            "metadata": metadata
        })
        
    return unique_results

# ── 4. STREAMLIT UI ────────────────────────────────────────────────────────

st.set_page_config(page_title="Material RAG Explorer", page_icon="🏗️", layout="wide")

st.title("🏗️ Material Search")
st.caption("Search the Material that are in the uploaded Material Database by inputing a name." \
" We try to be as agnostic to language and exact forumlations as possible.")

collection_name=cfg["vector_store"]["material_collection_name"],
model_name=cfg["vector_store"]["embedding_model"],
# Sidebar status info driven completely via config properties
with st.sidebar:
    st.header("System Status")
    st.info(f"**Collection:** `{collection_name}`\n\n**Embedding Model:** `{model_name}`")
    st.success("✅ `MaterialExtractionAgent` Loaded")

# User entry window 
user_query = st.text_input(
    "Enter raw identifier or material description:", 
    placeholder="e.g., B051_01_Morter or high-density mineral wool panels"
)

#limit = st.slider("Max unique items to return", min_value=1, max_value=50, value=10)
limit = 10000
if user_query:
    search_service, extraction_agent, material_collection, model_name = get_material_RAG_search_services(cfg)

    extracted_material = None
    
    # 1. Execute LLM Agent parsing
    with st.spinner("🤖 Agent is cleaning and parsing the material identifier..."):
        try:
            agent_output = extraction_agent.extract(user_query)
            
            # Extract clean material keyword from response map {"name": ...}
            extracted_material = agent_output.get("name")
            
            if extracted_material:
                st.markdown(f"🎯 **Agent Extracted Keyword:** `{extracted_material}`")
            else:
                st.warning("⚠️ Agent could not deduce a clear material name. Falling back to raw entry.")
                extracted_material = user_query
                
        except Exception as e:
            st.error(f"Failed to process query with MaterialExtractionAgent: {e}")
            extracted_material = user_query  # Fallback to original raw input on error

    # 2. Vector Database Search
    with st.spinner("🔍 Querying Vector Database..."):
        try:
            # Query database utilizing agent metadata transformations
            raw_hits = search_service.search(query=extracted_material, limit=limit * 3)
            processed_results = process_hits(raw_hits)[:limit]
            
            if not processed_results:
                st.warning(f"No matching materials found for query string: '{extracted_material}'")
            else:
                st.success(f"Found {len(processed_results)} unique matching material types.")
                
                # Render unique materials inside interactive components
                for idx, item in enumerate(processed_results, 1):
                    with st.expander(f"[{idx}] Material ID: {item['material_id']} (Score: {item['score']})"):
                        if item['metadata']:
                            st.json(item['metadata'])
                        else:
                            st.text("No metadata payload found for this record.")
                            
        except Exception as e:
            st.error(f"Database Query Failed: {e}")
        finally:
            # Enforce clean background environment terminations on thread exit
            clear_cuda_runtime()