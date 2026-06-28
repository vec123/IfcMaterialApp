import os
import sys
import json
import streamlit as st
from dotenv import load_dotenv

# ── 1. ENVIRONMENT & PATH SETTING ──────────────────────────────────────────
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
# Adjust the nesting depth depending on where you place this script
project_root = os.path.abspath(os.path.join(current_dir, "../..")) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.load_conf import load_config
from src.RAG.vectorstore import VectorStoreManager
from src.RAG.searchengine import ConstructionSearchService

# Import your custom Material Extraction Agent
from src.agents.helpers.material_id import MaterialExtractionAgent

# ── 2. INITIALIZATION (Cached for Streamlit Performance) ───────────────────

@st.cache_resource
def initialize_services():
    """Initializes Vector DB services and the custom LLM agent once."""
    VECTOR_DB_PATH = os.environ.get("VECTOR_DB_PATH")
    IFC_VECTOR_DB_PATH = os.environ.get("IFC_VECTOR_DB_PATH")
    CFG_PATH = os.environ.get("CONFIG_PATH", os.path.join(project_root, "conf/config.yaml"))

    if not IFC_VECTOR_DB_PATH:
        st.error("❌ IFC_VECTOR_DB_PATH environment variable not found.")
        st.stop()

    cfg = load_config(CFG_PATH)
    material_collection = cfg['vector_store'].get('material_collection_name', 'ifc_materials_v1')
    model_name = cfg['vector_store']['model_name']

    # Initialize Vector DB
    vs_manager = VectorStoreManager(
        db_path=VECTOR_DB_PATH, 
        collection_name=material_collection, 
        model_name=model_name
    )
    search_service = ConstructionSearchService(vector_store=vs_manager)
    
    # Initialize your custom agent
    # Assumes it inherits configuration setup natively from BaseLLMAgent
    llm_agent = MaterialExtractionAgent()
    
    return search_service, llm_agent, material_collection, model_name

search_service, extraction_agent, material_collection, model_name = initialize_services()

# ── 3. HELPER UTILITY FOR DEDUPLICATION ────────────────────────────────────

def process_hits(hits):
    """Processes, deduplicates, and structures hits identically to your diagnostic script."""
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

st.set_page_config(page_title="IFC Material RAG Explorer", page_icon="🏗️", layout="wide")

st.title("🏗️ IFC Material Search Agent")
st.caption("Extracts material terminology using your specialized LLM Agent and queries the Vector Database.")

# Sidebar status info
with st.sidebar:
    st.header("System Status")
    st.info(f"**Collection:** `{material_collection}`\n\n**Embedding Model:** `{model_name}`")
    st.success("✅ `MaterialExtractionAgent` Loaded")

# User Entry 
user_query = st.text_input(
    "Enter raw identifier or material description:", 
    placeholder="e.g., B051_01_Morter or high-density mineral wool panels"
)

limit = st.slider("Max unique items to return", min_value=1, max_value=50, value=10)

if user_query:
    extracted_material = None
    
    # 1. Execute LLM Agent parsing
    with st.spinner("🤖 Agent is cleaning and parsing the material identifier..."):
        try:
            agent_output = extraction_agent.extract(user_query)
            
            # Extract name from response dictionary {"name": ...}
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
            # Query the database using the processed keyword
            # Multiplied internally to account for unique deduplication dropping items
            raw_hits = search_service.search(query=extracted_material, limit=limit * 3)
            processed_results = process_hits(raw_hits)[:limit]
            
            if not processed_results:
                st.warning(f"No matching materials found for query string: '{extracted_material}'")
            else:
                st.success(f"Found {len(processed_results)} unique matching material types.")
                
                # Render unique materials inside expanders
                for idx, item in enumerate(processed_results, 1):
                    with st.expander(f"[{idx}] Material ID: {item['material_id']} (Score: {item['score']})"):
                        if item['metadata']:
                            st.json(item['metadata'])
                        else:
                            st.text("No metadata payload found for this record.")
                            
        except Exception as e:
            st.error(f"Database Query Failed: {e}")