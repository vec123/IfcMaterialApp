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
from src.RAG.searchengine import ConstructionSearchService
from src.agents.query.query_typologies.orchestrator import PipelineOrchestrator
from src.agents.query.query_typologies.execution import ExecutionAgent
# Central Cache Routing
from app.loader.loader import get_vector_store, get_llm_agents, clear_cuda_runtime

# Clean fragmented VRAM early before initialization sequences
clear_cuda_runtime()

# Load the dynamic configuration object (Single Source of Truth)
if "cfg" not in st.session_state:
    st.session_state.cfg = build_cfg()
    st.session_state.project_root = project_root

cfg = st.session_state.cfg

# ── 2. CONFIGURATION-DRIVEN PROPERTY TRACKING ───────────────────────────
# Pull paths completely out of your compiled cfg rather than raw environment indicators
CONSTRUCTION_JSON = cfg['ingestion'].get('typologies_file')
DB_PATH = cfg['paths'].get('vector_db_path')

# --- Construction DB Validation & Loading ---
if not CONSTRUCTION_JSON or not os.path.exists(CONSTRUCTION_JSON):
    st.warning(f"⚠️ Construction typologies file missing or not yet uploaded at target: `{CONSTRUCTION_JSON}`")
    st.stop()

if not DB_PATH:
    st.error("🚨 Vector database path configuration property is missing from config schema.")
    st.stop()

if "construction_db" not in st.session_state:
    try:
        with open(CONSTRUCTION_JSON, "r", encoding="utf-8") as f:
            st.session_state.construction_db = json.load(f)
    except Exception as e:
        st.error(f"🚨 Failed to read the typologies file: {e}")
        st.stop()
# --------------------------------------------

# Safe isolated dynamic cached retrievers utilizing material parameters from config
vs_material = get_vector_store(collection_name=cfg['vector_store']['material_collection_name'])
material_search_service = ConstructionSearchService(vector_store=vs_material)

agents = get_llm_agents(cfg)
executor = ExecutionAgent(database=st.session_state.construction_db)
orchestrator = PipelineOrchestrator(agents=agents, search_service=material_search_service, executor=executor)

# ── 3. STREAMLIT UI ────────────────────────────────────────────────────────

st.header("🔍 Intelligent Typology Search Engine")
st.markdown("Discover construction assembly records through natural language queries.")
st.divider()

user_query = st.text_input("Describe your search requirements", placeholder="e.g. 'mineral wool'...", key="orchestrated_query")
top_k = st.slider("Max Search Results", min_value=1, max_value=100, value=5)
run_pipeline = st.button("Execute Query", type="primary")

if run_pipeline and user_query.strip():
    with st.spinner("Processing deep query vectors..."):
        try:
            intents, raw_params, mapped_params, results = orchestrator.run(user_query, top_k=top_k)
            
            with st.expander("🛠️ Multi-Filter Chain Diagnostics Data"):
                st.json({
                    "Executed Intent Sequence": intents, 
                    "Accumulated Raw Parameters": raw_params, 
                    "Accumulated Mapped Parameters": mapped_params
                })
            st.divider()
            
            if not results:
                st.warning("No construction typologies met the structural bounds of this query.")
            else:
                results_subset = results[:top_k] if isinstance(results, list) else list(results.values())[:top_k]
                for idx, element in enumerate(results_subset):
                    details = element if isinstance(element, dict) else {}
                    typo_id = element.get("id") or element.get("typology_id") or element.get("code")
                    
                    if not typo_id or str(typo_id).startswith("Matched Item"):
                        typo_id = next((k for k, v in st.session_state.construction_db.items() if v == element), f"Item #{idx}")
                    
                    with st.expander(f"🏗️ Assembly Typology Name: **{typo_id}**", expanded=True):
                        st.markdown(f"**Category:** `{details.get('category', 'N/A')}` | **U-Value:** {details.get('u_value', 'N/A')}")
                        layers = details.get("layers", [])
                        if layers:
                            # Cast thickness to a string to keep PyArrow from breaking on specialized text sequences
                            st.table([
                                {
                                    "Layer": i, 
                                    "Material": l.get("material"), 
                                    "Thickness (m)": str(l.get("thickness")) if l.get("thickness") is not None else "N/A"
                                } 
                                for i, l in enumerate(layers, 1)
                            ])
                                                                        
        except Exception as e:
            st.error(f"Pipeline Runtime Exception: {e}")
        finally:
            # Enforce clean background environment terminations on thread exit
            clear_cuda_runtime()