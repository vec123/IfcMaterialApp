import os
import sys
import json
import streamlit as st
from dotenv import load_dotenv

# ── 1. PROJECT SETUP ──────────────────────────────────────────────────────
load_dotenv()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path: sys.path.insert(0, project_root)

from src.utils.load_conf import build_cfg
from src.RAG.vectorstore import VectorStoreManager
from src.RAG.searchengine import ConstructionSearchService
from app.loader.loader import clear_cuda_runtime, get_material_RAG_search_services

# ── 2. INITIALIZATION ─────────────────────────────────────────────────────

cfg = st.session_state.cfg

MATERIALS_FILE = cfg['paths']['materials_file']

if "materials_db" not in st.session_state:
    with open(MATERIALS_FILE, encoding="utf-8") as f:
        st.session_state.materials_db = json.load(f)

# ── 3. INTERFACE ──────────────────────────────────────────────────────────
st.header("Update Existing Material")
st.markdown("""Here you can update materials that are in the database. 
            Simply search for it and then change the properties as you desire.""")
# Persistent Display of Success Message
if "last_save_msg" in st.session_state:
    st.toast("Success! The materials file has been updated.", icon="✅")
    st.success(st.session_state.last_save_msg)
    del st.session_state.last_save_msg

# Download button for the source file
st.download_button(
    "📥 Download Current Materials JSON",
    data=json.dumps(st.session_state.materials_db, indent=4),
    file_name="materials_updated.json",
    mime="application/json"
)

search_query = st.text_input("Search for a material in the database so you can update it...")

if search_query:
    search_service, extraction_agent, material_collection, model_name = get_material_RAG_search_services(cfg)

    raw_hits = search_service.search(search_query, limit=20)
    options = {}
    for hit in raw_hits:
        # Robust ID extraction logic
        meta = hit.metadata if hasattr(hit, 'metadata') else (hit.get('metadata') if isinstance(hit, dict) else {})
        mid = meta.get('material_id') or (hit.id if hasattr(hit, 'id') else hit.get('id', ''))
        if mid in st.session_state.materials_db:
            options[f"{mid}"] = mid
    st.session_state.search_options = options

# ── 4. UPDATE FORM ────────────────────────────────────────────────────────
target_key = None
if st.session_state.get("search_options"):
    selected = st.selectbox("Select material to update:", list(st.session_state.search_options.keys()))
    target_key = st.session_state.search_options[selected]

if target_key:
    curr = st.session_state.materials_db[target_key]
    with st.form("update_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", value=target_key)
            dens = st.number_input("Density (kg/m³)", value=float(curr.get("density", 0)), step=1.0, format="%.2f")
        with col2:
            tc = st.number_input("Thermal Cond. (W/m·K)", value=float(curr.get("thermal_conductivity", 0)), step=0.001, format="%.4f")
            sh = st.number_input("Specific Heat (J/kg·K)", value=float(curr.get("specific_heat", 0)), step=10.0, format="%.1f")
        
        if st.form_submit_button("Save Changes"):
            # Update local state
            payload = {
                "name": name, 
                "density": dens, 
                "thermal_conductivity": tc, 
                "specific_heat": sh
            }
            st.session_state.materials_db[name] = payload
            
            # Save to disk
            try:
                with open(MATERIALS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(st.session_state.materials_db, f, indent=4)
                
                # Set feedback message and trigger rerun
                st.session_state.last_save_msg = f"Material '{name}' updated successfully in {os.path.basename(MATERIALS_FILE)}."
                st.rerun()
            except Exception as e:
                st.error(f"Error saving to disk: {e}")