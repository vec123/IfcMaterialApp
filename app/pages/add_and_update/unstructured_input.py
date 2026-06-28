import os
import sys
import json
import streamlit as st
from dotenv import load_dotenv

# ── 1. PROJECT SETUP ──────────────────────────────────────────────────────
load_dotenv()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path: 
    sys.path.insert(0, project_root)

from src.utils.load_conf import build_cfg
from src.agents.helpers.unstructured_input_agents import Orchestrator

# ── 2. INITIALIZATION ─────────────────────────────────────────────────────
@st.cache_resource
def get_config_and_orchestrator():
    cfg = build_cfg()
    orchestrator = Orchestrator(cfg)
    return cfg, orchestrator

cfg = st.session_state.cfg

cfg, orchestrator = get_config_and_orchestrator()
MATERIALS_FILE = cfg['paths'].get('unstructured_materials_file', 'data_in/unstructured_materials.json')

os.makedirs(os.path.dirname(MATERIALS_FILE), exist_ok=True)
if not os.path.exists(MATERIALS_FILE):
    with open(MATERIALS_FILE, "w", encoding="utf-8") as f:
        json.dump({}, f)
    st.session_state.materials_db = {}
else:
    with open(MATERIALS_FILE, encoding="utf-8") as f:
        st.session_state.materials_db = json.load(f)

# ── 3. INTERFACE ──────────────────────────────────────────────────────────
st.header("🪄 Robust Material Parsing Pipeline")
st.markdown("""
            If your data is not in a .json you may use this text field to just dump it in. 
            An AI-Agent will try to extract the relevant information and create the .json that you can download.
            Note that each material should have a name, density, heat capacity and thermal conductivity with the stated units.
            Furthermore, the materials can directly be uploaded to the vector database, 
            enabling semantic search with natural language. You should double check it.
            This feature is the newest, so we apologize if it does not work perfectly yet.""")
if "parsed_results" not in st.session_state:
    st.session_state.parsed_results = []

raw_text = st.text_area("Paste large datasheets, CSVs, or raw text here:", height=250)

if st.button("🚀 Process Input"):
    if raw_text.strip():
        with st.spinner("Chunking, extracting, and deduplicating data..."):
            st.session_state.parsed_results = orchestrator.process_raw_input(raw_text)
            st.rerun()

# ── 4. BATCH PROCESSING & COMMIT ─────────────────────────────────────────
if st.session_state.parsed_results:
    st.subheader("Review & Edit Extracted Records")
    
    # Sort missing values to the top
    def is_missing(item):
        return 0.0 in [item.get("density", 0.0), item.get("thermal_conductivity", 0.0), item.get("specific_heat", 0.0)]
    st.session_state.parsed_results.sort(key=is_missing, reverse=True)

    # Use a container instead of a form to prevent button conflicts
    with st.container():
        updated_records = []
        has_errors = False
        
        for i, item in enumerate(st.session_state.parsed_results):
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.write(f"### Record {i+1}")
            with col2:
                if st.button(f"🗑️ Remove", key=f"del_{i}"):
                    st.session_state.parsed_results.pop(i)
                    st.rerun()

            # Input fields
            name = st.text_input(f"Name_{i}", value=item.get("name", ""), key=f"n_{i}")
            
            def get_label(txt, val): return f":red[{txt} (MISSING)]" if val == 0.0 else txt
            
            dens = st.number_input(get_label("Density", item.get("density", 0.0)), value=float(item.get("density", 0.0)), step=0.1, format="%.2f", key=f"d_{i}")
            tc = st.number_input(get_label("Thermal Cond", item.get("thermal_conductivity", 0.0)), value=float(item.get("thermal_conductivity", 0.0)), step=0.001, format="%.4f", key=f"t_{i}")
            sh = st.number_input(get_label("Spec Heat", item.get("specific_heat", 0.0)), value=float(item.get("specific_heat", 0.0)), step=1.0, format="%.1f", key=f"s_{i}")
            
            if 0.0 in [dens, tc, sh]: has_errors = True
            
            updated_records.append({"name": name, "density": dens, "thermal_conductivity": tc, "specific_heat": sh})
            st.divider()

        # Final Commit button outside the loop
        if st.button("Commit All to Database", disabled=has_errors):
            for rec in updated_records:
                if rec["name"]:
                    st.session_state.materials_db[rec["name"]] = rec
            
            with open(MATERIALS_FILE, 'w', encoding='utf-8') as f:
                json.dump(st.session_state.materials_db, f, indent=4)
            
            st.toast("Batch committed successfully!", icon="✅")
            st.session_state.parsed_results = []
            st.rerun()
            
        if has_errors:
            st.error("⚠️ Some fields are still 0.0. Please correct them or remove the records to enable submission.")