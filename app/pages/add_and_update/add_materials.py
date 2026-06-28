import os
import sys
import json
import streamlit as st
from dotenv import load_dotenv

# ── 1. ENVIRONMENT & CENTRALIZED CONFIG LOADING ───────────────────────────
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.load_conf import build_cfg  # Import your unified config compiler
from src.RAG.searchengine import ConstructionSearchService
from app.loader.loader import get_llm_agents, get_vector_store, get_material_RAG_search_services
from src.agents.add_mats.agents import MaterialDuplicateCheckAgent, MaterialPersistenceAgent


# Initialize Single Source of Truth configuration via Session State
if "cfg" not in st.session_state:
    st.session_state.cfg = build_cfg()
    st.session_state.project_root = project_root

cfg = st.session_state.cfg

# Extract declarative system file coordinates out of the central config map
VECTOR_DB_PATH    = cfg['paths'].get('vector_db_path')
MATERIALS_FILE    = cfg['paths'].get('materials_file')
COLLECTION_NAME   = cfg['vector_store'].get('material_collection_name', 'materials_v1')
MODEL_NAME = cfg['vector_store'].get('embedding_model', None)

if not VECTOR_DB_PATH:
    st.error("🚨 Configuration Error: 'vector_db_path' property key missing inside the loaded setup map.")
    st.stop()
if "materials_db" not in st.session_state:
    if MATERIALS_FILE and os.path.exists(MATERIALS_FILE):
        with open(MATERIALS_FILE, encoding="utf-8") as f:
            st.session_state.materials_db = json.load(f)
    else:
        st.error(f"🚨 Target Materials Storage Document cannot be found at track location: {MATERIALS_FILE}")
        st.stop()


st.header("Add New Material")
st.markdown("Create a new entry in your database. " \
"Use the AI text extraction or type manually.")
st.divider()

# ── 2. STATE STORAGE ASSIGNMENT INITIALIZATION ────────────────────────────
if "add_name" not in st.session_state: st.session_state.add_name = ""
if "add_density" not in st.session_state: st.session_state.add_density = None
if "add_tc" not in st.session_state: st.session_state.add_tc = None
if "add_sh" not in st.session_state: st.session_state.add_sh = None

st.markdown("### 🪄 Step 1: AI Prompt Extraction (Optional)")
user_prompt = st.text_area("Describe parameters in plain text", placeholder="e.g., add veritasium with density of 5000...", key="add_ai_prompt")

if st.button("AI Extract & Autofill Fields", type="secondary"):
    if user_prompt.strip():
        search_service, extraction_agent, material_collection, model_name = get_material_RAG_search_services(cfg)
        
        with st.spinner("Extracting parameters using LLM Agent..."):
            try:
                extracted = extraction_agent.extract(user_prompt)
                if extracted:
                    if extracted.get("name"): st.session_state.add_name = extracted["name"]
                    st.session_state.add_density = float(extracted["density"]) if extracted.get("density") is not None else None
                    st.session_state.add_tc = float(extracted["thermal_conductivity"]) if extracted.get("thermal_conductivity") is not None else None
                    st.session_state.add_sh = float(extracted["specific_heat"]) if extracted.get("specific_heat") is not None else None
                    st.success("Autofilled fields below. Review parameters before submitting.")
                    st.rerun()
                else:
                    st.error("LLM failed to return structured property bounds.")
            except Exception as e:
                st.error(f"Extraction agent failed: {e}")
    else:
        st.warning("Please type a description string first.")

st.divider()
st.markdown("### 🛠️ Step 2: Verify & Submit Properties")

# ── 3. FORM VALIDATION & PERSISTENCE MANAGEMENT ───────────────────────────
with st.form("add_materials_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        name_input = st.text_input("Material Name", value=st.session_state.add_name)
        density_input = st.number_input("Density (kg/m³)", value=st.session_state.add_density, step=1.0, format="%.2f")
    with col2:
        tc_input = st.number_input("Thermal Conductivity (W/m·K)", value=st.session_state.add_tc, step=0.001, format="%.4f")
        sh_input = st.number_input("Specific Heat (J/kg·K)", value=st.session_state.add_sh, step=10.0, format="%.1f")
        
    submit_add = st.form_submit_button("Commit New Material to Database", type="primary")

if submit_add:
   
    agents = get_llm_agents(cfg)
    raw_name = name_input.strip()
    clean_name = raw_name
    for suffix in ["material", "materiales", "producto", "capa", "aislante", "board", "insulation", "panel"]:
        if clean_name.lower().endswith(suffix): clean_name = clean_name[:-len(suffix)].strip()
    if not clean_name: clean_name = raw_name
    
    working_payload = {"name": clean_name, "density": density_input, "thermal_conductivity": tc_input, "specific_heat": sh_input}
    
    # Route through configured system module models
    name_validator = agents["name_validator"]
    prop_validator = agents["prop_validator"]
    dup_checker    = MaterialDuplicateCheckAgent(st.session_state.materials_db)
    persistence    = MaterialPersistenceAgent(MATERIALS_FILE)
    
    with st.spinner("Processing structural validations..."):
        name_res = name_validator.validate(working_payload["name"])
        prop_res = prop_validator.validate(working_payload)
        dup_res  = dup_checker.check(working_payload["name"], search_service)
        
        validation_errors = []
        if not name_res.get("valid", True): validation_errors.append(name_res.get("issue"))
        if not prop_res.get("valid", True): validation_errors.extend(prop_res.get("issues", []))
        
        for key, label in [("density", "Density"), ("thermal_conductivity", "Thermal Conductivity"), ("specific_heat", "Specific Heat")]:
            if working_payload[key] is None or float(working_payload[key]) == 0.0:
                validation_errors.append(f"Validation Error: '{label}' cannot be blank or 0.0.")
        
        if dup_res.get("exists"): validation_errors.append(f"Validation Error: A material named '{working_payload['name']}' already exists.")
            
        if dup_res.get("similar_materials"):
            st.info("💡 **Similar materials found in database search:**")
            for sug in dup_res["similar_materials"]: st.markdown(f"- `{sug['name']}` (Distance: {sug['score']})")

        if not validation_errors:

            try:
                vs_material    = get_vector_store(db_path=VECTOR_DB_PATH,collection_name=COLLECTION_NAME,model_name=MODEL_NAME)
                persistence.save(working_payload, vs_material, st.session_state.materials_db)
                st.success(f"🎉 Success: Material '{working_payload['name']}' successfully added.")
                
                # Zero out scratch states upon confirmation of a successful payload write operation
                st.session_state.add_name = ""
                st.session_state.add_density = None
                st.session_state.add_tc = None
                st.session_state.add_sh = None
                st.rerun()
            except Exception as e:
                st.error(f"Persistence Storage Failure: {e}")
        else:
            for err in validation_errors: st.error(err)