"""
scripts/app/app.py
Main entrypoint using Streamlit's Native Page Routing. Run from IfcMaterialAgent root:
    PYTHONPATH=. streamlit run scripts/app/app.py
"""

import os
import sys
import json
import streamlit as st

# 1. Path Fix: Go up 2 levels from scripts/app/app.py to project root (IfcMaterialAgent)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.loader.loader import get_vector_store, get_llm_agents
from src.utils.load_conf import load_config
from app.pages.components.sidebar import render_shared_sidebar


# ── Session State Initializers ────────────────────────────────────────────────

def init_shared_state():
    if "cfg" not in st.session_state:
        st.session_state.cfg = load_config(os.path.join(project_root, "conf/config.yaml"))
        st.session_state.project_root = project_root
     
    #cfg = st.session_state.cfg
     
    #if "construction_db" not in st.session_state:
    #    p = os.path.join(project_root, cfg["ingestion"]["input_file"])
    #    with open(p, encoding="utf-8") as f:
    #        st.session_state.construction_db = json.load(f)
            
    #if "materials_db" not in st.session_state:
    #    p = os.path.join(project_root, cfg["ingestion"]["materials_file"])
    #    with open(p, encoding="utf-8") as f:
    #        st.session_state.materials_db = json.load(f)
    



# ── Landing Page Render ───────────────────────────────────────────────────────
def render_landing_page():
    st.header("🏗️ EnergIA BIM-BEPS Intelligence Platform")
    st.markdown("""
    This tool is designed to facilitate interoperability between material databases and **IFC** files. To use it effectively, please ensure your IFC files meet the following requirements:

        * **1. Material Specification:** Materials must be specified by name and available via one of the following:
            * `IfcMaterialLayerSetUsage`
            * `IfcMaterialLayerSet`
            * `IfcMaterialLayer`
            * `IfcMaterial`
            * `IfcMaterialList`
        * **2. Descriptive Naming:** The material name must be descriptive to ensure accurate matching.

    These names can matched to an material database by semantically aligning names.
    This links important properties such as physical parameters and/or LCA metrics directly to the IFC project.

    > **Note:** As AI can make mistakes, a human validation step is encouraged before finalizing the project.
                
    Please use the **Sidebar Navigation** panel to discover and manage items across sections:
    
    ### 🛠️ IFC
    * **Upload Ifc:** Upload your .ifc and get the ifc element names, the layer compostions of each element and unique material names used in each layer,            
    * **Search Ifc Materialc:** Find Materials in the IFC. 
                
    ### 🛠️ Material Database            
    * **Upload Materials:** Upload your Materials as structured .json.   
    * **Search Material:** Search for Materials by asking the AI Agent.
    * **Add Material:** Add a new Material through an input field or text description.
    * **AI Assisted Upload:** Dump Your Database through copy paste. Let the AI Agent convert it.  
    * **Update existing Material:** Search for a Material, then change its properies.          
                          
    ### 🛠️ Matching
    * **Match:** Match material names from the IFC with material names from the Database.

    """)


# ── Global Execution Configuration ───────────────────────────────────────────
st.set_page_config(
    page_title="EnergIA BIM-BEPS · Construction Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize runtime files
init_shared_state()
if "cfg" not in st.session_state:
    st.session_state.cfg = load_config(os.path.join(project_root, "conf/config.yaml"))

with st.sidebar:
    render_shared_sidebar(init_state_fn=init_shared_state)
    
# ── NATIVE STREAMLIT MULTI-PAGE ROUTER DECLARATION ─────────────────────────
# Declare references targeting individual page execution files cleanly
landing_view = st.Page(render_landing_page, title="Platform Overview", icon="🏢", default=True)


upload_ifc  = st.Page("pages/upload/upload_ifc.py", title="Upload IFC Model", icon="📂")
upload_typo= st.Page("pages/upload/upload_typologies.py", title="Upload Your Typologies", icon="📂")
upload_materials = st.Page("pages/upload/upload_materials.py", title="Upload Your Materials", icon="📂")


search_typo = st.Page("pages/search/search_typologies.py", title="Search Typologies", icon="🔍")
search_mat  = st.Page("pages/search/search_materials.py", title="Search Materials", icon="🔍")
query_ifc   = st.Page("pages/search/search_ifc_materials.py", title="Search IFC Materials", icon="🔍")

add_mat     = st.Page("pages/add_and_update/add_materials.py", title="Add New Material", icon="📂")
update_mat  = st.Page("pages/add_and_update/update_materials.py", title="Update Existing Material", icon="🔄")
unstructured_input  = st.Page("pages/add_and_update/unstructured_input.py", title="AI-assisted Upload", icon="📂")

match_mats   = st.Page("pages/match/material_matching.py", title="Match Materials", icon="🧬")

# Structure navigation collections visually by category headers
pg = st.navigation({
    "General": [landing_view],
    "IFC": [upload_ifc, query_ifc],
    "Material Database": [upload_materials,search_mat, add_mat, unstructured_input, update_mat],
    "Matching": [match_mats]
})

# Run global brand components alongside native pages runtime execution
#render_shared_sidebar(init_state_fn=init_shared_state)
pg.run()