# scripts/app/pages/upload_ifc.py
import os
import sys
import json
import tempfile
import shutil
import streamlit as st
from pathlib import Path

# ── 1. PROJECT PATH RESOLUTION & CONFIG INITIALIZATION ─────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.load_conf import build_cfg  # Import your config compiler
from src.ifc.extract_ifc_data import extract_ifc_data
from scripts.build_ifc_graph_db import build_ifc_graph_rag
from app.loader.loader import clear_cuda_runtime, get_app_config

# Clean fragmented VRAM early before starting upload calculations
#clear_cuda_runtime()
def render_ifc_json_downloads(cfg):
    target_file_paths = {
        cfg["paths"]["ifc_extracted_element_to_typology"]: "Element Mapping to Typology",
        cfg["paths"]["ifc_extracted_element_missing_typology"]: "Elements Missing Typology",
        cfg["paths"]["ifc_extracted_typology_layers"]: "Typology Layers",
        cfg["paths"]["ifc_extracted_unique_materials"]: "Unique Materials",
    }

    existing_files = {
        path: label
        for path, label in target_file_paths.items()
        if path and os.path.exists(path)
    }

    if not existing_files:
        st.info("No extracted IFC JSON files found yet.")
        return

    st.markdown("### Currently uploaded Ifc Data")

    for path, label in existing_files.items():
        with open(path, "r", encoding="utf-8") as json_f:
            file_data = json.load(json_f)

        file_name = os.path.basename(path)

        with st.expander(f"{label} (`{file_name}`)", expanded=False):
            st.caption(f"File location: `{path}`")
            st.download_button(
                label=f"Download {file_name}",
                data=json.dumps(file_data, indent=2, ensure_ascii=False),
                file_name=file_name,
                mime="application/json",
                key=f"download_{path}",
            )
            st.json(file_data)

# Load the dynamic configuration object (Source of Truth for Paths & API Credentials)
cfg = st.session_state.cfg

# Safely resolve target working directories and runtime parameters from the cfg object
IFC_EXTRACTED_DIR = cfg['paths'].get('ifc_extracted_dir', 'data_in/ifc_extracted')

st.header("📂 IFC Data Engineering & Extraction")
st.markdown(f"""
Upload an Industry Foundation Classes (`.ifc`) BIM file.
The code will isolate IFC ELEMENT NAMES, the COMPOSITION of Material Layers and the MATERIALS themselves.
You can then download these as .json files.
Furthermore, a vector database will be created which enables queries in natural language.
""")

# File Uploader component
uploaded_file = st.file_uploader("Choose an IFC file to analyze", type=["ifc"])
if uploaded_file is not None:
    try:
        # --- STAGE 1: PARSING ---
        with st.spinner("Executing your custom parser logic on IFC elements..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_ifc_path = os.path.join(tmp_dir, "uploaded_model.ifc")
                with open(tmp_ifc_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                tmp_extract_dir = os.path.join(tmp_dir, "extracted_payloads")
                summary = extract_ifc_data(tmp_ifc_path, tmp_extract_dir)

                os.makedirs(IFC_EXTRACTED_DIR, exist_ok=True)
                for filename in os.listdir(tmp_extract_dir):
                    shutil.copy2(
                       os.path.join(tmp_extract_dir, filename),
                       os.path.join(IFC_EXTRACTED_DIR, filename)
                    )
                
                target_ifc_path = cfg['paths']['ifc_extracted_file']
                os.makedirs(os.path.dirname(target_ifc_path), exist_ok=True)
                shutil.copy2(tmp_ifc_path, target_ifc_path)

        # --- STAGE 2: VECTOR EMBEDDING GENERATION ---
        with st.spinner("Building vector database from extracted IFC data..."):
            # Pass the complete configuration dictionary directly into the RAG engine script
            build_ifc_graph_rag(cfg=cfg)

        st.success("🎉 IFC data pipeline executed successfully! "
                   "Generated defined IFC Elements, Construction Typologies, and Unique Material Catalogs.")

        # --- Metrics Grid Layout ---
        st.markdown("### 📊 Extracted Model Schema Summary")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Number of Construction Typologies", summary["unique_typologies"])
        m_col2.metric("Number of Materials", summary["unique_materials_total"])

        total_elements = summary["elements_with_typology"] + summary["elements_without_typology"]
        coverage_pct = f"{(summary['elements_with_typology'] / total_elements) * 100:.1f}%" if total_elements > 0 else "0%"
        m_col3.metric("Elements with defined Construction Typologies", coverage_pct)

        st.divider()

        # --- Target Local Directory File Registry and Viewer ---
        st.markdown("### 📄 Review Local JSON Content")
        """ 
        target_jsons = {
            "element_to_typology.json": "🔗 Element Mapping to Typology",
            "typologies_as_layers.json": "🥞 Material Layers Configurations",
            "unique_materials.json": "🧪 Unique Materials Catalog",
            "elements_without_typology.json": "⚠️ Elements Lacking Material Assignment"
        }
        """
        target_file_paths = {
            cfg["paths"]["ifc_extracted_element_to_typology"]: "🔗 Element Mapping to Typology",
            cfg["paths"]["ifc_extracted_element_missing_typology"]: "🥞 Material Layers Configurations",
            cfg["paths"]["ifc_extracted_typology_layers"]: "🧪 Unique Materials Catalog",
            cfg["paths"]["ifc_extracted_unique_materials"]: "⚠️ Elements Lacking Material Assignment"
        }

        for json_filename, display_label in target_file_paths.items():
            full_file_path = json_filename #os.path.join(IFC_EXTRACTED_DIR, json_filename)
            folder = os.path.dirname(json_filename)
            os.makedirs(folder, exist_ok=True)
            if os.path.exists(full_file_path):
                with open(full_file_path, "r", encoding="utf-8") as json_f:
                    file_data = json.load(json_f)
                with st.expander(f"{display_label} (`{json_filename}`)", expanded=False):
                    st.caption(f"File location: `{full_file_path}`")
                    st.download_button(
                        label=f"⬇ Download {json_filename}",
                        data=json.dumps(file_data, indent=2, ensure_ascii=False),
                        file_name=json_filename,
                        mime="application/json",
                        key=json_filename,
                    )
                    st.json(file_data)

    except Exception as e:
        st.error(f"An error occurred during pipeline parsing sequence: {str(e)}")
        st.exception(e)
    finally:
        # Enforce strict VRAM teardown immediately when page processing drops out of context
        clear_cuda_runtime()
st.divider()
render_ifc_json_downloads(cfg)