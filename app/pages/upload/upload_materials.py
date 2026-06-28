import os
import sys
import json
import streamlit as st

# ── 1. PROJECT PATH RESOLUTION & CONFIG INITIALIZATION ─────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.load_conf import build_cfg
from scripts.build_graph_db import build_separated_graph_rag
from app.loader.loader import clear_cuda_runtime, get_app_config

# Clean fragmented VRAM early
#clear_cuda_runtime()
cfg = get_app_config()

MAT_FILE_PATH  = cfg['paths'].get('materials_file')
TYPO_FILE_PATH = cfg['paths'].get('typologies_file')

st.header("🧪 Upload Materials")
st.markdown("""
Upload a `materials.json`. Only materials with positive values for **density**, 
**thermal_conductivity**, and **specific_heat** will be processed.
The expected format is:
            
    {
        "name 1": {
            "density": 780.0,
            "thermal_conductivity": 0.3499999999999999,
            "specific_heat": 1000.0
        },
            .
            .
            .
    }    
If you do not have this format available, you may use the AI-assisted Upload tab to get it.
            
""")

if not MAT_FILE_PATH:
    st.error("❌ Destination path for `materials_file` is missing.")
    st.stop()

uploaded = st.file_uploader("Upload `materials.json`", type=["json"])

if uploaded:
    # 1. Parse and Validate
    try:
        content = uploaded.getvalue()
        data = json.loads(content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        st.error("❌ Invalid JSON format.")
        st.stop()

    required_fields = ["density", "thermal_conductivity", "specific_heat"]
    valid_data = {}
    invalid_report = []

    for name, props in data.items():
        if not isinstance(props, dict):
            invalid_report.append(f"**{name}**: Invalid structure.")
            continue

        errors = []
        for field in required_fields:
            val = props.get(field)
            if val is None:
                errors.append(f"missing '{field}'")
            elif not isinstance(val, (int, float)) or val <= 0:
                errors.append(f"invalid value for '{field}' ({val})")
        
        if errors:
            invalid_report.append(f"**{name}**: {', '.join(errors)}")
        else:
            valid_data[name] = props

    # 2. Immediate Feedback Loop
    if invalid_report:
        st.error("❌ Some materials were excluded due to issues:")
        for error in invalid_report:
            st.markdown(f"- {error}")
    
    if not valid_data:
        st.warning("⚠️ No valid materials found. Please correct the JSON and re-upload.")
        st.stop()

    # 3. Save Cleaned Data
    mat_dir = os.path.dirname(MAT_FILE_PATH)
    os.makedirs(mat_dir, exist_ok=True)
    
    try:
        cleaned_content = json.dumps(valid_data, indent=4).encode("utf-8")
        with open(MAT_FILE_PATH, "wb") as f:
            f.write(cleaned_content)
        st.success(f"💾 Registered {len(valid_data)} valid materials.")
    except Exception as save_err:
        st.error(f"❌ Failed writing file: {str(save_err)}")
        st.stop()

    # 4. Check for Typologies
    typo_ready = bool(TYPO_FILE_PATH and os.path.exists(TYPO_FILE_PATH))
    if typo_ready:
        st.success("✅ Typologies have been uploaded — Materials will be connected.")
    else:
        st.warning("⚠️ Typologies have not been uploaded - Materials will not be connected.")

    # 5. Build DB
    try:
        # Initialize session state key if it doesn't exist
        if "build_results" not in st.session_state:
            st.session_state.build_results = None

        if st.button("Build Database"):
            with st.spinner("Building vector database..."):
                # Run the build and store in session_state
                st.session_state.build_results = build_separated_graph_rag(cfg=cfg)
                st.success("🎉 Build complete!")

        # Always check the session state for data to display
        results = st.session_state.build_results
        
        if isinstance(results, dict):
            c1, c2, c3 = st.columns(3)
            c1.metric("Materials indexed", results.get("materials_count", 0))
            c2.metric("Typologies indexed", results.get("typologies_count", 0))
            c3.metric("Connections found", results.get("edges_count", 0))
            
    except Exception as e:
        st.error(f"Build failed: {str(e)}")
        st.exception(e)
    finally:
        clear_cuda_runtime()