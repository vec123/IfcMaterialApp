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
from scripts.build_graph_db import build_separated_graph_rag
from app.loader.loader import clear_cuda_runtime

# Clean fragmented VRAM early before starting upload calculations
clear_cuda_runtime()

# Load the dynamic configuration object (Single Source of Truth)
cfg = build_cfg()

# Safely resolve target working file paths directly from the fully compiled cfg object
MAT_FILE_PATH  = cfg['ingestion'].get('materials_file')
TYPO_FILE_PATH = cfg['ingestion'].get('typologies_file')

st.header("🥞 Upload Typologies")
st.markdown("""
Upload your `typologies.json` to vectorize it into the database.
The typology names will be vectorized it into the database enabling searching it with natural language.
Currently the App focuses on Materials, so this is not strictly necessary. 
In the future we might support natural language interaction with typologies.
            
This page expects a fixed format. 
{
  "typology name 1": {
    "key_1": "value_1",
    "key_2": "value_2",
    "key_3": "value_3",
            .
            .
            .
    "layers": [
      {
        "material": "name 1",
        "thickness": 0.115
      },
      {
        "material": "name 2",
        "thickness": 0.01
      },
      {
        "material": "name 3",
        "thickness": "eAT"
      },
            .
            .
            .
      }
    ]
  },
  "typology name 2": {
            
If `materials.json` with equal material names has already been uploaded, the connections will be created automatically.
""")

# --- Validation Checks ---
if not TYPO_FILE_PATH:
    st.error("❌ Destination path for `typologies_file` is missing from the configuration schema.")
    st.stop()

uploaded = st.file_uploader("Upload `typologies.json`", type=["json"])

if uploaded:
    # 1. Validate JSON content safely
    try:
        content = uploaded.getvalue()
        json.loads(content.decode("utf-8"))  # validate structure
    except (json.JSONDecodeError, UnicodeDecodeError):
        st.error("❌ Invalid JSON — file was not saved.")
        st.stop()

    # 2. Extract directory dynamically and save the uploaded file
    typo_dir = os.path.dirname(TYPO_FILE_PATH)
    if typo_dir:
        os.makedirs(typo_dir, exist_ok=True)
        
    try:
        with open(TYPO_FILE_PATH, "wb") as f:
            f.write(content)
        st.success(f"💾 Saved file to `{TYPO_FILE_PATH}`")
    except Exception as save_err:
        st.error(f"❌ Failed writing file to disk space: {str(save_err)}")
        st.stop()

    # 3. Check for Materials Counterpart
    mat_ready = bool(MAT_FILE_PATH and os.path.exists(MAT_FILE_PATH))

    if mat_ready:
        st.success(f"✅ Materials file found at `{MAT_FILE_PATH}` — graph edges will be generated.")
    else:
        st.warning(
            f"Materials file not found at `{MAT_FILE_PATH or 'undefined path'}`. "
            "Graph edges will be skipped. Upload materials to generate them."
        )

    # 4. Trigger DB Build Process
    try:
        with st.spinner("Building vector database..."):
            # Pass the complete, pre-compiled configuration dictionary directly as 'cfg'
            results = build_separated_graph_rag(cfg=cfg)

        st.success("🎉 Build complete!")
        
        # Display Metrics Safely
        if isinstance(results, dict):
            c1, c2, c3 = st.columns(3)
            c1.metric("Materials indexed",   results.get("materials_count", 0))
            c2.metric("Typologies indexed",  results.get("typologies_count", 0))
            c3.metric("Graph edges written", results.get("edges_count", 0))
        else:
            st.info("Pipeline executed successfully!")

    except Exception as e:
        st.error(f"Build failed: {str(e)}")
        st.exception(e)
    finally:
        # Enforce strict VRAM teardown immediately when processing context terminates
        clear_cuda_runtime()