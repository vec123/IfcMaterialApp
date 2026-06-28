import streamlit as st
import json, os
from scripts.make_matching import run_matrix_generation
from src.utils.load_conf import build_cfg
from src.ifc.write_wall_thermal_props import update_ifc_with_assembly_properties
st.set_page_config(layout="wide", page_title="Material Matching")

# ── 1. CONFIGURATION & STATE ──────────────────────────────────────────
cfg = build_cfg()
MATRIX_JSON = cfg['paths'].get('tmp_match_matrix')
MATCH_MATRIX = cfg['paths'].get('match_matrix')

if "manual_mappings" not in st.session_state:
    if os.path.exists(MATCH_MATRIX):
        with open(MATCH_MATRIX, "r", encoding="utf-8") as f:
            st.session_state.manual_mappings = json.load(f)
    else:
        st.session_state.manual_mappings = {}

# ── 2. INTERFACE ──────────────────────────────────────────────────────
st.title("🏗️ Material Matching Workspace")
st.markdown(""" Match the material names in the uploaded Database to those in the IFC
            The Calculate Semantic Matching Button compares the meaning of the names and 
            proposes a first ranked matching. You are expeceted to validate it, since it can be wrong.
            Hopefully the best match is in the first few proposals.
            Of course you can download the ranked matching and your own matching as a .json.
            """)
# TOP ACTION CENTER
col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    if st.button("🚀 (Re-) Calculate Semantic Matching ", use_container_width=True):
        # 1. Define paths from your cfg
        # We check the directory path itself
        mat_db_dir = cfg['paths'].get('vector_db_path')
        ifc_db_dir = cfg['paths'].get('ifc_vector_db_path')
        
        # 2. Check existence and basic "is it a DB" validation
        def is_valid_db(path):
            # A valid ChromaDB dir must exist and contain the sqlite file
            return path and os.path.exists(path) and os.path.exists(os.path.join(path, "chroma.sqlite3"))

        missing = []
        if not is_valid_db(mat_db_dir): missing.append("Canonical Library DB")
        if not is_valid_db(ifc_db_dir): missing.append("IFC Project Store DB")
        
        if missing:
            st.warning(f"⚠️ Warning: Databases not found or incomplete: {', '.join(missing)}.\n\nEnsure your embedding/extraction pipeline has populated these folders.")
        else:
            with st.spinner("Processing vectors..."):
                try:
                    # Execute the matching matrix generation
                    run_matrix_generation(cfg=cfg)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during vector processing: {e}")

with col_b:
    if st.session_state.manual_mappings:
        st.download_button(
            "💾 Download Manually Verified Matches (.json)", 
            data=json.dumps(st.session_state.manual_mappings, indent=2), 
            file_name="manual_validated_matches.json",
            mime="application/json",
            use_container_width=True
        )

with col_c:
    if os.path.exists(MATRIX_JSON):
        with open(MATRIX_JSON, "r", encoding="utf-8") as f:
            st.download_button(
                "⚙️ Download Automatic Semantic Matches (.json)", 
                data=f.read(), 
                file_name="automatic_matches.json",
                mime="application/json",
                use_container_width=True
            )
with col_d:
    # 1. Selection Mechanism
    mapping_source = st.radio(
        "Choose Mapping Source:",
        ["Automatic Semantic", "Manually Verified"],
        horizontal=True
    )
    
    # 2. Logic to determine which mapping to use
    if st.button("🏗️ Generate & Download Enriched IFC", use_container_width=True):
        with st.spinner("Injecting thermal properties into IFC..."):
            ifc_in = cfg["paths"]["ifc_extracted_file"]
            ifc_out = "enriched_model.ifc"
            
            # Load mappings based on selection
            if mapping_source == "Manually Verified":
                mapping = st.session_state.manual_mappings
            else:
                with open(MATRIX_JSON, "r", encoding="utf-8") as f:
                    mapping = json.load(f)
            
            element_map = json.load(open(cfg["paths"]["ifc_extracted_element_to_typology"], "r", encoding="utf-8"))
            layers_map = json.load(open(cfg["paths"]["ifc_extracted_typology_layers"], "r", encoding="utf-8"))
            
            # Call your update function
            update_ifc_with_assembly_properties(ifc_in, ifc_out, mapping, element_map, layers_map)
            
            # Offer download
            with open(ifc_out, "rb") as f:
                st.download_button(
                    "📥 Click to Download .ifc",
                    data=f,
                    file_name=f"enriched_{mapping_source.replace(' ', '_').lower()}.ifc",
                    mime="application/octet-stream",
                    use_container_width=True
                )

st.divider()

if not os.path.exists(MATRIX_JSON):
    st.error("Matrix file not found. Please click 'Re-calculate Matrices' above.")
    st.stop()

with open(MATRIX_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# DROPDOWN
dropdown_map = {}
for ifc_id in data.keys():
    status = "✓" if ifc_id in st.session_state.manual_mappings else " "
    val = st.session_state.manual_mappings.get(ifc_id, {}).get("canonical_material_id", "Not Yet Validated")
    dropdown_map[f"[{status}] {ifc_id} ({val})"] = ifc_id

selected_label = st.selectbox("Select IFC Material:", list(dropdown_map.keys()))
selected_ifc = dropdown_map[selected_label]

# ── 3. VALIDATION VIEW ────────────────────────────────────────────────
if selected_ifc:
    item = data[selected_ifc]
    current_val = st.session_state.manual_mappings.get(selected_ifc)
    
    if current_val:
        st.success(f"**Current Validation:** {current_val['canonical_material_id']} (Score: {float(current_val.get('score', 0)):.4f})")
    else:
        st.warning("**Status:** Not Yet Validated")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Source IFC Info")
        st.json(item['source_metadata'])
    
    with col2:
        st.subheader("Canonical Suggestions")
        for match in item['matches']:
            c_id = match['canonical_material_id']
            score = float(match.get('score') or 0.0)
            is_verified = bool(current_val and current_val.get("canonical_material_id") == c_id)
            
            with st.expander(f"Match: {c_id} (Score: {score:.4f})", expanded=is_verified):
                st.json(match.get("metadata", {}))
                
                btn_label = "Validated" if is_verified else "✅ Validate Match"
                if st.button(btn_label, key=f"btn_{selected_ifc}_{c_id}", type="primary" if is_verified else "secondary", disabled=is_verified):
                    st.session_state.manual_mappings[selected_ifc] = match
                    
                    output_dir = os.path.dirname(MATCH_MATRIX)
                    if output_dir and not os.path.exists(output_dir): 
                        os.makedirs(output_dir, exist_ok=True)
                    
                    with open(MATCH_MATRIX, "w", encoding="utf-8") as f:
                        json.dump(st.session_state.manual_mappings, f, indent=2)
                    
                    st.toast(f"Saved: {c_id}", icon="✨")
                    st.rerun()