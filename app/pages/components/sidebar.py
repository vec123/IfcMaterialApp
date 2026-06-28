import streamlit as st
import os
import shutil
import gc  # Essential for releasing file handles

def render_shared_sidebar(init_state_fn=None) -> None:
    st.sidebar.markdown("## EnergIA BIM-BEPS")
    st.sidebar.divider()
    st.sidebar.markdown("### 🛠️ Data Management")
    
    cfg = st.session_state.get("cfg")
    root = st.session_state.get("project_root", os.getcwd())

    if cfg and "paths" in cfg:
        def get_abs(key):
            p = cfg['paths'].get(key)
            if not p: return None
            return p if os.path.isabs(p) else os.path.abspath(os.path.join(root, p))

        management_map = {
            "Delete IFC Data": [
                get_abs('ifc_extracted_dir'),
                get_abs('ifc_extracted_file'),
                get_abs('ifc_extracted_element_to_typology'),
                get_abs('ifc_extracted_element_missing_typology'),
                get_abs('ifc_extracted_typology_layers'),
                get_abs('ifc_extracted_unique_materials'),
                get_abs('ifc_extracted_LLM_parsing'),
                get_abs('ifc_graph_edges_path'),
                get_abs('tmp_match_matrix'),
                get_abs('match_matrix'),
            ],
            "Delete Materials Data": [
                get_abs('materials_file'),
                get_abs('unstructured_materials_file'),
                get_abs('vector_db_path'),
                get_abs('graph_edges_path'),
                get_abs('tmp_match_matrix'),
                get_abs('match_matrix'),
            ]
        }

        for label, paths in management_map.items():
            valid_paths = [p for p in paths if p and os.path.exists(p)]
            
            if st.sidebar.button(label, key=f"btn_del_{label.replace(' ', '_')}", use_container_width=True):
                try:
                    # 1. CRITICAL: Clear memory references to unlock files
                    if "vector_store" in st.session_state:
                        del st.session_state.vector_store
                    gc.collect() 
                    
                    for path in valid_paths:
                        norm_path = os.path.normpath(path)
                        
                        if os.path.isdir(norm_path):
                            try:
                                # Try to remove the directory
                                shutil.rmtree(norm_path)
                            except OSError:
                                # If it fails, it's a lock. We clear the contents instead.
                                print(f"DEBUG: Directory locked. Clearing contents only: {norm_path}")
                                for filename in os.listdir(norm_path):
                                    file_path = os.path.join(norm_path, filename)
                                    try:
                                        if os.path.isfile(file_path) or os.path.islink(file_path):
                                            os.unlink(file_path)
                                        elif os.path.isdir(file_path):
                                            shutil.rmtree(file_path)
                                    except Exception as e:
                                        print(f"Failed to delete {file_path}. Reason: {e}")
                                        
                            os.makedirs(norm_path, exist_ok=True)
                        else:
                            os.remove(norm_path)
                    
                    st.sidebar.success(f"Cleared {label}")
                    st.rerun()
                except PermissionError:
                    st.sidebar.error("Database locked! Close any active DB connections or browser tabs using this data and try again.")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
            
    else:
        st.sidebar.warning("Config not loaded.")

    st.sidebar.caption("Data: `constructions.json` · `materials.json`")