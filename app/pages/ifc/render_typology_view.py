# scripts/app/pages/domain_views.py
# scripts/app/pages/domain_views.py

import os
import re
import sys
from typing import *
import streamlit as st

# ── 1. PROJECT PATH RESOLUTION ─────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from src.agents.ifc.query_typology import TypologyIntentAgent, TypologyFunctionInputAgent, TypologyPipelineOrchestrator


def render_typology_space(query: str, cfg: dict, typo_search_service: Any, typo_database: dict, edges_db: list, top_k: int):
    """Handles Multi-Intent Typology classification and progressive asset attrition rules."""
    if not query.strip():
        return

    with st.spinner("Executing Multi-Intent Classification & Attrition Routing Pipeline..."):
        # 1. Initialize structural agent grouping
        agents = {
            "intent": TypologyIntentAgent(config=cfg),
            "extractor": TypologyFunctionInputAgent(config=cfg)
        }
        
        # 2. Fire the Pipeline Orchestration Engine
        orchestrator = TypologyPipelineOrchestrator(
            agents=agents, 
            search_service=typo_search_service, 
            typology_database=typo_database
        )
        active_intents, accumulated_params, pipeline_results = orchestrator.run(query, vector_limit=top_k)
        
    # Display processing insights on screen
    st.info(f"⚙️ **Active Multi-Intent Chain Execution Steps:** `{', '.join(active_intents)}`")
    with st.expander("🔍 Inspection: Extracted Pipeline Parameters", expanded=False):
        st.json(accumulated_params)
        
    if not pipeline_results:
        st.warning("No structural typologies matched your compound query constraints.")
        return

    display_slices = pipeline_results[:top_k]
    st.success(f"Matched **{len(pipeline_results)}** typologies. Rendering top {len(display_slices)} results:")
    
    for typo_data in display_slices:
        typo_id = typo_data["id"]
        thickness = typo_data.get("total_thickness_m", 0.0)
        
        with st.expander(f"🏗️ Typology Layer Set: **{typo_id}** (Total Thickness: {thickness:.3f}m)", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**IFC Base Class Entity:** `{typo_data.get('ifc_element_class')}`")
                st.markdown("**Composition Stack Layers:**")
                layers_rows = []
                for layer in typo_data.get("layers", []):
                    layers_rows.append({
                        "Pos": layer.get("layer_position"),
                        "Material ID": layer.get("material_id"),
                        "Thickness (m)": f"{layer.get('thickness_m'):.4f}" if layer.get('thickness_m') else "N/A",
                        "Layer Name": layer.get("layer_assigned_name")
                    })
                if layers_rows: 
                    st.table(layers_rows)
                else: 
                    st.caption("No embedded layer records extracted.")

            with col2:
                linked_elements = sorted(list({e["element_instance_id"] for e in edges_db if e.get("mapped_typology_id") == typo_id}))
                st.markdown(f"**Instantiated Model Elements Using This Layer Set ({len(linked_elements)}):**")
                if linked_elements:
                    st.dataframe({"Global Object Instance Labels ID": linked_elements}, width="stretch", hide_index=True)
                else:
                    st.caption("No discrete object elements inherit this typology set.")

