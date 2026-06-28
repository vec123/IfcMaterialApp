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
from src.agents.ifc.regex_agent import RegexProposalAgent
import src.agents.ifc.regex_agent as ga


def render_physical_elements(cfg: dict, elem_search_service: Any, edges_db: list, top_k: int, default_context: str) -> str:
    """Handles spatial element routing discovery via Semantic RAG searches or direct lookups."""
    st.markdown("### Locate Physical Model Instances")
    instance_access_mode = st.radio("Discovery Strategy:", ["Semantic RAG Search", "Direct Dropdown Identifier Selector"], horizontal=True)
    selected_element_id = None

    if instance_access_mode == "Semantic RAG Search":
        with st.expander("📝 Edit Agent Architectural Pattern Context", expanded=False):
            st.caption("The agent parses this natural language context document to understand how to formulate its cleaning regex:")
            active_context = st.text_area("Active Pattern Context Guidelines (.md)", value=default_context, height=250, key="architect_context_input")
        
        elem_query = st.text_input("Describe the element or paste a partial string:", placeholder="e.g. Solera, MurPantalla", key="elem_rag_q")
        use_grouping = st.toggle("Enable AI Archetype Grouping", value=True)

        if elem_query.strip():
            with st.spinner("Processing vector space query..."):
                if not use_grouping:
                    hits = elem_search_service.search(elem_query, limit=top_k)
                    if hits:
                        found_options = []
                        for h in hits:
                            h_id = getattr(h, "id", None) or h.get("id") if isinstance(h, dict) else h.id
                            score = getattr(h, "score", 0) or h.get("score", 0) if isinstance(h, dict) else h.score
                            found_options.append((h_id, score))
                        selected_item = st.radio("Select instance target to inspect:", options=found_options, format_func=lambda x: f"🎯 {x[0]} (Score: {x[1]:.4f})")
                        if selected_item: 
                            selected_element_id = selected_item[0]
                    else:
                        st.warning("No elements matched your search query.")
                else:
                    raw_hits = elem_search_service.search(elem_query, limit=60)
                    if not raw_hits:
                        st.warning("No elements found matching your description.")
                    else:
                        all_hit_ids = [getattr(h, "id", None) or h.get("id") if isinstance(h, dict) else h.id for h in raw_hits]
                        ga._ARCHITECT_CONTEXT = active_context
                        
                        regex_agent = RegexProposalAgent(config=cfg)
                        proposal = regex_agent.propose(all_hit_ids)
                        strip_pattern = proposal.get("pattern", r"_\d+_\d+$")
                        
                        st.info("🤖 **Active AI Pattern Grouping Diagnostics:**")
                        col_dia1, col_dia2 = st.columns(2)
                        with col_dia1:
                            st.markdown(f"**Calculated Python Regex:** `{strip_pattern}`")
                            st.markdown(f"**Agent Focus:** *{proposal.get('explanation')}*")
                        with col_dia2:
                            example_input = all_hit_ids[0]
                            example_output = re.sub(strip_pattern, "", example_input).strip()
                            st.markdown(f"**Example Input:** `{example_input}`")
                            st.markdown(f"**Resulting Group Key:** `{example_output}`")
                        
                        grouped_archetypes = {}
                        for h in raw_hits:
                            h_id = getattr(h, "id", None) or h.get("id") if isinstance(h, dict) else h.id
                            score = getattr(h, "score", 0) or h.get("score", 0) if isinstance(h, dict) else h.score
                            archetype_key = re.sub(strip_pattern, "", h_id).strip()
                            
                            if archetype_key not in grouped_archetypes:
                                grouped_archetypes[archetype_key] = {"top_score": score, "nodes": []}
                            grouped_archetypes[archetype_key]["nodes"].append((h_id, score))
                        
                        sorted_groups = sorted(grouped_archetypes.items(), key=lambda x: x[1]["top_score"], reverse=True)[:top_k]
                        st.success(f"Discovered **{len(sorted_groups)}** distinct construction archetypes:")
                        
                        for arch_key, arch_data in sorted_groups:
                            with st.expander(f"🧱 Archetype: **{arch_key}** (Top Score: {arch_data['top_score']:.4f})"):
                                selected_node = st.selectbox(
                                    "Pick an instance tracking ID to map:",
                                    options=arch_data["nodes"],
                                    format_func=lambda x: f"🎯 ID: {x[0]} (Score: {x[1]:.4f})",
                                    key=f"select_{arch_key}"
                                )
                                if selected_node: 
                                    selected_element_id = selected_node[0]
    else:
        all_elements = sorted(list({e["element_instance_id"] for e in edges_db}))
        if all_elements:
            selected_element_id = st.selectbox("Select model tracking ID:", options=all_elements, index=0)
        else:
            st.info("No elements tracked inside the relational edge database repository.")
            
    return selected_element_id