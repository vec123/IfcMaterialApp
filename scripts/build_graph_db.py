import os
import sys
import json
import requests
from pathlib import Path

# Setup paths relative to script location
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.RAG.vectorstore import VectorStoreManager
from src.utils.load_conf import build_cfg  # Swapped to your custom configuration engine
from src.agents.helpers.extension_agent import MaterialExtensionAgent 
from app.loader.loader import clear_cuda_runtime


def build_separated_graph_rag(cfg: dict) -> dict:
    """
    Vectorize materials (enriched via LLM Agent) and/or typologies into the vector DB.
    
    Args:
        cfg (dict): The integrated configuration dictionary loaded from build_cfg().
    """
    print("🚀 Initializing Isolated Multi-Collection Vector Stores using structural cfg...")
    
    # ── Path Resolutions (Now using clean keys straight from the updated cfg['paths']) ──
    print(cfg)
    db_path          = cfg['paths']['vector_db_path']
    graph_edges_path = cfg['paths']['graph_edges_path']
    print(" cfg['paths']: ",  cfg)
    materials_path   = cfg['paths']['materials_file']
    typologies_path  = cfg['paths']['typologies_file']

    # ── Database Model and Collection Settings ──
    model_name            = cfg['vector_store']['embedding_model']
    material_collection   = cfg['vector_store']['material_collection_name']
    typology_collection   = cfg['vector_store']['typology_collection_name']

    results = {"materials_count": 0, "typologies_count": 0, "edges_count": 0}
    graph_edges = []

    # ──── 1. MATERIALS (With Agent Extension Pass) ────────────────────────────
    if materials_path and os.path.exists(materials_path):
        print("📦 Processing & Extending Materials via LLM Agent...")
        clear_cuda_runtime()
        
        # Initialize extension agent using the config parameters directly
        extension_agent = MaterialExtensionAgent(cfg)
        
        with open(materials_path, "r", encoding="utf-8") as f:
            materials_db = json.load(f)

        mat_ids, mat_docs, mat_metas = [], [], []
        agent_inspection_log = {}

        for mat_name, props in materials_db.items():
            print(f"🌍 Agent processing & extending: '{mat_name}'...")
            
            # Request translation and expansion variants from the LLM agent
            extend_result = extension_agent.extract(user_input=mat_name)
            
            # Log the transform mapping for inspection data integrity
            agent_inspection_log[mat_name] = {
                "input": {"raw_mat_name": mat_name, "properties": props},
                "output": extend_result
            }
            
            clean_mat_name = extend_result.get("name") or mat_name
            variations = extend_result.get("variations") or [clean_mat_name]
            english_description = extend_result.get("english_description") or clean_mat_name
            spanish_description = extend_result.get("spanish_description") or clean_mat_name

            # Structural metadata node template
            base_meta = props.copy()
            base_meta.update({
                "material_id": mat_name,  
                "clean_name": clean_mat_name,
            })

            # Create cross-language structural sub-nodes for higher retrieval accuracy
            # Node A: English Context
            mat_ids.append(f"{mat_name}##desc_en")
            mat_docs.append(f"Material Description (EN): {english_description}. Density: {props.get('density')} kg/m³.")
            mat_metas.append(base_meta)

            # Node B: Spanish Context
            mat_ids.append(f"{mat_name}##desc_es")
            mat_docs.append(f"Descripción del Material (ES): {spanish_description}. Conductividad Térmica: {props.get('thermal_conductivity')} W/mK.")
            mat_metas.append(base_meta)
            
            # Node Group C: Alias/Synonym Variant Nodes
            for idx, variant in enumerate(variations):
                mat_ids.append(f"{mat_name}##var_{idx}")
                mat_docs.append(f"Material Variant Reference Keyword: {variant}")
                mat_metas.append(base_meta)

        # Export mapping log to the same directory as materials
        inspection_output_path = os.path.join(os.path.dirname(materials_path), "agent_inspection.json")
        with open(inspection_output_path, "w", encoding="utf-8") as f:
            json.dump(agent_inspection_log, f, indent=2, ensure_ascii=False)
            
        del extension_agent

        # EVICT OLLAMA VRAM IMMEDIATELY IF RUNNING LOCAL MODELS
        if cfg['agent'].get('agent_provider') == 'ollama':
            try:
                ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
                ollama_model = cfg['agent'].get('agent_model', '')
                if ollama_model:
                    print(f"🧹 Directing Ollama daemon to unload '{ollama_model}' from memory...")
                    requests.post(f"{ollama_url}/api/generate", json={"model": ollama_model, "keep_alive": 0})
            except Exception as e:
                print(f"⚠️ Non-critical notice: Could not auto-evict Ollama cache weights: {e}")

        clear_cuda_runtime()

        # Commit Material Sub-nodes to DB
        vs_material = VectorStoreManager(
            db_path=db_path,
            collection_name=material_collection,
            model_name=model_name,
        )
        vs_material.upsert_materials(mat_ids, mat_docs, mat_metas)
        results["materials_count"] = len(materials_db)
        print(f"✅ Loaded {len(mat_ids)} localized sub-nodes into '{material_collection}'.")

    # ──── 2. TYPOLOGIES ───────────────────────────────────────────────────────
    if typologies_path and os.path.exists(typologies_path):
        print("🏗️  Vectorizing Typologies Collection...")
        vs_typology = VectorStoreManager(
            db_path=db_path,
            collection_name=typology_collection,
            model_name=model_name,
        )
        with open(typologies_path, "r", encoding="utf-8") as f:
            typologies_db = json.load(f)

        typo_ids, typo_docs, typo_metas = [], [], []
        for typo_id, info in typologies_db.items():
            doc_str = (
                f"Typology: {typo_id}. "
                f"Category: {info.get('category')}. "
                f"CTE Code: {info.get('cte_code')}."
            )
            typo_ids.append(typo_id)
            typo_docs.append(doc_str)
            typo_metas.append({
                "category":  info.get("category"),
                "cte_code":  info.get("cte_code"),
                "u_formula": info.get("u_formula"),
            })
            for index, layer in enumerate(info.get("layers", [])):
                graph_edges.append({
                    "typology_id":    typo_id,
                    "material_id":    layer.get("material"),
                    "layer_position": index,
                    "thickness":      layer.get("thickness"),
                })

        vs_typology.upsert_to_collection(
            collection_name=typology_collection,
            ids=typo_ids,
            documents=typo_docs,
            metadatas=typo_metas,
        )
        results["typologies_count"] = len(typo_ids)
        print(f"✅ Loaded {len(typo_ids)} typologies into '{typology_collection}'.")

    # ──── 3. GRAPH EDGES ──────────────────────────────────────────────────────
    if results["materials_count"] > 0 and results["typologies_count"] > 0 and graph_edges_path:
        edges_dir = os.path.dirname(graph_edges_path)
        if edges_dir:
            os.makedirs(edges_dir, exist_ok=True)
        with open(graph_edges_path, "w", encoding="utf-8") as f:
            json.dump(graph_edges, f, indent=2, ensure_ascii=False)
        results["edges_count"] = len(graph_edges)
        print(f"🔗 Successfully established {len(graph_edges)} structural graph relationships.")

    clear_cuda_runtime()
    return results


if __name__ == "__main__":
    # Load dynamic unified configuration via single cleanly abstract function entry point
    runtime_cfg = build_cfg()
    # Run pipeline seamlessly using only the unified configuration payload
    build_separated_graph_rag(cfg=runtime_cfg)