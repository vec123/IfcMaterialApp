import os
import json
import sys
import requests
from pathlib import Path

# Setup project root relative paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.RAG.vectorstore import VectorStoreManager
from src.utils.load_conf import build_cfg  # Custom dynamic configuration compiler
from src.agents.helpers.extension_agent import MaterialExtensionAgent 
from app.loader.loader import clear_cuda_runtime
from app.loader.loader import get_shared_embedding_encoder
    

def build_ifc_graph_rag(cfg: dict):
    """
    Vectorize unique IFC materials (enriched via LLM), typologies, and element 
    instances into distinct vector store collections.
    
    Args:
        cfg (dict): Fully integrated configuration object from build_cfg()
    """
    print("🚀 Initializing Dynamic IFC Multi-Collection Vector Database Store...")
    clear_cuda_runtime() 
    
    # ── Path Resolutions (Extracted from cfg['paths'] and cfg['ingestion']) ──
    ifc_extracted_dir     = cfg['paths'].get('ifc_extracted_dir', 'data_in/ifc_extracted')
    ifc_vector_db_path    = cfg['paths']['ifc_vector_db_path']
    ifc_graph_edges_path  = cfg['paths']['ifc_graph_edges_path']

    path_elem_to_typo = os.path.join(ifc_extracted_dir, "element_to_typology.json")
    path_typo_layers  = os.path.join(ifc_extracted_dir, "typologies_as_layers.json")
    path_unique_mats  = os.path.join(ifc_extracted_dir, "unique_materials.json")

    for p in [path_elem_to_typo, path_typo_layers, path_unique_mats]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required extraction component context: '{p}'.")

    # ── Database Model and Collection Settings ──
    model_name             = cfg['vector_store']['embedding_model']
    material_collection    = cfg['vector_store'].get('material_collection_name', 'materials_v1')
    typology_collection    = cfg['vector_store'].get('typology_collection_name', 'typologies_v1')
    ifc_element_collection = cfg['vector_store'].get('ifc_element_collection_name', 'ifc_elements_v1')

    # ── STAGE 1: AGENT GENERATION PASS ────────────────────────────────────────
    extension_agent = MaterialExtensionAgent(cfg)

    print(f"🧪 Processing Unique Materials via LLM...")
    with open(path_unique_mats, "r", encoding="utf-8") as f:
        unique_mats_db = json.load(f)

    mat_ids, mat_docs, mat_metas = [], [], []
    all_seen_materials = set()
    agent_inspection_log = {}

    for ifc_type, materials_list in unique_mats_db.items():
        for raw_mat_name in materials_list:
            if not raw_mat_name or raw_mat_name in all_seen_materials:
                continue
            all_seen_materials.add(raw_mat_name)
            
            print(f"🌍 Agent processing & extending: '{raw_mat_name}'...")
            extend_result = extension_agent.extract(user_input=raw_mat_name)
            
            agent_inspection_log[raw_mat_name] = {
                "input": {
                    "raw_mat_name": raw_mat_name,
                    "origin_ifc_type": ifc_type
                },
                "output": extend_result
            }
            
            clean_mat_name = extend_result.get("name") or raw_mat_name
            variations = extend_result.get("variations") or [clean_mat_name]
            english_description = extend_result.get("english_description") or clean_mat_name
            spanish_description = extend_result.get("spanish_description") or clean_mat_name

            base_meta = {
                "material_id": raw_mat_name,  
                "clean_name": clean_mat_name,
                "origin_ifc_type": ifc_type
            }

            # 1. English context mapping
            mat_ids.append(f"{raw_mat_name}##desc_en")
            mat_docs.append(f"Material Description (EN): {english_description}")
            mat_metas.append(base_meta)

            # 2. Spanish context mapping
            mat_ids.append(f"{raw_mat_name}##desc_es")
            mat_docs.append(f"Descripción del Material (ES): {spanish_description}")
            mat_metas.append(base_meta)
            
            # 3. Variance references mapping
            for idx, variant in enumerate(variations):
                mat_ids.append(f"{raw_mat_name}##var_{idx}")
                mat_docs.append(f"Material Variant Reference: {variant}")
                mat_metas.append(base_meta)

    #inspection_output_path = os.path.join(ifc_extracted_dir, "agent_inspection.json")
    inspection_output_path = cfg["paths"]["ifc_extracted_LLM_parsing"]
    print(f"💾 Saving LLM agent inspection log to: {inspection_output_path}...")
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

    # ── STAGE 2: PARSE STRUCTURAL TYPOLOGIES ──────────────────────────────────
    print(f"🧱 Processing Structural Typology Assemblies...")
    with open(path_typo_layers, "r", encoding="utf-8") as f:
        typo_layers_db = json.load(f)

    typo_ids, typo_docs, typo_metas = [], [], []
    all_seen_typologies = set()

    for ifc_type, typologies in typo_layers_db.items():
        for typo_name, layers in typologies.items():
            if typo_name in all_seen_typologies:
                continue
            all_seen_typologies.add(typo_name)
            layer_materials = [l.get("material") or "Unknown" for l in layers]
            materials_str = ", ".join(layer_materials)
            
            doc_str = f"Typology Profile: {typo_name}. Class Type: {ifc_type}. Layer Recipe Materials order: [{materials_str}]."
            
            typo_ids.append(typo_name)
            typo_docs.append(doc_str)
            typo_metas.append({"ifc_element_class": ifc_type, "total_layers_count": len(layers)})

    # ── STAGE 3: MAP RELATIONAL GRAPH EDGES & PARSE PHYSICAL ELEMENTS ────────
    print("🔗 Linking explicit elements to database structural layers...")
    with open(path_elem_to_typo, "r", encoding="utf-8") as f:
        elem_to_typo_db = json.load(f)

    graph_edges = []
    elem_ids, elem_docs, elem_metas = [], [], []
    all_seen_elements = set()

    for ifc_type, instances in elem_to_typo_db.items():
        for element_id, typo_name in instances.items():
            target_layers = typo_layers_db.get(ifc_type, {}).get(typo_name, [])
            base_edge = {"element_instance_id": element_id, "ifc_element_class": ifc_type, "mapped_typology_id": typo_name, "total_layers": len(target_layers)}
            
            if target_layers:
                for idx, layer in enumerate(target_layers):
                    edge_entry = base_edge.copy()
                    edge_entry.update({
                        "layer_position": idx,
                        "material_id": layer.get("material") or "Unknown",
                        "thickness_m": layer.get("thickness_m"),
                        "layer_assigned_name": layer.get("layer_name")
                    })
                    graph_edges.append(edge_entry)
            else:
                base_edge.update({"layer_position": 0, "material_id": "Unknown", "thickness_m": None, "layer_assigned_name": None})
                graph_edges.append(base_edge)

            if element_id in all_seen_elements:
                continue
            all_seen_elements.add(element_id)
            doc_str = f"Physical Element Instance ID: {element_id}. Category Classification: {ifc_type}. Parent Structural Typology Assembly: {typo_name}."
            elem_ids.append(element_id)
            elem_docs.append(doc_str)
            elem_metas.append({"ifc_element_class": ifc_type, "mapped_typology_id": typo_name})

    # ── STAGE 4: VECTORIZATION ENTRY ENGINE POINT ────────────────────────────
    print(f"🎯 Loading Shared Embedding Engine weights ONCE: {model_name}...")
  
    try:
        shared_encoder = get_shared_embedding_encoder(model_name)
    except Exception as oom_e:
        if "CUDA out of memory" in str(oom_e):
            print("⚠️ GPU remains saturated! Falling back to CPU for embedding generation step...")
            from sentence_transformers import SentenceTransformer
            shared_encoder = SentenceTransformer(model_name, device="cpu")
        else:
            raise oom_e

    vs_manager = VectorStoreManager(
        db_path=ifc_vector_db_path, 
        collection_name=material_collection, 
        model_name=model_name,
        shared_encoder=shared_encoder
    )

    # 1. Commit Materials Collection
    if mat_ids:
        print(f"📥 Writing to '{material_collection}'...")
        vs_manager.upsert_materials(mat_ids, mat_docs, mat_metas)
        print(f"✅ Successfully Embedded {len(mat_ids)} cross-language nodes.")

    # 2. Commit Typologies Collection
    if typo_ids:
        print(f"📥 Writing to '{typology_collection}'...")
        vs_manager.collection_name = typology_collection
        vs_manager.collection = vs_manager.client.get_or_create_collection(name=typology_collection)
        vs_manager.upsert_to_collection(collection_name=typology_collection, ids=typo_ids, documents=typo_docs, metadatas=typo_metas)
        print(f"✅ Embedded {len(typo_ids)} unique compound typologies.")

    # 3. Commit Elements Collection
    if elem_ids:
        print(f"📥 Writing to '{ifc_element_collection}'...")
        vs_manager.collection_name = ifc_element_collection
        vs_manager.collection = vs_manager.client.get_or_create_collection(name=ifc_element_collection)
        vs_manager.upsert_to_collection(collection_name=ifc_element_collection, ids=elem_ids, documents=elem_docs, metadatas=elem_metas)
        print(f"✅ Embedded {len(elem_ids)} physical elements.")

    # ── STAGE 5: RECORD RELATIONAL ARTIFACTS TO DISK ──────────────────────────
    os.makedirs(os.path.dirname(ifc_graph_edges_path), exist_ok=True)
    with open(ifc_graph_edges_path, "w", encoding="utf-8") as f:
        json.dump(graph_edges, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Successfully written local graph dependency network map: {ifc_graph_edges_path}")
    
    del vs_manager
    clear_cuda_runtime()


if __name__ == "__main__":
    # Compile unified dict object entirely from build_cfg() wrapper logic
    runtime_cfg = build_cfg()
    
    # Run pipeline cleanly using only the configuration structure dictionary
    build_ifc_graph_rag(cfg=runtime_cfg)