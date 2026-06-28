"""
scripts/app/loader/loader.py
Shared infrastructure cache for EnergIA BIM-BEPS platform.
Supports both active Streamlit runtimes and headless CLI fallback contexts.
"""

import os
import sys
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer


# 1. Path Fix: Safely navigate 3 levels up to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.RAG.vectorstore import VectorStoreManager
from src.utils.load_conf import load_config

# Agent infrastructure core pipelines
from src.agents.query.query_typologies.agents import IntentAgent, FunctionInputAgent, CleaningAgent, StructureResponseAgent
from src.agents.add_mats.agents import (
    AddMaterialIntentAgent, MaterialExtractionAgent, 
    MaterialNameValidatorAgent, MaterialPropertiesValidatorAgent,
    MaterialDuplicateCheckAgent, MaterialPersistenceAgent
)
from src.agents.add_typologies.agents import AddTypologyIntentAgent, TypologyExtractionAgent
from src.utils.load_conf import build_cfg
from src.RAG.searchengine import ConstructionSearchService


# Global private memory caches for non-Streamlit (headless CLI script) executions
_CLI_CACHE = {}

def _is_streamlit() -> bool:
    """Detects if the code is executing inside a live Streamlit framework instance."""
    try:
        return st.runtime.exists()
    except (ImportError, AttributeError):
        return False


# ── SHARED EMBEDDING ENGINE ENCODER ──────────────────────────────────────────
@st.cache_resource
def _get_encoder_raw(model_name: str) -> SentenceTransformer:
    """Instantiates raw SentenceTransformer weights onto the optimal processing target."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Loading SentenceTransformer model onto [{device}]...")
    return SentenceTransformer(model_name, device=device)

@st.cache_resource
def get_shared_embedding_encoder(model_name: str) -> SentenceTransformer:
    """Guarantees exactly ONE instance of heavy embedding weights is ever allocated."""
    if _is_streamlit():
        cached_func = st.cache_resource(_get_encoder_raw)
        return cached_func(model_name)
    
    # CLI / Script Fallback Cache
    if model_name not in _CLI_CACHE:
        _CLI_CACHE[model_name] = _get_encoder_raw(model_name)
    return _CLI_CACHE[model_name]

@st.cache_resource
def get_vector_store(db_path=None, collection_name=None, model_name=None ):
    """Retrieves a shared vector store wrapper with an injected shared encoder."""

    shared_encoder = get_shared_embedding_encoder(model_name)
    
    return VectorStoreManager(
        db_path=db_path,
        collection_name=collection_name,
        model_name=model_name,
        shared_encoder=shared_encoder  # Protects GPU VRAM from multi-load duplication
    )

@st.cache_resource
def get_app_config():
    return build_cfg()

@st.cache_resource
def get_llm_agents(cfg):
    """Caches all LLM extraction and validation sub-agents globally across memory pages."""
    #cfg = load_config(os.path.join(project_root, "conf/config.yaml"))
    return {
        "query_intent":    IntentAgent(cfg),
        "query_extractor": FunctionInputAgent(cfg),
        "cleaner":         CleaningAgent(cfg),
        "structurer":      StructureResponseAgent(cfg),
        "mat_intent":      AddMaterialIntentAgent(cfg),
        "mat_extractor":   MaterialExtractionAgent(cfg),
        "typo_intent":     AddTypologyIntentAgent(cfg),
        "typo_extractor":  TypologyExtractionAgent(cfg),
        "name_validator":  MaterialNameValidatorAgent(cfg),
        "prop_validator":  MaterialPropertiesValidatorAgent(cfg),
        "material_duplicates": MaterialDuplicateCheckAgent(cfg['paths'].get('materials_file')),
        "material_persistence": MaterialPersistenceAgent(cfg['paths'].get('materials_file'))
    }


@st.cache_resource
def get_search_services(db_path, collection_name, model_name, _cfg):
    vs_manager = get_vector_store(
        db_path=db_path,
        collection_name=collection_name,
        model_name=model_name,
    )
    search_service = ConstructionSearchService(vector_store=vs_manager)
    extraction_agent = MaterialExtractionAgent(_cfg)

    return search_service, extraction_agent, collection_name, model_name

def get_ifc_RAG_search_services(cfg):
    return get_search_services(
        db_path=cfg["paths"]["ifc_vector_db_path"],
        collection_name=cfg["vector_store"]["material_collection_name"],
        model_name=cfg["vector_store"]["embedding_model"],
        _cfg=cfg,
    )

def get_material_RAG_search_services(cfg):
    return get_search_services(
        db_path=cfg["paths"]["vector_db_path"],
        collection_name=cfg["vector_store"]["material_collection_name"],
        model_name=cfg["vector_store"]["embedding_model"],
        _cfg=cfg,
    )

def clear_cuda_runtime():
    """Forces unmanaged tensor blocks out of CUDA memory allocations."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

