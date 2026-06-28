
import sys
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_cfg(config_relative_path: str = "conf/config.yaml") -> dict:
    load_dotenv()
    
    # 1. Resolve project root and locate config.yaml
    project_root = next((p for p in Path(__file__).resolve().parents if (p / "conf").exists() or (p / ".env").exists()), Path(__file__).resolve().parent)
    cfg_path = project_root / config_relative_path

    # 2. Load the base config layout
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # 3. Adapt all configuration paths based on PROJECT_PATH from .env if present
    base_project_path = os.environ.get("PROJECT_PATH")
    if base_project_path and "paths" in cfg:
        for path_key, relative_path in cfg["paths"].items():
            if relative_path:  # Ensure the path value is not empty
                cfg["paths"][path_key] = os.path.join(base_project_path, relative_path)

    # 4. Inject API keys dynamically based on your YAML providers
    if cfg.get("agent", {}).get("agent_provider") == "google" or cfg.get("vector_store", {}).get("embedding_provider") == "google":
        key  = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        cfg.setdefault("agent", {})["api_key"] = key
        cfg.setdefault("vector_store", {})["api_key"] = key
        
    elif cfg.get("agent", {}).get("agent_provider") == "groq":
        cfg.setdefault("agent", {})["api_key"] = os.environ.get("GROQ_API_KEY")

    return cfg