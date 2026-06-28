import json
import ollama
from tqdm import tqdm
from RAG.vectorstore import VectorStoreManager
from typing import Optional

class MaterialIngestor:
    def __init__(self, vector_store: VectorStoreManager, translator_model: Optional[str] = None):
        self.vs = vector_store
        self.translator = translator_model

    def translate(self, text: str) -> str:
        if not self.translator:
            return text
        try:
            response = ollama.generate(
                model=self.translator,
                prompt=f"Translate to English and Catalan!. Return ONLY the translations: {text}",
                options={"temperature": 0}
            )
            return response['response'].strip()
        except Exception:
            return text

    def process_json(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        ids, docs, metas = [], [], []
        
        # Deduplicate and aggregate
        unique_materials = {}
        for const_id, info in data.items():
            for layer in info.get("layers", []):
                name = layer.get("material", "Unknown").strip()
                if name not in unique_materials:
                    unique_materials[name] = {
                        "density": layer.get("density", 0.0),
                        "tc": layer.get("thermal_conductivity", 0.0),
                        "source": const_id
                    }

        for es_name, props in tqdm(unique_materials.items(), desc="Ingesting"):
            en_name = self.translate(es_name)
            doc_string = f"{es_name} ({en_name}). Density: {props['density']}, TC: {props['tc']}"
            
            ids.append(es_name)
            docs.append(doc_string)
            metas.append({
                "density": props['density'],
                "tc": props['tc'],
                "source": props['source'],
                "material": es_name,
                "material_translations": en_name
            })

        self.vs.upsert_materials(ids, docs, metas)