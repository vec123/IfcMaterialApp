from src.RAG.vectorstore import VectorStoreManager
from typing import Any, Dict, List, Optional


_TYPOLOGY_COLLECTION = "typology_v1"


class ConstructionSearchService:
    def __init__(self, vector_store: VectorStoreManager):
        self.vs = vector_store

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Performs vector similarity search."""
        query_vector = self.vs.get_embeddings([query], is_query=True)
        
        results = self.vs.collection.query(
            query_embeddings=query_vector,
            n_results=limit
        )

        formatted_results = []
        if not results or not results.get('ids') or len(results['ids'][0]) == 0:
            return []

        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "score": round(1 - results['distances'][0][i], 4) if 'distances' in results else 0,
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })
        return formatted_results

    def map_to_canonical_materials(self, material_query: str) -> List[str]:
        """
        Takes a raw material string and finds the closest 
        canonical material names in the database.
        """
        # Search for the specific material term
        hits = self.search(material_query, limit=2)
        
        # KEY FIX: Ensure we are pulling the correct metadata key 
        # (Check if your DB uses 'material_name', 'label', or 'type')
        names = []
        for h in hits:
            # Try common metadata keys for material names
            m_name = h["metadata"].get("material_name") or h["metadata"].get("name")
            if m_name:
                names.append(m_name)
        
        return list(dict.fromkeys(names))
    
    def map_params_materials(self, params: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
            in_params = params.copy()
            raw_materials = in_params.get("materials")

            if isinstance(raw_materials, str):
                material_list = [m.strip() for m in raw_materials.split(",") if m.strip()]
            elif isinstance(raw_materials, list):
                material_list = [str(m).strip() for m in raw_materials if m]
            else:
                return in_params

            updated_materials = []
            for mat in material_list:
                # 1. Exact Match Attempt
                try:
                    exact_check = self.vs.collection.get(
                        where={"material": mat},
                        limit=1
                    )
                except Exception:
                    exact_check = None

                # Safe extraction from exact search structure
                if exact_check and exact_check.get('ids') and len(exact_check['ids']) > 0:
                    meta = exact_check['metadatas'][0] if exact_check['metadatas'] else {}
                    # Handle possible list-of-lists or single list structural variations safely
                    if isinstance(meta, list) and len(meta) > 0:
                        meta = meta[0]
                    
                    canonical_name = meta.get("material") or meta.get("material_name") or meta.get("name")
                    if canonical_name:
                        updated_materials.append(canonical_name)
                        continue  # Found exact, skip semantic vector fallback

                # 2. Fallback to Vector Expansion
                hits = self.search(mat, limit=top_k)
                if hits:
                    for h in hits:
                        h_meta = h.get("metadata") or {}
                        # Cascade lookup to safely catch any naming schema variation
                        c_name = h_meta.get("material") or h_meta.get("material_name") or h_meta.get("name") or h.get("id")
                        if c_name:
                            updated_materials.append(c_name)
                else:
                    # Absolute fallback: retain the user's raw term if nothing else matches
                    updated_materials.append(mat)

            # Deduplicate names while maintaining deterministic order
            in_params["materials"] = list(dict.fromkeys([m for m in updated_materials if m]))
            return in_params

    # ─────────────────────────────────────────────────────────────────
    # Typology index (collection: typology_v1)
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _typology_doc(construction: Dict[str, Any]) -> str:
        """Build the embeddable document string for a typology."""
        category  = construction.get("category", "")
        materials = " ".join(
            layer["material"]
            for layer in construction.get("layers", [])
            if layer.get("material")
        )
        return f"{category}: {materials}".strip()

    def upsert_typology(self, typology_id: str, construction: Dict[str, Any]) -> None:
        """Index (or re-index) one typology in the typology_v1 collection."""
        doc = self._typology_doc(construction)
        self.vs.upsert_to_collection(
            collection_name=_TYPOLOGY_COLLECTION,
            ids=[typology_id],
            documents=[doc],
            metadatas=[{
                "typology_id": typology_id,
                "category":    construction.get("category", ""),
                "materials":   " ".join(
                    l["material"] for l in construction.get("layers", []) if l.get("material")
                ),
            }],
        )

    def upsert_typologies_bulk(self, constructions_db: Dict[str, Any]) -> int:
        """Index all typologies in bulk. Returns count indexed."""
        if not constructions_db:
            return 0
        ids, docs, metas = [], [], []
        for tid, construction in constructions_db.items():
            ids.append(tid)
            docs.append(self._typology_doc(construction))
            metas.append({
                "typology_id": tid,
                "category":    construction.get("category", ""),
                "materials":   " ".join(
                    l["material"] for l in construction.get("layers", []) if l.get("material")
                ),
            })
        self.vs.upsert_to_collection(
            collection_name=_TYPOLOGY_COLLECTION,
            ids=ids,
            documents=docs,
            metadatas=metas,
        )
        return len(ids)

    def search_typologies(
        self, query: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search typology_v1 by a free-text query (category name, material list, or both).
        Returns ranked hits with typology_id, score, category, materials.
        """
        raw = self.vs.query_collection(_TYPOLOGY_COLLECTION, query, n_results=limit)
        if not raw or not raw.get("ids") or not raw["ids"][0]:
            return []
        results = []
        for i in range(len(raw["ids"][0])):
            dist = raw["distances"][0][i] if "distances" in raw else 0
            results.append({
                "typology_id": raw["ids"][0][i],
                "score":       round(1 - dist, 4),
                "content":     raw["documents"][0][i],
                "metadata":    raw["metadatas"][0][i],
            })
        return results

    def typology_index_size(self) -> int:
        """Return how many typologies are currently indexed."""
        return self.vs.get_collection(_TYPOLOGY_COLLECTION).count()