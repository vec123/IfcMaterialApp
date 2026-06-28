import os
import json
from typing import List, Dict, Any
from src.RAG.vectorstore import VectorStoreManager

class GraphNavigationService:
    def __init__(self, vs_typology: VectorStoreManager, graph_edges_path: str):
        """
        Handles relational graph traversals over multi-collection vector structures.
        
        :param vs_typology: VectorStoreManager bound to the typologies collection.
        :param graph_edges_path: System path pointing to 'graph_relationships.json'.
        """
        self.vs_typology = vs_typology
        self.graph_edges_path = graph_edges_path
        self.graph_edges = self._load_edges()

    def _load_edges(self) -> List[Dict[str, Any]]:
        """Safely loads graph edges into memory."""
        if os.path.exists(self.graph_edges_path):
            with open(self.graph_edges_path, "r", encoding="utf-8") as f:
                return json.load(f)
        print(f"⚠️ Warning: Graph relationships file missing at {self.graph_edges_path}")
        return []

    def get_connected_typologies(self, canonical_material_id: str) -> List[Dict[str, Any]]:
        """
        Finds all construction typologies that use the given material,
        and hydrates them with text data from the Vector DB.
        """
        # Step 1: Filter edges across the structural matrix
        linked_edges = [
            edge for edge in self.graph_edges 
            if edge["material_id"] == canonical_material_id
        ]
        
        connected_results = []
        typology_collection = self.vs_typology.collection

        # Step 2: Hydrate relational matches with target vector properties
        for edge in linked_edges:
            typology_id = edge["typology_id"]
            
            try:
                # Fetch data node from Chroma
                typo_data = typology_collection.get(ids=[typology_id])
                context = typo_data["documents"][0] if typo_data["documents"] else ""
            except Exception:
                context = "Error pulling document context from storage matrix."

            connected_results.append({
                "typology_id": typology_id,
                "layer_position": edge["layer_position"],
                "thickness": edge["thickness"],
                "typology_context": context
            })
            
        return connected_results