"""
Typology Matching Agents.

TypologyIndexer  — reads element_to_typology.json and encodes ONE representative
                   element name per unique typology into a persistent Chroma DB
                   (stored in IFCintegration/typologies_db/).
                   The Chroma document ID is keyed on the typology name so that
                   upserts are idempotent and each typology appears exactly once.

TypologyRetriever — given one or more element names that have NO typology, finds
                    the closest indexed typologies and returns them ranked by
                    embedding similarity.
"""

import shutil
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow imports from RAGTest root regardless of where this file is invoked from.
_RAGTEST_ROOT = Path(__file__).resolve().parents[2]
if str(_RAGTEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAGTEST_ROOT))

from src.RAG.vectorstore import VectorStoreManager  # noqa: E402

_DEFAULT_DB_PATH = str(Path(__file__).resolve().parent.parent / "typologies_db")
_COLLECTION = "ifc_element_typologies"
_MODEL = "intfloat/multilingual-e5-large"


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

class TypologyIndexer:
    """
    Builds and populates the typologies_db Chroma collection.

    One entry per unique (ifc_type, typology_name) pair — the document text
    is a representative element name so searches by element name remain
    semantically meaningful.  The Chroma ID is
    ``"{ifc_type}::{typology_name}"`` which guarantees uniqueness per typology.
    """

    def __init__(
        self,
        db_path: str = _DEFAULT_DB_PATH,
        model_name: str = _MODEL,
    ):
        self.db_path = db_path
        self.vs = VectorStoreManager(db_path, _COLLECTION, model_name)

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Delete and recreate the Chroma collection (wipes all entries)."""
        try:
            self.vs.client.delete_collection(_COLLECTION)
        except Exception:
            pass
        self.vs.collection = self.vs.client.get_or_create_collection(_COLLECTION)

    # ------------------------------------------------------------------
    def index_from_json(
        self,
        element_to_typology_path: str,
        rebuild: bool = True,
    ) -> Dict[str, int]:
        """
        Read element_to_typology.json and index ONE representative entry per
        unique typology per IFC type.

        Args:
            element_to_typology_path: path to the JSON file.
            rebuild: if True, wipe the collection first to ensure a clean state.

        Returns:
            {ifc_type: n_unique_typologies_indexed}
        """
        if rebuild:
            self.clear()

        with open(element_to_typology_path, "r", encoding="utf-8") as f:
            data: Dict[str, Dict[str, str]] = json.load(f)

        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, str]] = []
        counts: Dict[str, int] = {}

        for ifc_type, elements in data.items():
            seen_typologies: set = set()
            type_count = 0
            for element_name, typology_name in elements.items():
                if typology_name in seen_typologies:
                    continue                          # skip duplicate typology
                seen_typologies.add(typology_name)

                # ID keyed on typology so upserts are idempotent
                uid = f"{ifc_type}::{typology_name}"
                ids.append(uid)
                docs.append(element_name)            # representative element text
                metas.append({
                    "element_name": element_name,
                    "typology_name": typology_name,
                    "ifc_type": ifc_type,
                })
                type_count += 1
            counts[ifc_type] = type_count

        if ids:
            self.vs.upsert_materials(ids, docs, metas)

        return counts

    @property
    def size(self) -> int:
        return self.vs.collection.count()


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class TypologyRetriever:
    """
    Semantic retriever: given an element name with no known typology, finds
    the most similar element names in the DB and surfaces their typologies.
    """

    def __init__(
        self,
        db_path: str = _DEFAULT_DB_PATH,
        model_name: str = _MODEL,
    ):
        self.vs = VectorStoreManager(db_path, _COLLECTION, model_name)

    def retrieve(
        self,
        element_name: str,
        ifc_type: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Return the top_k closest indexed elements (and their typologies) for
        *element_name*.  Pass *ifc_type* to restrict the search to the same
        element class (e.g. "IfcWall").

        Each result dict:
          {element_name, typology_name, ifc_type, similarity_score}
        """
        query_vec = self.vs.get_embeddings([element_name], is_query=True)

        where_filter = {"ifc_type": ifc_type} if ifc_type else None

        # Chroma raises if n_results > collection size; guard against that.
        n_results = min(top_k, max(self.vs.collection.count(), 1))

        results = self.vs.collection.query(
            query_embeddings=query_vec,
            n_results=n_results,
            where=where_filter,
        )

        hits: List[Dict[str, Any]] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return hits

        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i] if "distances" in results else 0.0
            hits.append({
                "element_name": meta.get("element_name"),
                "typology_name": meta.get("typology_name"),
                "ifc_type": meta.get("ifc_type"),
                "similarity_score": round(1.0 - dist, 4),
            })

        return hits

    def batch_retrieve(
        self,
        element_names: List[str],
        ifc_type: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve closest typologies for every element name in *element_names*.
        Returns {element_name: [hits]}.
        """
        return {
            name: self.retrieve(name, ifc_type=ifc_type, top_k=top_k)
            for name in element_names
        }

    def retrieve_from_json(
        self,
        no_typology_path: str,
        top_k: int = 3,
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Load elements_without_typology.json and run retrieval for every entry.
        Returns {ifc_type: {element_name: [hits]}}.
        """
        with open(no_typology_path, "r", encoding="utf-8") as f:
            data: Dict[str, List[str]] = json.load(f)

        return {
            ifc_type: self.batch_retrieve(names, ifc_type=ifc_type, top_k=top_k)
            for ifc_type, names in data.items()
            if names
        }
