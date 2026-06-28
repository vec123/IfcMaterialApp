import torch
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any, Optional,Union


import torch
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any, Optional, Union

    
class VectorStoreManager:
    def __init__(
        self, 
        db_path: str, 
        collection_name: str, 
        model_name: str = 'intfloat/multilingual-e5-large',
        shared_encoder=None
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if shared_encoder is not None:
            self.model = shared_encoder
        else:
            self.model = SentenceTransformer(model_name, device=self.device)

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)


    def get_embeddings(self, texts: Union[str, List[str]], is_query: bool = False) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        prefix = "query: " if is_query else "passage: "
        processed_texts = [f"{prefix}{t}" for t in texts]
        return self.model.encode(processed_texts, batch_size=32).tolist()
            
    def get_collection(self, name: str):
        """Return (or create) any named collection in the same persistent DB."""
        return self.client.get_or_create_collection(name=name)

    def upsert_materials(self, ids: List[str], documents: List[str], metadatas: List[Dict]):
        embeddings = self.get_embeddings(documents)
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def upsert_to_collection(
        self,
        collection_name: str,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict],
    ):
        """Embed and upsert documents into an arbitrary named collection."""
        col = self.get_collection(collection_name)
        embeddings = self.get_embeddings(documents)
        col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query_collection(
        self,
        collection_name: str,
        query: str,
        n_results: int = 5,
    ) -> dict:
        """Embed query and run similarity search against a named collection."""
        col = self.get_collection(collection_name)
        if col.count() == 0:
            return {}
        query_vector = self.get_embeddings(query, is_query=True)
        return col.query(query_embeddings=query_vector, n_results=min(n_results, col.count()))