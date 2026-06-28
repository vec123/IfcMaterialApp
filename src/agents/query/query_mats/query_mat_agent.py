import json
from typing import Any, Dict, List

from src.agents.core.core import BaseLLMAgent, logger


class SearchMaterialAgent(BaseLLMAgent):
    """
    Orchestrator agent for material search, mirroring the
    StructureResponseAgent dispatcher pattern.

    Pipeline:
        1. MaterialTermExtractorAgent — pull a clean material term out of
           the user's free-text input.
        2. ConstructionSearchService — vector-DB lookup of canonical
           materials matching that term.
        3. MaterialRankingAgent — re-order the candidate set by priority
           against the original user query.
    """

    def __init__(self, config: dict, search_service: Any, verbose: bool = True):
        super().__init__(config, verbose)
        self.search_service = search_service
        self.extractor = MaterialTermExtractorAgent(config, verbose)
        self.ranker = MaterialRankingAgent(config, verbose)

    def search(self, user_input: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not user_input or not user_input.strip():
            return []

        # 1. Extract a clean material term
        term = self.extractor.extract(user_input)

        if self.verbose:
            print("SearchMaterialAgent - RAG search with extracted term:", term)

        # 2. Vector-DB search → candidate set
        hits = self.search_service.search(term, limit=top_k)

        if self.verbose:
            print("SearchMaterialAgent - RAG hits:", hits)

        candidates: List[Dict[str, Any]] = []
        seen: set = set()

        for h in hits:
            meta = h.get("metadata") or {}
            
            name = (
                meta.get("material") or 
                meta.get("material_name") or 
                meta.get("name") or 
                h.get("id") or 
                h.get("content")
            )
            
            if not name or name in seen:
                continue

            seen.add(name)
            candidates.append({
                "material": name,
                "score": h.get("score", 0),
                "metadata": meta,
            })
        
        if self.verbose:
            print("SearchMaterialAgent - candidates:", candidates)

        out = self.ranker.rank(user_input, candidates)

        if self.verbose:
            print("SearchMaterialAgent - return:", out)

        return out


class MaterialTermExtractorAgent(BaseLLMAgent):
    """Extracts a single material term from a free-text user query."""

    def extract(self, user_input: str) -> str:
        prompt = f"""
        Extract the single material term the user is asking about.
        If the input already looks like a bare material name, return it as-is.
        Strip thicknesses, quantities, and surrounding prose.

        User Input: "{user_input}"

        Return ONLY JSON: {{"material": "<single material term>"}}
        """
        raw = self._call_llm(prompt, is_json=True)
        try:
            data = json.loads(self._extract_json_string(raw))
            mat = str(data.get("material", "")).strip()
            return mat or user_input.strip()
        except Exception as e:
            logger.error(f"MaterialTermExtractorAgent parse failed: {e}")
            return user_input.strip()


class MaterialRankingAgent(BaseLLMAgent):
    """
    Re-orders a candidate set of materials by priority against the user's
    original query, using an LLM as the semantic ranker.

    The vector store returns hits by cosine similarity; this layer adds an
    LLM judgment so cues in the prompt (e.g. "ventilated", "rigid",
    Spanish vs English wording) bubble the best match to the top.
    """

    def rank(
        self,
        user_input: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not candidates or len(candidates) < 2:
            return candidates

        catalog = [
            {"idx": i, "material": c["material"], "metadata": c.get("metadata", {})}
            for i, c in enumerate(candidates)
        ]

        prompt = f"""
        You are a material-ranking agent. Given a user query and a set of
        candidate materials retrieved from the database, decide the order
        that best matches the user's intent — most relevant first.

        User Query: "{user_input}"

        Candidates (each annotated with its original "idx"):
        {json.dumps(catalog, ensure_ascii=False)}

        Rules:
        - Prefer candidates whose name and properties best match the user's
          wording (synonyms and translations across ES/EN/CAT are fine).
        - If no candidate is clearly more relevant, keep the original order.
        - The output MUST contain every original idx exactly once.

        Return ONLY JSON of the form:
        {{"order": [<idx>, <idx>, ...]}}
        """

        raw = self._call_llm(prompt, is_json=True)
        try:
            data = json.loads(self._extract_json_string(raw))
            order = data.get("order", [])
        except Exception as e:
            logger.error(f"MaterialRankingAgent parse failed: {e}")
            return candidates

        if not isinstance(order, list):
            return candidates

        n = len(candidates)
        seen: set = set()
        ordered: List[Dict[str, Any]] = []
        for idx in order:
            if isinstance(idx, int) and 0 <= idx < n and idx not in seen:
                ordered.append(candidates[idx])
                seen.add(idx)
        for i in range(n):
            if i not in seen:
                ordered.append(candidates[i])
        return ordered
