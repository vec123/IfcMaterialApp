from typing import Any, Dict, List
import json
import os
from src.agents.core.core import BaseLLMAgent, logger


class LLMMaterialReranker(BaseLLMAgent):
    """
    Reranks consolidated vector database material candidates alongside all their 
    linguistic variations dynamically against a target architectural query using an LLM.
    """

    def rerank(self, user_query: str, candidates: List[Any]) -> List[Dict[str, Any]]:
        """
        Groups all candidates into a unified evaluation payload, prompts the LLM 
        to execute a global comparative sorting matrix, and returns the raw parsed results.
        """
        # ── 1. PREPARE THE CONSOLIDATED CANDIDATE PAYLOAD ───────────────────
        formatted_candidates = []
        for index, cand in enumerate(candidates):
            if isinstance(cand, dict):
                raw_id = cand.get("id") or cand.get("material_id") or f"Unknown_{index}"
                metadata = cand.get("metadata", {})
                variations = metadata.get("variations") or metadata.get("language_variants") or []
            else:
                raw_id = getattr(cand, "id", None) or getattr(cand, "material_id", f"Unknown_{index}")
                metadata = getattr(cand, "metadata", {})
                variations = metadata.get("variations", []) if isinstance(metadata, dict) else []

            formatted_candidates.append({
                "candidate_index": index,
                "raw_id": raw_id,
                "variations": variations
            })

        # ── 2. CONSTRUCT THE COMPREHENSIVE BATCH COMPARISON PROMPT ───────────
        prompt = f"""
        You are an expert IFC BIM material matching system. 
        Your task is to look at a user's architectural query and rerank a list of material candidates based on semantic relevance.
        
        USER QUERY: "{user_query}"

        RETRIEVED CANDIDATES (With their 10 translation variations):
        {json.dumps(formatted_candidates, indent=2, ensure_ascii=False)}

        CRITICAL RERANKING INSTRUCTIONS:
        1. Evaluate the "raw_id" AND all 10 language strings inside the "variations" list against the USER QUERY.
        2. Assign a "relevance_score" between 0.00 (completely irrelevant) and 1.00 (perfect definitive match).
        3. Provide a cleaned "display_name" by stripping away alphanumeric
          database hash patterns, project codes, prefixes, suffixes, and trailing random index numbers 
        Preserve the core linguistic essence (e.g., Catalan/Spanish/English).
        4. Sort the output array in descending order, putting the highest "relevance_score" items first.

        Return ONLY a JSON object containing a top-level list named "ranked_results" matching this exact schema:
        {{
            "ranked_results": [
                {{
                    "candidate_index": integer,
                    "raw_id": "string",
                    "display_name": "string",
                    "relevance_score": float,
                    "justification": "string"
                }}
            ]
        }}
        """

        # ── 3. TELEMETRY LOGS: PRINT INPUT ─────────────────────────────────
        print("=" * 80)
        print(f"📥 LLM RERANKER INPUT QUERY: {user_query}")
        print(f"📥 RETRIEVED CANDIDATES PAYLOAD SENT TO LLM:")
        print(json.dumps(formatted_candidates, indent=2, ensure_ascii=False))
        print("=" * 80)
        
        # ── 4. EXECUTE PIPELINE TRANSACTION WITH ZERO FALLBACKS ────────────
        output = self._call_and_parse(prompt)

        # ── 5. TELEMETRY LOGS: PRINT OUTPUT ────────────────────────────────
        print("=" * 80)
        print("📤 LLM RERANKER RAW RETURNED OUTPUT:")
        print(json.dumps(output, indent=2, ensure_ascii=False))
        print("=" * 80)
                
        return output["ranked_results"]