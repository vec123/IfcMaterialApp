# src/agents/ifc/rag_agents.py

import json
from typing import List, Dict, Any
from src.agents.core.core import BaseLLMAgent

class QueryExpansionAgent(BaseLLMAgent):
    """
    Pre-RAG Agent: Normalizes and expands BIM/IFC queries by adding technical terms,
    multilingual structural context (Catalan, Spanish, English), and enforcing dimension flags.
    """
    def __init__(self, config: dict, verbose: bool = False):
        super().__init__(config, verbose=verbose)

    def expand(self, raw_query: str, domain: str) -> Dict[str, Any]:
        prompt = f"""You are an expert BIM Manager and Ontologist specialized in IFC structural schemas.
The user is searching for a node in a vector database corresponding to the domain: '{domain}'.

Raw User Search Query: "{raw_query}"

Task: Optimize this query for a dense vector search engine.
Generate an expanded profile containing:
1. Normalized base term.
2. Structural synonyms across English, Spanish, Catalan, Italian.

REGIONAL & ARCHITECTURAL TRANSLATION PROTOCOL:
- Use localized Peninsular Spanish (Castilian) and Standard Catalan architectural/engineering terminology.
- Avoid literal, colloquial, or cross-regional dictionary translations (e.g., ensure regional European BIM industry standards are met for materials and structural components).
- Strip out conversational verbs.
- Include variations with and without regional accents/diacritics in the synonyms list to maximize vector matching.

3. Critical numeric dimensions parsed explicitly (if any).
4. An enhanced semantic search string combined from the elements above.

Return ONLY valid JSON (no markdown block formatting, no extra explanation outside the structure):
{{
  "normalized": "<string>",
  "synonyms": ["<synonym_1>", "<synonym_2>"],
  "dimensions": "<string_or_null>",
  "expanded_query": "<optimized text string for vector retrieval>"
}}
"""
        try:
            return self._call_and_parse(prompt)
        except Exception:
            return {
                "normalized": raw_query,
                "synonyms": [],
                "dimensions": None,
                "expanded_query": raw_query
            }