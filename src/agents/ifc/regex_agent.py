# src/agents/query/query_mats/grouping_agents.py

import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, List

from src.agents.core.core import BaseLLMAgent

_FALLBACK_PATTERN = r"_\d+_\d+$"
_FALLBACK_EXPLANATION = "Strip the numeric instance identifiers (e.g. _26379_)."

# ── LOAD NATURAL LANGUAGE ARCHITECTURAL CONTEXT ─────────────────────────────
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[3]  # Points directly back to project root

CONTEXT_PATH = PROJECT_ROOT / "IFCintegration/context/naming_patterns.md"

if CONTEXT_PATH.exists():
    with open(CONTEXT_PATH, "r", encoding="utf-8") as f:
        _ARCHITECT_CONTEXT = f.read()
else:
    _ARCHITECT_CONTEXT = "IFC instance names contain trailing numeric tracking hashes that must be stripped."


class RegexProposalAgent(BaseLLMAgent):
    """
    Analyzes architectural IFC strings using natural language patterns 
    and returns a regex suitable for re.sub() to reveal the true archetype.
    """
    _SAMPLE = 25

    def __init__(self, config: dict, verbose: bool = False):
        """
        No default configs allowed. Pulls directly from unified system configuration mapping.
        """
        # Ensure the underlying BaseLLMAgent structural expectations are met
        super().__init__(config, verbose=verbose)
        
        # Read parameters from the active config dictionary payload
        self.model_name = config.get('vector_store', {}).get('model_name', 'gpt-oss:20b')

    def propose(self, element_names: List[str]) -> Dict[str, str]:
        sample = element_names[: self._SAMPLE]
        names_block = "\n".join(f"  - {n}" for n in sample)

        prompt = f"""
{_ARCHITECT_CONTEXT}

You are analyzing the IFC building-element names listed below to find their unique per-instance suffix.

Sample names from this file:
{names_block}

Task: Propose a Python regex pattern to pass to `re.sub(pattern, "", name)` that strips out only the unique instance tracking identifiers and trailing numeric hashes, leaving the shared architectural "type key" identical for all instances of the same composition.

Return ONLY valid JSON (no markdown block wrapper, no text explanations outside the JSON structure):
{{
  "pattern": "<Python regex string>",
  "explanation": "<Description of what the pattern strips in one short sentence>"
}}
"""
        try:
            # self._call_and_parse uses the instantiated model definitions under the hood
            result = self._call_and_parse(prompt)
            pattern = result.get("pattern", _FALLBACK_PATTERN)
            explanation = result.get("explanation", _FALLBACK_EXPLANATION)
            re.compile(pattern)  # Quick compile test for valid regex expressions
        except Exception:
            pattern = _FALLBACK_PATTERN
            explanation = _FALLBACK_EXPLANATION + " (Fallback applied due to agent parser timeout)."

        return {"pattern": pattern, "explanation": explanation}