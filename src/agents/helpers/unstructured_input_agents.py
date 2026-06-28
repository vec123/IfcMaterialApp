from typing import Any, Dict, List
from src.agents.core.core import BaseLLMAgent, logger

class Orchestrator(BaseLLMAgent):
    def __init__(self, cfg):
        # Pass the config to the parent class (BaseLLMAgent)
        super().__init__(cfg)
        # Store the config locally so children can access it
        self.cfg = cfg 

    def process_raw_input(self, raw_input: str) -> List[Dict[str, Any]]:
        # Chunking: 2000 chars with 200 overlap
        chunk_size, overlap = 2000, 200
        chunks = [raw_input[i:i+chunk_size] for i in range(0, len(raw_input), chunk_size - overlap)]
        
        all_extracted = []
        for chunk in chunks:
            # Sanitizer -> Extraction -> Validation flow
            clean = SanitizerWorker(self.cfg).sanitize(chunk)
            raw_materials = ExtractionWorker(self.cfg).extract(clean)
            all_extracted.extend([ValidatorWorker(self.cfg).validate(m) for m in raw_materials])
            
        # Deduplication
        deduplicated = {m["name"]: m for m in all_extracted if m["name"]}
        return list(deduplicated.values())

class SanitizerWorker(BaseLLMAgent):
    def sanitize(self, text: str) -> str:
        prompt = f"Convert this messy input into a clean, tab-separated format or raw text: {text}"
        return self._call_and_parse(prompt)

class ExtractionWorker(BaseLLMAgent):
    def extract(self, clean_text: str) -> List[Dict[str, Any]]:
        prompt = f"Map the following data to a JSON list with keys [name, density, thermal_conductivity, specific_heat]: {clean_text}"
        return self._call_and_parse(prompt).get("materials", [])

class ValidatorWorker(BaseLLMAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    def _safe_float(self, value) -> float:
        """Helper to convert any input to a valid float, defaulting to 0.0."""
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def validate(self, m: Dict[str, Any]) -> Dict[str, Any]:
            # Track quality flag
            has_missing = any(val in [None, 0.0] for val in [m.get("density"), m.get("thermal_conductivity"), m.get("specific_heat")])
            
            validated = {
                "name": m.get("name") or "Unknown Material",
                "density": self._safe_float(m.get("density")),
                "thermal_conductivity": self._safe_float(m.get("thermal_conductivity")),
                "specific_heat": self._safe_float(m.get("specific_heat")),
                "is_dirty": has_missing # Flag for the UI
            }
            return validated