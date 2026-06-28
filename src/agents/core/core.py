import ollama
import json
import re
import logging
import os
from typing import Dict, Any
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ConstructionAgentSystem")

class BaseLLMAgent(ABC):
    def __init__(self, config: dict, verbose: bool = True):
        # ── EXACT CONFIGURATION KEY ALIGNMENT ────────────────────────────────
        agent_config = config.get('agent', {})
        
        # Explicitly targets your config keys: agent_provider and agent_model
        self.provider = agent_config.get('agent_provider', 'ollama').lower()
        self.model = agent_config.get('agent_model', 'qwen3:8b')
        self.verbose = verbose
        
        # Initialize selected pipeline providers
        self._init_provider_client(config)

    def _init_provider_client(self, config: dict):
        """Initializes third-party SDK connections dynamically."""
        agent_config = config.get('agent', {})
        api_key = agent_config.get('api_key') or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if self.provider == 'openai':
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except ImportError:
                logger.error("OpenAI package not installed. Run 'pip install openai'")
                
        elif self.provider == 'anthropic':
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=api_key)
            except ImportError:
                logger.error("Anthropic package not installed. Run 'pip install anthropic'")
                
        elif self.provider == 'google':
            try:
                from google import genai
                # Sets up official Google GenAI execution wrapper
                self.client = genai.Client(api_key=api_key)
            except ImportError:
                logger.error("Google GenAI SDK not installed. Run 'pip install google-genai'")
                
        elif self.provider == 'ollama':
            self.client = None 

    def _call_llm(self, prompt: str, is_json: bool = False) -> str:
        try:
            if self.verbose:
                print(f"Calling LLM ({self.provider}) with model: {self.model}")

            # Route execution to correct client block
            if self.provider == 'ollama':
                return self._call_ollama(prompt, is_json)
            elif self.provider == 'openai':
                return self._call_openai(prompt, is_json)
            elif self.provider == 'anthropic':
                return self._call_anthropic(prompt, is_json)
            elif self.provider == 'google':
                return self._call_google(prompt, is_json)
            else:
                raise ValueError(f"Unsupported provider parameter configuration: {self.provider}")

        except Exception as e:
            logger.error(f"LLM Call failed [{self.provider} - {self.model}]: {e}")
            return ""

    def _call_ollama(self, prompt: str, is_json: bool) -> str:
        options = {"format": "json"} if is_json else {}
        response = ollama.generate(
            model=self.model, 
            prompt=prompt,
            **options
        )
        if self.verbose:
            print("LLM Raw response['response']:", response['response'])
        return response['response']

    def _call_openai(self, prompt: str, is_json: bool) -> str:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        if is_json:
            kwargs["response_format"] = {"type": "json_object"}
            
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if self.verbose:
            print("LLM Raw response:", content)
        return content

    def _call_anthropic(self, prompt: str, is_json: bool) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text
        if self.verbose:
            print("LLM Raw response:", content)
        return content

    def _call_google(self, prompt: str, is_json: bool) -> str:
        """Executes content extraction native calls via google-genai runtime wheels."""
        from google.genai import types
        
        config_kwargs = {}
        if is_json:
            # Enforce true JSON structured outputs directly from the Google API engine layer
            config_kwargs["response_mime_type"] = "application/json"
            
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        )
        if self.verbose:
            print("LLM Raw response:", response.text)
        return response.text

    def _extract_json_string(self, raw_text: str) -> str:
        """Uses regex to find the first JSON-like structure in a string."""
        match = re.search(r'(\{.*\}|\[.*\])', raw_text, re.DOTALL)
        return match.group(0) if match else raw_text
    
    def _call_and_parse(self, prompt: str) -> Dict[str, Any]:
        """Utility to handle the repetitive LLM -> String -> Dict flow."""
        raw = self._call_llm(prompt, is_json=True)
        print("Raw LLM Output:", raw)  
        if not raw.strip():
            return {"error": "Empty response from LLM"}

        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return {"materials": data}
            out = data if isinstance(data, dict) else {}
            print("Parsed LLM Output:", out)  
            return out
            
        except Exception as e:
            try:
                clean_raw = self._extract_json_string(raw)
                data = json.loads(clean_raw)
                return data if isinstance(data, dict) else {}
            except Exception as final_e:
                logger.error(f"Parse failed: {final_e}")
                return {"parsing_error": str(final_e), "raw_response": raw}