import requests
import json
import re
from typing import List, Any, Dict
from sentence_transformers import CrossEncoder

class StructuralReranker:
    def __init__(self, model_name: str = "qwen2.5:7b-instruct"):
        # Explicitly enforce a conversational model if a cross-encoder string is passed accidentally
        if "reranker" in model_name.lower():
            print(f"⚠️ Warning: Configuration specified a Cross-Encoder '{model_name}'. Overriding with conversational 'qwen2.5:7b-instruct'.")
            self.model_name = "qwen2.5:7b-instruct"
        else:
            self.model_name = model_name
            
        self.api_url = "http://127.0.0.1:11434/api/generate" 
        print(f"💬 Conversational Batch-Reranker Engine Initialized using: '{self.model_name}'")

    def _extract_text(self, cand: Any) -> str:
        """Extracts text descriptors cleanly from structural IFC elements and strips technical metadata noise."""
        raw_text = ""
        if isinstance(cand, str):
            raw_text = cand
        elif isinstance(cand, dict):
            for key in ["document", "text", "id", "material_id"]:
                if key in cand and cand[key]:
                    raw_text = str(cand[key])
                    break
            if not raw_text:
                raw_text = str(cand)
        else:
            for attr in ["document", "text", "id", "material_id"]:
                if hasattr(cand, attr):
                    val = getattr(cand, attr)
                    if val:
                        raw_text = str(val)
                        break
            if not raw_text:
                raw_text = str(cand)

        # CLEAN NOMENCLATURE: e.g., 'E325_FormigoMursContencio_HA' -> 'Formigo Murs Contencio HA'
        cleaned = re.sub(r'^[A-Z0-9]{3,4}_', '', raw_text)  # Remove structural sequence prefixes like E325_ or E5Z2_
        cleaned = cleaned.replace('_', ' ')                 # Swap underscores out for readable spaces
        return cleaned.strip()

    def rerank(self, query: str, candidates: List[Any], top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        
        print("candidates incoming to batch:_ ", len(candidates))
        
        # 1. Map candidates into a predictable collection with isolated string indices
        candidate_map = []
        for idx, cand in enumerate(candidates):
            candidate_map.append({
                "index_key": str(idx),
                "clean_text": self._extract_text(cand),
                "raw_object": cand
            })

        # 2. Serialize all entities into a single batch text block for the prompt
        batch_entries_string = ""
        for item in candidate_map:
            batch_entries_string += f"ID {item['index_key']}: {item['clean_text']}\n"

        # 3. Formulate the global multi-candidate comparative tracking prompt
        prompt = (
            f"<|im_start|>system\n"
            f"You are a specialized BIM and architecture assistant scoring material matches.\n"
            f"Evaluate the semantic match relevance between the user search query and the provided list of IFC Material Entities.\n"
            f"Language Context: Catalan, Spanish, and English translations are expected.\n\n"
            f"Output your scores ONLY as a single valid JSON object mapping the string ID to its corresponding float score between 0.00 and 1.00.\n"
            f"Example format:\n"
            f"{{\n"
            f"  \"0\": 0.95,\n"
            f"  \"1\": 0.00\n"
            f"}}\n"
            f"Do not write any introduction text, markdown code blocks, explanations, or reasoning.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Search Query: {query}\n\n"
            f"Material Entities to Evaluate:\n"
            f"{batch_entries_string}\n"
            f"JSON Output:\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{{"  # Pre-fill JSON bracket to lock the model structure
        )

        scored_results = []
        llm_success = False
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json", 
                "options": {
                    "temperature": 0.0
                }
            }
            
            # Increased timeout slightly to account for the larger localized batch generation token lengths
            response = requests.post(self.api_url, json=payload, timeout=120)
            if response.status_code == 200:
                raw_response = response.json().get("response", "").strip()
                
                # Dynamic syntax patching if truncation strips formatting bounds
                if not raw_response.startswith("{"):
                    raw_response = "{" + raw_response
                if not raw_response.endswith("}"):
                    raw_response = raw_response + "}"
                
                scores_dict = json.loads(raw_response)
                
                # Extract and parse values out of the consolidated response structure
                for item in candidate_map:
                    assigned_score = float(scores_dict.get(item["index_key"], 0.0))
                    scored_results.append({
                        "candidate": item["raw_object"],
                        "score": assigned_score
                    })
                llm_success = True
                
        except Exception as e:
            print(f"⚠️ Batch parsing anomaly caught: {e}. Falling back to token verification algorithm.")
            pass
            
        # 4. Fallback: Local loop token scan if the batch payload experiences a problem
        if not llm_success:
            scored_results = []
            clean_q = query.lower()
            query_words = [w for w in re.split(r'[\s,]+', clean_q) if len(w) > 3]
            
            for item in candidate_map:
                clean_d = item["clean_text"].lower()
                score = 0.50 if any(w in clean_d for w in query_words) else 0.00
                scored_results.append({
                    "candidate": item["raw_object"],
                    "score": score
                })

        # Sort elements cleanly by evaluation scores descending
        scored_results = sorted(scored_results, key=lambda x: x["score"], reverse=True)
        print("------------------------------------")
        print("batch_scored_results:_ ", scored_results)
        return scored_results[:top_k]
    


import torch
from typing import Any, List, Dict
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str):
        # e.g., "BAAI/bge-reranker-large" or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.model = CrossEncoder(model_name) 

    def rerank(self, query: str, candidates: list, top_k: int = 10) -> list:
        if not candidates:
            return []

        pairs = []
        for c in candidates:
            # Extract the actual text document seen by the vector store
            # Adapt this block depending on your VectorStoreManager return type
            if isinstance(c, dict):
                doc_text = c.get("document") or c.get("text") or ""
            else:
                doc_text = getattr(c, "document", None) or getattr(c, "page_content", None) or str(c)
            
            pairs.append([query, doc_text])

        # Compute real transformer scores
        scores = self.model.predict(pairs)

        # Build clean results
        results = []
        for idx, score in enumerate(scores):
            results.append({
                "candidate": candidates[idx],
                "score": float(score)  # Real transformer confidence score
            })

        # Sort descending by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
class CrossEncoderReranker__:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", verbose: bool = True):
        """Initializes a cross-encoder with a defensive CPU fail-safe routing mechanism."""
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔌 Attempting to load structural evaluation model '{model_name}' onto {target_device.upper()}...")
        self.verbose = verbose
        try:
            self.model = CrossEncoder(model_name, device=target_device)
            print(f"🚀 Cross-Encoder active on engine device: '{target_device.upper()}'.")
        except (RuntimeError, Exception) as e:
            if "Out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
                print("⚠️ GPU VRAM Congested/Exhausted! Gracefully routing runtime to CPU...")
                try:
                    self.model = CrossEncoder(model_name, device="cpu")
                    print("✅ Multilingual Engine safely initialized on HOST CPU.")
                except Exception as cpu_err:
                    print("💥 Severe error allocation failure across both processing environments.")
                    raise cpu_err
            else:
                raise e

    def rerank(self, query: str, candidates: List[Any], edges_db: List[Dict[str, Any]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes cross-attention matching by hydrating database IDs with 
        their downstream structural relational typologies extracted out of the graph map.
        """
        if not candidates:
            return []
        
        print(f"📥 Batch-scoring {len(candidates)} records against unified anchor: '{query[:60]}...'")
        
        pairs = []
        for cand in candidates:
            # Isolate material identity key
            if isinstance(cand, str): 
                raw_name = cand
            elif isinstance(cand, dict): 
                raw_name = cand.get("id") or cand.get("material_id") or str(cand)
            else: 
                raw_name = getattr(cand, "id", None) or getattr(cand, "material_id", None) or str(cand)
            
            # Extract and append structural graph context to the text footprint 
            structural_context = ""
            if edges_db:
                linked_typos = {
                    edge["mapped_typology_id"] 
                    for edge in edges_db 
                    if edge.get("material_id") == raw_name
                }
                if linked_typos:
                    structural_context = " | " + " | ".join(sorted(list(linked_typos)))
            
            # Assemble hydrated dual-sided evaluation token pairs
            formatted_query = f"Search Material Description: {query}"
            formatted_candidate = f"IFC Material Entity Database Entry: {raw_name}{structural_context}"
            pairs.append([formatted_query, formatted_candidate])

        if self.verbose:
            print("pairs: ", pairs)

        scores = self.model.predict(pairs, batch_size=32, convert_to_numpy=True)

        scored_results = []
        for cand, score in zip(candidates, scores):
            scored_results.append({
                "candidate": cand,
                "score": float(score)
            })

        scored_results = sorted(scored_results, key=lambda x: x["score"], reverse=True)
        print("------------------------------------")
        print("✅ Multilingual structural scoring completed successfully.")
        return scored_results[:top_k]



import re
import httpx
import torch
import unicodedata
from typing import Any, List, Dict
from sentence_transformers import CrossEncoder

class BilingualRerankManager:
    def __init__(self, hf_reranker_name: str = "BAAI/bge-reranker-v2-m3", ollama_translator: str = "salamandra-ta:latest", verbose = True):
        self.verbose = verbose
        self.ollama_url = "http://localhost:11434/api/generate"
        self.translator_model = ollama_translator
        
        print(f"🔌 Loading structural evaluation model: {hf_reranker_name}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(hf_reranker_name, device=device)
        print(f"🚀 Cross-Encoder active on device: '{device.upper()}'.")

    def _normalize_string(self, text: str) -> str:
        """Removes all accents, punctuation, and converts to lowercase for flawless cross-lingual string matching."""
        text = text.lower().replace("_", " ").replace("-", " ")
        normalized = unicodedata.normalize('NFD', text)
        return "".join([c for c in normalized if unicodedata.category(c) != 'Mn'])

    def translate_and_expand(self, raw_query: str) -> str:
        """Translates the user query space into core synonymous keywords."""
        prompt = (
            f"Translate the following raw material description into Catalan, and Spanish.\n"
            f"Provide only synonymous expressions as a simple comma-separated list. No notes, no markdown.\n"
            f"Input: {raw_query} "
        )
        
        payload = {
            "model": self.translator_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }
        
        response = httpx.post(self.ollama_url, json=payload, timeout=2400.0)
        response.raise_for_status()
        raw_output = response.json().get("response", "").strip()

        clean_terms = re.split(r'(?i)notes|català|español|english|translation|:|ー', raw_output)[0]
        clean_terms = re.sub(r'[\*\-\#\[\]\(\)\n\r]', ' ', clean_terms)
        clean_words = [word.strip() for word in clean_terms.replace(",", " ").split() if word.strip()]
        
        seen = set()
        unique_terms = []
        for w in raw_query.lower().split():
            seen.add(self._normalize_string(w))
            unique_terms.append(w)
            
        for w in clean_words:
            norm_w = self._normalize_string(w)
            if norm_w not in seen and len(norm_w) > 2:
                seen.add(norm_w)
                unique_terms.append(w)

        return " ".join(unique_terms).strip()

    def translate_candidate(self, raw_candidate_id: str) -> str:
        """
        Queries Ollama to translate and decode complex architectural database keys 
        into plain multilingual descriptive terms.
        """
        # Clean alphanumeric code strings so the LLM reads them as natural tokens
        cleaned_id = raw_candidate_id.replace("_", " ").replace("-", " ")
        
        prompt = (
            f"Analyze this building material code: '{cleaned_id}'.\n"
            f"Identify the main material and return its name translated into English, Spanish, and Catalan.\n"
            f"Keep your response strictly to a short, comma-separated list of names. No extra text, no notes, no markdown.\n"
            f"Output Example: steel, acero, acer"
        )
        
        payload = {
            "model": self.translator_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 20} # Strict length limits speed up runtime execution
        }
        
        try:
            response = httpx.post(self.ollama_url, json=payload, timeout=10.0)
            response.raise_for_status()
            translated_output = response.json().get("response", "").strip()
            
            # Clean up potential LLM conversational garbage text leaks
            clean_output = re.split(r'(?i)notes|translation|:|ー', translated_output)[0]
            clean_output = re.sub(r'[\*\-\#\[\]\(\)\n\r]', ' ', clean_output)
            return ", ".join([w.strip() for w in clean_output.split(",") if w.strip()])
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Failed to translate candidate '{raw_candidate_id}' due to error: {e}. Falling back to tokenized ID.")
            return cleaned_id

    def rerank(self, query: str, candidates: List[Any], top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        
        # Step 1: Translate the query once
        multilingual_query = self.translate_and_expand(query)
        print(f"🔀 Normalized Query Space: '{multilingual_query}'")
        print(f"📥 Processing {len(candidates)} candidate translations directly via LLM API calls...")
        
        pairs = []
        for idx, cand in enumerate(candidates):
            if isinstance(cand, str):
                raw_name = cand
            elif isinstance(cand, dict):
                raw_name = cand.get("id") or cand.get("material_id") or str(cand)
            else:
                raw_name = getattr(cand, "id", None) or getattr(cand, "material_id", None) or str(cand)
            
            # Step 2: Dynamic Candidate Translation Pass
            translated_candidate_profile = self.translate_candidate(raw_name)
            
            if self.verbose and (idx + 1) % 10 == 0:
                print(f"   🔄 Progress: [{idx + 1}/{len(candidates)}] translated...")
                
            # Both sides are now dynamically translated prose vectors
            formatted_candidate = f"Material Element: {raw_name} ({translated_candidate_profile})"
            pairs.append([multilingual_query, formatted_candidate])

        if self.verbose:
            print("\n🔥 Fully-Translated Cross-Encoder Pairs (Sample):")
            for p in pairs[:2]:
                print(f"  Query:     '{p[0]}'\n  Candidate: '{p[1]}'\n")

        # Step 3: Balanced Batch Inference Pass
        if self.verbose:
            print("pairs: ", pairs)
        scores = self.model.predict(pairs, batch_size=32, convert_to_numpy=True)
        
        scored_results = []
        for cand, score in zip(candidates, scores):
            scored_results.append({
                "candidate": cand,
                "score": float(score)
            })

        scored_results = sorted(scored_results, key=lambda x: x["score"], reverse=True)
        print("------------------------------------")
        print(f"✅ Dynamic bidirectional translation complete. Ranking is fully calibrated.")
        return scored_results[:top_k]
import re
import httpx
import torch
from typing import Any, List, Dict
from sentence_transformers import CrossEncoder

class BilingualRerankManager_good:
    def __init__(self, hf_reranker_name: str = "BAAI/bge-reranker-base", ollama_translator: str = "salamandra-ta:latest", verbose = True):
        """
        Manages cross-lingual expansion via a single query pass over Salamandra-TA 
        and validates structural alignment through CrossEncoder.
        """
        self.verbose = verbose
        self.ollama_url = "http://localhost:11434/api/generate"
        self.translator_model = ollama_translator
        
        print(f"🔌 Loading structural evaluation model: {hf_reranker_name}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(hf_reranker_name, device=device)
        print(f"🚀 Cross-Encoder active on device: '{device.upper()}'.")

    def translate_and_expand(self, raw_query: str) -> str:
        """
        Processes the complete raw user input via Salamandra-TA once,
        extracting clean, flat structural translations across targeted languages.
        """
        prompt = (
            f"Translate the following raw material description into Catalan, and Spanish.\n"
            f"Provide only synonymous expressions as a simple comma-separated list. No notes, no markdown.\n"
            f"Input: {raw_query} "
        )
        
        payload = {
            "model": self.translator_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }
        
        if self.verbose:
            print("ollama prompt: ", prompt)
        
        response = httpx.post(self.ollama_url, json=payload, timeout=30.0)
        response.raise_for_status()
        
        raw_output = response.json().get("response", "").strip()
        if self.verbose:
            print("ollama output: ", raw_output)

        # --- Aggressive Output Filtering Block ---
        # 1. Strip out everything following explanatory headers or Note labels
        clean_terms = re.split(r'(?i)notes|català|español|english|translation|:|ー', raw_output)[0]
        
        # 2. Strip structural markdown styling characters, syntax elements, and brackets
        clean_terms = re.sub(r'[\*\-\#\[\]\(\)\n\r]', ' ', clean_terms)
        
        # 3. Standardize spacing and convert commas into a flat space-delimited query string
        clean_words = [word.strip() for word in clean_terms.replace(",", " ").split() if word.strip()]
        
        # Deduplicate terms while preserving local indexing sequence order
        seen = set()
        unique_terms = []
        # Seed with original query terms
        for w in raw_query.lower().split():
            seen.add(w)
            unique_terms.append(w)
            
        for w in clean_words:
            w_low = w.lower()
            if w_low not in seen and len(w_low) > 2:
                seen.add(w_low)
                unique_terms.append(w)

        combined_query = " ".join(unique_terms).strip()
        print(f"🔀 Filtered Symmetrical Query Space: '{combined_query}'")
        return combined_query

    def rerank(self, query: str, candidates: List[Any], top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        
        # Step 1: Translate and clean the input sequence once
        multilingual_query = self.translate_and_expand(query)
        
        # Build core material matching validation keys (e.g., ['concrete', 'concreto', 'formigo', 'calcestruzzo'])
        translation_tokens = [t.lower() for t in multilingual_query.split() if len(t) > 2]
        
        print(f"📥 Received {len(candidates)} candidates for direct batch processing.")
        
        pairs = []
        for cand in candidates:
            if isinstance(cand, str):
                raw_name = cand
            elif isinstance(cand, dict):
                raw_name = cand.get("id") or cand.get("material_id") or str(cand)
            else:
                raw_name = getattr(cand, "id", None) or getattr(cand, "material_id", None) or str(cand)
            
            # Normalize database string characters for reliable token matching
            clean_name = raw_name.lower().replace("_", " ").replace("-", " ")
            
            # Symmetrical Footprint Alignment Match Check
            if any(token in clean_name for token in translation_tokens if token not in ["material", "entity"]):
                formatted_candidate = f"IFC Material Entity: {raw_name} ({multilingual_query})"
            else:
                # If no matching keywords are found, do not append the translation query string
                formatted_candidate = f"IFC Material Entity: {raw_name}"
                
            pairs.append([multilingual_query, formatted_candidate])

        if self.verbose:
            print("🚀 Executing Rerank Inference on Pairs: ", pairs) # Log a sample snippet
            
        scores = self.model.predict(pairs, batch_size=32, convert_to_numpy=True)
        
        scored_results = []
        for cand, score in zip(candidates, scores):
            scored_results.append({
                "candidate": cand,
                "score": float(score)
            })

        # Sort elements by evaluation score descending
        scored_results = sorted(scored_results, key=lambda x: x["score"], reverse=True)
        print("------------------------------------")
        print(f"✅ Symmetrical batch evaluation finalized successfully.")
        return scored_results[:top_k]
    

import httpx
import torch
import re
from typing import Any, List, Dict
from sentence_transformers import CrossEncoder


class BilingualRerankManager_old:
    def __init__(self, hf_reranker_name: str = "BAAI/bge-reranker-base", ollama_translator: str = "salamandra-ta:latest", verbose = True):
        """
        Manages cross-lingual expansion via a single query pass over Salamandra-TA 
        and validates structural alignment through CrossEncoder.
        """
        self.verbose = verbose
        self.ollama_url = "http://localhost:11434/api/generate"
        self.translator_model = ollama_translator
        
        print(f"🔌 Loading structural evaluation model: {hf_reranker_name}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(hf_reranker_name, device=device)
        print(f"🚀 Cross-Encoder active on device: '{device.upper()}'.")

    def translate_and_expand(self, raw_query: str) -> str:
        """
        Processes the complete raw user input via Salamandra-TA once,
        extracting direct material translations across targeted romance languages.
        """
        prompt = (
            f"Translate the following raw material description into Catalan, and Spanish.\n"
            f"as many synonymous expressions as possible"
            f"Input: {raw_query} "
        )
        
        payload = {
            "model": self.translator_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1  # Force deterministic formatting
            }
        }
        
        if self.verbose:
            print("ollama prompt: ", prompt)
        
        response = httpx.post(self.ollama_url, json=payload, timeout=30.0)
        response.raise_for_status()
        
        raw_output = response.json().get("response", "").strip()
        if self.verbose:
            print("ollama output: ", raw_output)

        # Guardrail: Clean out common conversational leaking phrases if present
        clean_terms = re.sub(r'(?i)here is a|translation|in english|in spanish|in catalan|:', '', raw_output)
        clean_terms = " ".join([word.strip() for word in clean_terms.split(",") if word.strip()])
        
        # Append the original query to ensure search compliance
        combined_query = f"{raw_query} {clean_terms}".strip()
        print(f"🔀 Unified Multilingual Query Space: '{combined_query}'")
        return combined_query

    def rerank(self, query: str, candidates: List[Any], top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        
        # Step 1: Translate the complete input sequence once
        multilingual_query = self.translate_and_expand(query)
        
        # Extract a list of token terms to help match/hydrate candidates natively
        translation_tokens = [t.lower() for t in multilingual_query.split() if len(t) > 2]
        
        print(f"📥 Received {len(candidates)} candidates for direct batch processing.")
        if self.verbose:
            print("multilingual_query: ", multilingual_query)
            
        pairs = []
        for cand in candidates:
            if isinstance(cand, str):
                raw_name = cand
            elif isinstance(cand, dict):
                raw_name = cand.get("id") or cand.get("material_id") or str(cand)
            else:
                raw_name = getattr(cand, "id", None) or getattr(cand, "material_id", None) or str(cand)
            
            # --- Simple Symmetrical Fallback Translation Alignment ---
            # If the database element name contains keywords found in the translated query space,
            # we inject the whole translated space right behind it to give the Cross-Encoder immediate clarity.
            clean_name = raw_name.lower().replace("_", " ").replace("-", " ")
            
            if any(token in clean_name for token in translation_tokens):
                # Symmetrical translation footprint alignment match
                formatted_candidate = f"IFC Material Entity: {raw_name} ({multilingual_query})"
            else:
                # Default blind ground context string hypothesis
                formatted_candidate = f"IFC Material Entity: {raw_name}"
                
            pairs.append([multilingual_query, formatted_candidate])

        # Step 2: Single batch inference computation over the entire pool
        if self.verbose:
            print("pairs: ", pairs)
            
        scores = self.model.predict(pairs, batch_size=32, convert_to_numpy=True)
        if self.verbose:
            print("scores: ", scores)
            
        scored_results = []
        for cand, score in zip(candidates, scores):
            scored_results.append({
                "candidate": cand,
                "score": float(score)
            })

        # Sort elements by evaluation score descending
        scored_results = sorted(scored_results, key=lambda x: x["score"], reverse=True)
        print("------------------------------------")
        print(f"✅ Batch evaluation finalized.")
        return scored_results[:top_k]