# IfcMaterialAgent

A RAG-powered microservice for managing construction materials and typologies within the **EnergIA BIM-BEPS** platform. It parses IFC BIM models, enriches extracted data with LLM-generated descriptions, and provides semantic search across materials and construction typologies via a Streamlit web UI.

---

## Overview

This service bridges IFC/BIM data with energy simulation pipelines by:

1. **Parsing** IFC models to extract materials, layer typologies, and element associations
2. **Ingesting** curated materials and construction typologies through AI-validated forms
3. **Embedding** all data into a multilingual vector database (ChromaDB)
4. **Matching** IFC-extracted materials against the curated material database
5. **Searching** materials and typologies with natural language queries

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Web UI                        │
│   Upload │ Search │ Add/Update │ Match │ IFC Explorer    │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────▼─────────────┐
        │       Agent Orchestrators  │
        │  AddMaterial  AddTypology  │
        │  IFC Agents   MatchAgents  │
        └──────┬─────────────┬──────┘
               │             │
   ┌───────────▼──┐   ┌──────▼──────────────┐
   │  Ollama LLMs │   │  RAG / Vector DB     │
   │  llama3.1:8b │   │  ChromaDB            │
   │  qwen3:8b    │   │  multilingual-e5-l.  │
   └──────────────┘   │  bge-reranker-v2-m3  │
                      └──────────────────────┘
                               │
               ┌───────────────▼──────────────┐
               │        JSON Persistence       │
               │  materials.json               │
               │  typologies.json              │
               │  extracted_from_ifc/*.json    │
               └──────────────────────────────┘
```

---

## Project Structure

```
IfcMaterialAgent/
├── conf/
│   └── config.yaml                  # Central config: models, collections, paths
├── data_in/
│   ├── provided_materials/          # Curated material database (JSON)
│   ├── provided_typologies/         # Construction typology definitions (JSON)
│   ├── extracted_from_ifc/          # IFC parsing output (auto-generated)
│   └── connections/                 # Graph edge relationships
├── src/
│   ├── agents/
│   │   ├── core/                    # BaseLLMAgent (Ollama interface)
│   │   ├── add_mats/                # Material ingestion pipeline
│   │   ├── add_typologies/          # Typology ingestion pipeline
│   │   ├── ifc/                     # IFC enrichment & retrieval agents
│   │   ├── query/                   # Query orchestration
│   │   ├── match/                   # Material matching logic
│   │   └── helpers/                 # Reranking, ID resolution
│   ├── RAG/
│   │   ├── vectorstore.py           # ChromaDB wrapper (SentenceTransformer)
│   │   ├── searchengine.py          # ConstructionSearchService
│   │   └── dataingestion.py         # JSON ingestion utilities
│   ├── ifc/
│   │   └── extract_ifc_data.py      # IFC → JSON extraction
│   ├── graph/
│   │   └── navigator.py             # GraphNavigationService
│   ├── vector_db_creators/
│   │   ├── build_graph_db.py        # Builds material + typology vector DBs
│   │   └── build_ifc_graph_db.py    # Builds IFC-extracted vector DB
│   └── utils/
│       └── load_conf.py             # YAML config loader
├── scripts/
│   ├── app/
│   │   ├── app.py                   # Main entry point (Streamlit router)
│   │   ├── loader/                  # Shared caching & agent initialization
│   │   └── pages/                   # Multi-page Streamlit UI
│   │       ├── upload/              # Upload IFC / materials / typologies
│   │       ├── search/              # Semantic search UI
│   │       ├── add_and_update/      # Material/typology CRUD
│   │       ├── match/               # Material matching view
│   │       └── ifc/                 # IFC explorer
│   └── tests/                       # Test suites
├── VectorDB/                         # Persisted ChromaDB data
├── Dockerfile
├── .env
└── conf/config.yaml
```

---

## Prerequisites

### Local LLMs via Ollama

Install [Ollama](https://ollama.com) and pull the required models:

```bash
ollama pull llama3.1:8b    # Agent orchestration (required)
ollama pull qwen3:8b       # Material translation (optional)
```

### Python Environment

```bash
pip install -r requirements.txt
```

Key dependencies:
- `streamlit` — Web UI
- `chromadb` — Vector database
- `sentence-transformers` — Embedding models
- `torch` — GPU/CPU acceleration
- `ollama` — Local LLM client
- `ifcopenshell` — IFC file parsing
- `pydantic` — Data validation
- `python-dotenv` — Environment variable loading

Embedding models are downloaded automatically on first run:
- `intfloat/multilingual-e5-large` — Multilingual embeddings
- `BAAI/bge-reranker-v2-m3` — Cross-encoder reranker

---

## Configuration

### Environment Variables (`.env`)

```env
CONFIG_PATH=conf/config.yaml
UPLOADED_TYPOLOGIES=data_in/provided_typologies/typologies.json
UPLOADED_MATERIALS=data_in/provided_materials/materials.json
IFC_EXTRACTED_DIR=data_in/extracted_from_ifc
VECTOR_DB_PATH=VectorDB/constructions_and_materials
IFC_VECTOR_DB_PATH=VectorDB/ifc_elements
```

### `conf/config.yaml`

```yaml
vector_store:
  material_collection_name: "materials_v1"
  typology_collection_name: "typologies_v1"
  ifc_element_collection_name: "ifc_elements_v1"
  model_name: "intfloat/multilingual-e5-large"
  reranker: "BAAI/bge-reranker-v2-m3"
  batch_size: 32

ingestion:
  translator_model: "qwen3:8b"

agent:
  model: "llama3.1:8b"

compute:
  device: "cuda"   # Use "cpu" if no GPU available
```

---

## Running the Application

### Local Development

```bash
PYTHONPATH=. streamlit run scripts/app/app.py
```

Open [http://localhost:8501](http://localhost:8501).

### Docker

```bash
docker build -t ifc-material-agent .

docker run -p 8501:8501 \
  -v $(pwd)/data_in:/app/data_in \
  -v $(pwd)/VectorDB:/app/VectorDB \
  --env-file .env \
  ifc-material-agent
```

---

## Data Formats

### Materials (`data_in/provided_materials/materials.json`)

```json
{
  "lana mineral": {
    "density": 40.0,
    "thermal_conductivity": 0.037,
    "specific_heat": 840.0
  },
  "fábrica de ladrillo cerámico": {
    "density": 780.0,
    "thermal_conductivity": 0.35,
    "specific_heat": 1000.0
  }
}
```

Required properties per material: `density` (kg/m³), `thermal_conductivity` (W/m·K), `specific_heat` (J/kg·K).

### Typologies (`data_in/provided_typologies/typologies.json`)

```json
{
  "S01": {
    "category": "Fachada",
    "cte_code": "F.1.1",
    "u_formula": "1/(0.54+RAT)",
    "layers": [
      {"material": "fábrica de ladrillo cerámico", "thickness": 0.115},
      {"material": "lana mineral", "thickness": "eAT"}
    ]
  }
}
```

`thickness` can be a fixed value in meters or `"eAT"` (variable thickness used in U-value formula).

### IFC Extraction Output (`data_in/extracted_from_ifc/`)

After uploading an IFC file, four JSON files are generated:

| File | Contents |
|------|----------|
| `element_to_typology.json` | Maps element names → typology (layer set) names |
| `typologies_as_layers.json` | Maps typology name → ordered list of `{material, thickness_m, layer_name}` |
| `unique_materials.json` | All unique material names grouped by IFC element type |
| `elements_without_typology.json` | Elements with no material association |

Supported IFC element types: `IfcWall`, `IfcSlab`, `IfcRoof`, `IfcDoor`, `IfcWindow`, `IfcBeam`, `IfcColumn`, `IfcStair`, `IfcRamp`, `IfcCurtainWall`.

---

## Agent Pipelines

### Add Material (6-step validation pipeline)

```
User input
  → 1. IntentAgent         — detect "add_new" vs "update"
  → 2. ExtractionAgent     — parse name, density, λ, Cp
  → 3. NameValidatorAgent  — reject placeholder names
  → 4. PropsValidatorAgent — ensure all 4 properties are present
  → 5. DuplicateCheckAgent — block duplicates (in add_new mode)
  → 6. PersistenceAgent    — save to JSON + upsert into ChromaDB
```

### Add Typology (6-step validation pipeline)

```
User input
  → 1. IntentAgent          — detect intent
  → 2. ExtractionAgent      — parse category, layers, CTE code, U-formula
  → 3. IDResolutionAgent    — auto-generate or validate unique ID
  → 4. StructureValidator   — check required fields
  → 5. MaterialExistence    — verify all layer materials exist in DB
  → 6. PersistenceAgent     — save to JSON + upsert into ChromaDB
```

### IFC Upload & Matching Workflow

```
1. Upload .ifc file
   → extract_ifc_data() → 4 JSON output files

2. Build IFC vector DB
   → MaterialExtensionAgent enriches each material:
     - English & Spanish descriptions
     - Keyword variations
   → Embed all enriched materials into ChromaDB

3. Match IFC materials to database materials
   → QueryExpansionAgent generates multilingual synonyms
   → ChromaDB retrieves top-k similar materials
   → RAGRerankerAgent (cross-encoder) reranks results
   → Matching matrix saved and displayed in UI
```

---

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `BaseLLMAgent` | `src/agents/core/core.py` | Ollama LLM interface; base for all agents |
| `AddMaterialOrchestrator` | `src/agents/add_mats/orchestrator.py` | Runs full material ingestion pipeline |
| `AddTypologyOrchestrator` | `src/agents/add_typologies/orchestrator.py` | Runs full typology ingestion pipeline |
| `VectorStoreManager` | `src/RAG/vectorstore.py` | ChromaDB wrapper with SentenceTransformer embeddings |
| `ConstructionSearchService` | `src/RAG/searchengine.py` | High-level semantic search API |
| `GraphNavigationService` | `src/graph/navigator.py` | Traverse material ↔ typology graph |
| `extract_ifc_data` | `src/ifc/extract_ifc_data.py` | Parse .ifc → JSON outputs |
| `RAGRerankerAgent` | `src/agents/helpers/reranker.py` | Cross-encoder result reranking |

---

## ChromaDB Collections

| Collection | Content |
|------------|---------|
| `materials_v1` | Curated materials with thermal properties |
| `typologies_v1` | Construction typologies with layer definitions |
| `ifc_elements_v1` | LLM-enriched materials extracted from IFC models |

Embeddings use the `"query: "` / `"passage: "` prefixes required by `multilingual-e5-large`.

---

## Tests

```bash
# Material ingestion tests
PYTHONPATH=. python scripts/tests/add_tests/test_add_material.py

# Search/query tests
PYTHONPATH=. python scripts/tests/query_tests/test_search.py

# Vector DB tests
PYTHONPATH=. python scripts/tests/vectordb_tests/test_vectordb.py
```

---

## Integration with EnergIA BIM-BEPS

This microservice is part of the **EnergIA BIM-BEPS** platform and feeds validated material thermal properties and construction typologies to downstream energy simulation (BEPS) services. Its outputs — structured JSON databases and semantic embeddings — serve as the material knowledge layer for the platform's AI-assisted energy analysis pipeline.
