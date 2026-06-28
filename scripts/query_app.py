import streamlit as st
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__)) # scripts folder
project_root = os.path.abspath(os.path.join(current_dir, "..")) # RAGTest folder
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.query.orchestrator import PipelineOrchestrator
from src.agents.query.agents import (
    IntentAgent,
    FunctionInputAgent,
    CleaningAgent,
    ExecutionAgent,
    StructureResponseAgent,
)
from src.utils.load_conf import load_config
from src.RAG.vectorstore import VectorStoreManager
from src.RAG.searchengine import ConstructionSearchService



@st.cache_resource
def get_orchestrator(cfg):
    """
    Initializes all heavy components once and caches them.
    This replaces the 'initialize_app' logic with a more robust pattern.
    """

   
    # 2. Load Physical Database
    db_path = cfg['ingestion'].get('input_file', "./data_in/constructions.json")
    if not os.path.exists(db_path):
        st.error(f"Database file not found at {db_path}")
        st.stop()
        
    with open(db_path, 'r', encoding='utf-8') as f:
        construction_db = json.load(f)

    # 3. Initialize Agents
    agents = {
        'intent': IntentAgent(cfg),
        'param_extractor': FunctionInputAgent(cfg),
        'cleaner': CleaningAgent(cfg),
        'structurer': StructureResponseAgent(cfg),
    }
    
    # 4. Initialize Core Services
    vs_manager = VectorStoreManager(
        db_path=cfg['vector_store']['db_path'],
        collection_name=cfg['vector_store']['collection_name'],
        model_name=cfg['vector_store']['model_name']
    )
    search_service = ConstructionSearchService(vector_store=vs_manager)
    executor = ExecutionAgent(database=construction_db)

    # 5. Return the Orchestrator
    return PipelineOrchestrator(agents, search_service, executor)

def main():
    st.set_page_config(
        page_title="EnergIA BIM-BEPS", 
        page_icon="🏗️", 
        layout="wide"
    )
    cfg = load_config("conf/config.yaml")
    # Initialize the heavy lift components
    orchestrator = get_orchestrator(cfg)

    st.sidebar.header("Pipeline Configuration")
    st.sidebar.info("System connected to ChromaDB and Ollama.")
    
    st.title("🏗️ Construction Assembly Intelligence")
    st.markdown("Query the technical database using natural language.")

    # User Input
    query = st.text_input("Enter your query:", key="user_query", placeholder="e.g., Find a wall with 4cm of mineral wool")

    if query:
        with st.spinner("🧠 AI Agents are processing your request..."):
            try:
                # The Orchestrator handles the sequence: Intent -> Extract -> Map -> Execute
                intent, params, mapped, results = orchestrator.run(query)
                
                # UI Layout
                tab1, tab2, tab3 = st.tabs(["📊 Results", "🔍 Parameters", "📝 Intelligence Log"])

                with tab1:
                    if results:
                        st.success(f"Found {len(results)} matching assemblies.")
                        # Display results in a clean way
                        st.json(results)
                        
                        # Prepare download
                        json_result = json.dumps(results, indent=4, ensure_ascii=False)
                        st.download_button(
                            label="📥 Download JSON Results",
                            data=json_result,
                            file_name=f"results_{intent}.json",
                            mime="application/json"
                        )
                    else:
                        st.warning("The filters were applied, but no assemblies matched your criteria.")

                with tab2:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**LLM Extracted Entities:**")
                        st.json(params)
                    with col2:
                        st.write("**Vector-Mapped Terms:**")
                        st.json(mapped)

                with tab3:
                    st.info(f"**Detected Intent:** {intent}")
                    st.write("The system identified this query type and routed it to the corresponding specialist agent.")
            
            except Exception as e:
                st.error(f"An error occurred during pipeline execution: {e}")

if __name__ == "__main__":
    main()