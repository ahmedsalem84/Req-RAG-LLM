# In app/main_api.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # This line is fine, you can keep it.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import networkx as nx
import traceback

# Import our core logic
import sys
sys.path.append('..')
from src.core.data_loader import load_github_issues, process_issues_for_rag
from src.core.llm_handler import LLMHandler
from src.core.kg_builder import KGBuilder
from src.core.conflict_resolver import ConflictResolver
from src.core.traceability_optimizer import TraceabilityOptimizer

# --- App Initialization ---
app = FastAPI(title="RAG-RE Experimental API")

# --- Global State ---
# We initialize these to None. They will be loaded on the first API call.
state = {
    "llm_handler": None,
    "kg_builder": None,
    "resolver": None,
    "tracer": None,
    "documents": None,
    "graph": None,
    "conflicts": []
}

# --- Pydantic Models ---
class TraceRequest(BaseModel):
    source_id: str
    target_id: str

# --- Helper Function for One-Time Initialization ---
def initialize_system():
    """This function loads all the heavy models into memory."""
    if state["llm_handler"] is None:
        print("--- LAZY LOADING MODELS (ONE-TIME SETUP) ---")
        print("1. Initializing LLMHandler with phi-3-mini...")
        # Using the model you've chosen
        state["llm_handler"] = LLMHandler(model_path='meta-llama-3-8b-instruct.Q4_K_M.gguf')
        print("   LLMHandler initialized.")

        print("2. Initializing KGBuilder...")
        state["kg_builder"] = KGBuilder(state["llm_handler"])
        print("   KGBuilder initialized.")
        print("--- MODEL LOADING COMPLETE ---")

# --- API Endpoints ---
# NOTICE: The @app.on_event("startup") block has been REMOVED.

@app.post("/api/load-and-process")
async def load_and_process_data():
    try:
        # This will load the models only on the first run.
        initialize_system()

        print("API CALL: /api/load-and-process")
        print("1. Loading GitHub issues from file...")
        issues = load_github_issues('data/superset_issues.json')
        # Using a smaller number for faster testing. Phi-3 is fast, so 50 is okay.
        documents = process_issues_for_rag(issues[:5])
        state["documents"] = documents
        print(f"   Loaded and processed {len(documents)} documents.")

        # Build the Knowledge Graph
        print("2. Starting Knowledge Graph construction...")
        graph = state["kg_builder"].build_graph_from_documents(documents)
        state["graph"] = graph
        print("   Knowledge Graph construction complete.")
        print(f"   KG has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

        # Initialize other modules that depend on the graph
        print("3. Initializing resolver and tracer...")
        doc_map = {doc.split('\n')[0].split(': ')[1]: doc for doc in documents}
        state["resolver"] = ConflictResolver(graph, state["llm_handler"], doc_map)
        state["tracer"] = TraceabilityOptimizer(graph)
        print("   Resolver and tracer are ready.")

        return {"message": f"Successfully loaded {len(documents)} docs and built a KG with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges."}
    except Exception as e:
        # This will now properly catch errors during model loading or processing
        print(f"AN ERROR OCCURRED in /load-and-process: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph-data")
async def get_graph_data():
    if state["graph"] is None:
        raise HTTPException(status_code=404, detail="Graph not built yet. Please run /load-and-process first.")
    g_json = nx.node_link_data(state["graph"])
    return g_json

@app.get("/api/conflicts/detect")
async def detect_conflicts():
    if state["resolver"] is None:
        raise HTTPException(status_code=404, detail="System not initialized.")
    conflicts = state["resolver"].detect_conflicts()
    state["conflicts"] = conflicts
    return {"conflicts": conflicts}

@app.post("/api/conflicts/resolve/{conflict_index}")
async def resolve_conflict(conflict_index: int):
    if not state["conflicts"] or conflict_index >= len(state["conflicts"]):
        raise HTTPException(status_code=404, detail="Conflict index out of bounds.")
    conflict = state["conflicts"][conflict_index]
    resolution = state["resolver"].resolve_conflict_with_metaheuristic(conflict)
    return resolution

@app.post("/api/traceability/find-impact")
async def find_impact(request: dict):
    start_node = request.get("start_node")
    if state["tracer"] is None:
        raise HTTPException(status_code=404, detail="System not initialized.")
    result = state["tracer"].find_all_impacted_nodes(start_node)
    return result