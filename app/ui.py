# In app/ui.py
import streamlit as st
import requests
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# --- Configuration ---
API_URL = "http://127.0.0.1:8000"
st.set_page_config(layout="wide")

# --- Helper Functions ---
def visualize_graph(graph_data):
    G = nx.node_link_graph(graph_data)
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True, cdn_resources='in_line')
    net.from_nx(G)
    
    # Add titles to nodes for hover-over information
    for node in net.nodes:
        node_data = G.nodes[node['id']]
        title = f"ID: {node['id']}\nType: {node_data.get('type', 'N/A')}\nName: {node_data.get('name', '')}"
        node['title'] = title
        if node_data.get('type') == 'Feature':
            node['color'] = '#FF5733' # Orange
        elif node_data.get('type') == 'Requirement':
            node['color'] = '#33C1FF' # Blue

    net.show_buttons(filter_=['physics'])
    html = net.generate_html()
    return html

# --- Streamlit UI ---
st.title("🔬 RAG-Enhanced Requirements Engineering Experimental Workbench")

# Sidebar for controls
st.sidebar.header("Controls")
if st.sidebar.button("Load Data & Build KG"):
    with st.spinner("Processing documents and building Knowledge Graph... This may take several minutes."):
        try:
            res = requests.post(f"{API_URL}/api/load-and-process")
            if res.status_code == 200:
                st.sidebar.success(res.json()['message'])
                st.session_state['graph_built'] = True
            else:
                st.sidebar.error(f"Error: {res.text}")
        except requests.exceptions.ConnectionError:
            st.sidebar.error("Connection Error: Is the FastAPI backend running?")


# Main content area with tabs
if 'graph_built' in st.session_state and st.session_state['graph_built']:
    tab1, tab2, tab3 = st.tabs(["📊 Knowledge Graph Visualization", "⚔️ Conflict Resolution", "🔗 Traceability Analysis"])

    with tab1:
        st.header("Interactive Knowledge Graph")
        with st.spinner("Fetching and rendering graph..."):
            res = requests.get(f"{API_URL}/api/graph-data")
            if res.status_code == 200:
                graph_html = visualize_graph(res.json())
                components.html(graph_html, height=800)
            else:
                st.error("Failed to load graph data.")

    with tab2:
        st.header("Conflict Detection & Resolution")
        if st.button("Detect Potential Conflicts"):
            with st.spinner("Scanning for conflicts..."):
                res = requests.get(f"{API_URL}/api/conflicts/detect")
                if res.status_code == 200:
                    st.session_state['conflicts'] = res.json()['conflicts']
                    st.success(f"Detected {len(st.session_state['conflicts'])} potential conflicts.")
                else:
                    st.error("Failed to detect conflicts.")
        
        if 'conflicts' in st.session_state and st.session_state['conflicts']:
            st.subheader("Detected Conflicts")
            for i, conflict in enumerate(st.session_state['conflicts']):
                with st.expander(f"Conflict {i+1}: `{conflict['req1']}` vs `{conflict['req2']}`"):
                    st.write(f"**Reason:** {conflict['reason']}")
                    if st.button(f"Resolve Conflict {i+1}", key=f"resolve_{i}"):
                        with st.spinner("Running metaheuristic to find resolution..."):
                            res = requests.post(f"{API_URL}/api/conflicts/resolve/{i}")
                            if res.status_code == 200:
                                resolution = res.json()
                                st.info(f"**Chosen Strategy:** {resolution['chosen_strategy']} (Score: {resolution['score']:.2f})")
                                st.markdown("**Suggested Action:**")
                                st.text_area("Suggestion", value=resolution['suggestion'], height=200, disabled=True, key=f"sugg_{i}")
                            else:
                                st.error("Failed to resolve.")

    with tab3:
        st.header("Traceability & Impact Analysis")
        graph_data = requests.get(f"{API_URL}/api/graph-data").json()
        node_ids = [node['id'] for node in graph_data['nodes']]
        start_node = st.selectbox("Select a start node to analyze its impact:", options=node_ids)
        
        if st.button("Analyze Impact"):
            with st.spinner("Tracing dependencies..."):
                res = requests.post(f"{API_URL}/api/traceability/find-impact", json={"start_node": start_node})
                if res.status_code == 200:
                    result = res.json()
                    st.success(f"Node `{result['start_node']}` impacts **{result['count']}** other nodes.")
                    st.json(result['impacted_nodes'])
                else:
                    st.error("Failed to run analysis.")

else:
    st.info("Welcome! Please click the 'Load Data & Build KG' button in the sidebar to begin the experiment.")