# In src/core/traceability_optimizer.py
import networkx as nx
import random
import math
from typing import List, Dict, Any, Tuple

class TraceabilityOptimizer:
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph

    def find_shortest_path_dijkstra(self, source_id: str, target_id: str) -> Dict[str, Any]:
        """Uses Dijkstra's algorithm (part of NetworkX) for a baseline shortest path."""
        if source_id not in self.graph or target_id not in self.graph:
            return {"error": f"Source ({source_id}) or target ({target_id}) node not in graph."}
        try:
            path = nx.shortest_path(self.graph, source=source_id, target=target_id)
            return {"path": path, "length": len(path) - 1, "method": "Dijkstra"}
        except nx.NetworkXNoPath:
            return {"error": "No path found between the specified nodes."}
    
    def find_all_impacted_nodes(self, start_node: str) -> Dict[str, Any]:
        """Finds all nodes downstream from a starting node (what is impacted by a change)."""
        if start_node not in self.graph:
            return {"error": f"Node {start_node} not in graph."}
        
        # Using descendants to find all reachable nodes in a directed graph
        # This is a standard way to find downstream impact
        impacted_nodes = list(nx.descendants(self.graph, start_node))
        return {"start_node": start_node, "impacted_nodes": impacted_nodes, "count": len(impacted_nodes)}

    # --- Full Ant Colony Optimization ---
    def find_path_with_aco(self, 
                           source_id: str, 
                           target_id: str, 
                           n_ants: int = 20, 
                           n_iterations: int = 100, 
                           alpha: float = 1.0,  # Pheromone importance
                           beta: float = 2.0,   # Heuristic importance (e.g., confidence)
                           gamma: float = 0.5,  # Heuristic importance (e.g., inverse path length)
                           evaporation_rate: float = 0.1,
                           pheromone_deposit_factor: float = 1.0,
                           min_pheromone: float = 0.1,
                           max_pheromone: float = 10.0) -> Dict[str, Any]:
        """
        Finds a path using Ant Colony Optimization, incorporating multiple heuristics.
        Optimization Goal: Find paths that are both 'short' and 'confident'.
        """
        if source_id not in self.graph or target_id not in self.graph:
            return {"error": f"Source ({source_id}) or target ({target_id}) node not in graph."}

        # Initialize pheromones on all edges
        # We need to access edge data, so iterate correctly
        initial_pheromone = 1.0
        for u, v, key in self.graph.edges(keys=True):
            self.graph[u][v][key]['pheromone'] = initial_pheromone

        best_path = None
        best_path_quality = float('-inf') # Higher is better quality (e.g., sum of confidence / length)

        # Store a mapping of node IDs to the full document text (from the KGBuilder context if available)
        # This will be helpful for computing 'cost' if needed, but for now we focus on path quality.
        
        for iteration in range(n_iterations):
            all_paths_and_qualities: List[Tuple[List[str], float]] = [] # Stores (path, quality_score)

            for _ in range(n_ants):
                path, quality_score = self._construct_path_for_ant(source_id, target_id, alpha, beta, gamma)
                if path and quality_score > float('-inf'):
                    all_paths_and_qualities.append((path, quality_score))
                    # Update global best path based on its quality score
                    if quality_score > best_path_quality:
                        best_path_quality = quality_score
                        best_path = path
            
            # 2. Evaporate pheromones on all edges
            for u, v, key in self.graph.edges(keys=True):
                current_pheromone = self.graph[u][v][key]['pheromone']
                evaporated_pheromone = current_pheromone * (1.0 - evaporation_rate)
                self.graph[u][v][key]['pheromone'] = max(min_pheromone, evaporated_pheromone) # Ensure min_pheromone

            # 3. Deposit pheromones based on collected paths (only if paths were found)
            if all_paths_and_qualities:
                # Sort paths by quality_score for better pheromone distribution
                all_paths_and_qualities.sort(key=lambda x: x[1], reverse=True)

                for path, quality_score in all_paths_and_qualities:
                    # Deposit more pheromone for higher quality paths
                    # Normalizing quality_score for deposit amount
                    if quality_score > 0: # Ensure positive quality for deposit
                        deposit_amount = pheromone_deposit_factor * quality_score
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i+1]
                            # Find the specific edge (key) if it's a MultiDiGraph
                            # For simplicity, we assume one edge type between u,v here.
                            # In NetworkX MultiDiGraph, you need to iterate through keys or be specific.
                            # For simplicity, let's just pick the first edge if multiple exist between u,v.
                            edge_keys = list(self.graph[u][v].keys())
                            if edge_keys:
                                key = edge_keys[0] # Pick the first key for pheromone update
                                current_pheromone = self.graph[u][v][key]['pheromone']
                                new_pheromone = current_pheromone + deposit_amount
                                self.graph[u][v][key]['pheromone'] = min(max_pheromone, new_pheromone) # Ensure max_pheromone

        return {"path": best_path, "quality": best_path_quality, "method": "Ant Colony Optimization"}

    def _construct_path_for_ant(self, 
                                start_node: str, 
                                end_node: str, 
                                alpha: float, 
                                beta: float, 
                                gamma: float) -> Tuple[List[str], float]:
        """
        Constructs a path for a single ant, considering pheromone and multiple heuristics.
        Returns the path and its accumulated quality score.
        """
        path = [start_node]
        current_node = start_node
        path_quality_sum = 0.0 # Accumulate quality along the path
        
        visited_nodes = {start_node} # To prevent cycles

        while current_node != end_node:
            neighbors = list(self.graph.successors(current_node))
            if not neighbors:
                return [], float('-inf') # Dead end, path invalid
            
            transition_probabilities = []
            possible_next_nodes = []

            for neighbor in neighbors:
                # Iterate through all edges between current_node and neighbor
                for edge_key in self.graph[current_node][neighbor]:
                    edge_data = self.graph[current_node][neighbor][edge_key]
                    
                    pheromone = edge_data.get('pheromone', 1.0)
                    
                    # Heuristic 1: Relationship Confidence (from LLM extraction)
                    # Higher confidence is better, so we use it directly.
                    confidence_heuristic = edge_data.get('confidence', 0.5) # Default to 0.5 if not found
                    
                    # Heuristic 2: Inverse Path Length (preference for shorter paths)
                    # For a single step, this is less pronounced but contributes to overall path length.
                    # Or, one can use a constant 1.0 heuristic if only confidence matters per step.
                    length_heuristic = 1.0 # Or 1/distance if edge weights were distances.

                    # Combine heuristics. Beta for confidence, Gamma for length.
                    combined_heuristic = (confidence_heuristic ** beta) * (length_heuristic ** gamma)
                    
                    probability_component = (pheromone ** alpha) * combined_heuristic
                    
                    transition_probabilities.append(probability_component)
                    possible_next_nodes.append((neighbor, edge_data.get('confidence', 0.5))) # Store neighbor and its confidence for path quality

            # Normalize probabilities
            sum_probs = sum(transition_probabilities)
            if sum_probs == 0:
                return [], float('-inf') # No valid moves from this node

            normalized_probs = [p / sum_probs for p in transition_probabilities]
            
            # Choose next node based on probabilities
            next_node_info = random.choices(possible_next_nodes, weights=normalized_probs, k=1)[0]
            next_node = next_node_info[0]
            next_edge_confidence = next_node_info[1]

            if next_node in visited_nodes:
                return [], float('-inf') # Avoid cycles, path invalid
                
            path.append(next_node)
            visited_nodes.add(next_node)
            path_quality_sum += next_edge_confidence # Add confidence of chosen edge to path quality
            current_node = next_node
        
        # Calculate final path quality: e.g., sum of confidences / total length
        # This can be adjusted. For now, sum of confidences.
        final_path_quality = path_quality_sum # / (len(path) - 1) if len(path) > 1 else path_quality_sum

        return path, final_path_quality