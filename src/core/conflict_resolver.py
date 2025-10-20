# In src/core/conflict_resolver.py
import networkx as nx
from .llm_handler import LLMHandler
import random
import math
from typing import List, Dict, Tuple, Any

class ConflictResolver:
    def __init__(self, graph: nx.MultiDiGraph, llm_handler: LLMHandler, documents: list):
        self.graph = graph
        self.llm = llm_handler
        
        doc_map = {}
        for doc in documents:
            try:
                first_line = doc.split('\n')[0]
                parts = first_line.split(': ', 1)
                if len(parts) == 2:
                    doc_id = parts[1].strip()
                    doc_map[doc_id] = doc
            except IndexError:
                print(f"Warning: Could not parse document ID from line: '{first_line}'. Skipping.")
        self.documents = doc_map
        self.resolution_strategies = ["Prioritize_Req1", "Prioritize_Req2", "Rephrase_Req1", "Rephrase_Req2", "Merge_Reqs", "Split_Reqs"] # Added Split_Reqs

    def detect_conflicts(self) -> list:
        conflicts = []
        # Find explicit "conflicts_with" relationships
        for u, v, data in self.graph.edges(data=True):
            if data['type'] == 'conflicts_with':
                conflicts.append({
                    "req1": u, "req2": v, 
                    "reason": data.get('reason', 'KG extraction'),
                    "status": "unresolved"
                })

        # Use LLM to check requirements that modify the same feature (or other semantic clashes)
        # We can also use relationship confidence here.
        potential_clashes = set() # To store (req_id1, req_id2) pairs to avoid redundant LLM checks

        # Check for requirements impacting the same feature with low confidence "conflicts_with" or no explicit conflict
        features = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'Feature']
        for feature in features:
            related_reqs = list(self.graph.predecessors(feature)) 
            if len(related_reqs) > 1:
                for i in range(len(related_reqs)):
                    for j in range(i + 1, len(related_reqs)):
                        req1_id = related_reqs[i]
                        req2_id = related_reqs[j]
                        if tuple(sorted((req1_id, req2_id))) not in potential_clashes:
                            if self._llm_check_conflict(req1_id, req2_id):
                                conflicts.append({
                                    "req1": req1_id, "req2": req2_id,
                                    "reason": f"Semantic clash detected: Both modify feature '{feature}' and LLM flagged potential conflict.",
                                    "status": "unresolved"
                                })
                            potential_clashes.add(tuple(sorted((req1_id, req2_id))))
        
        # Consider using the 'conflicts_with' relationship from KG extraction here too,
        # especially if LLM-generated relationships have confidence.
        # For simplicity, current conflicts are direct.
        return conflicts

    def _llm_check_conflict(self, req1_id: str, req2_id: str) -> bool:
        """Asks the LLM if two requirement texts conflict, returning a boolean."""
        doc1 = self.documents.get(req1_id, "")
        doc2 = self.documents.get(req2_id, "")
        if not doc1 or not doc2:
            return False

        prompt = f"""
        Analyze the following two software requirements from their full text. Do they conflict with each other?
        A conflict means they cannot both be implemented as written (e.g., one states "background is blue", the other states "background is green" for the same UI element).
        
        Requirement 1:
        {doc1}
        
        Requirement 2:
        {doc2}
        
        Answer with only "YES" or "NO".
        """
        try:
            # Using the general generate method for simple text output
            response = self.llm.generate(prompt).strip().upper()
            return "YES" in response
        except Exception as e:
            print(f"Error in LLM conflict check for {req1_id} and {req2_id}: {e}")
            return False # Default to no conflict on error

    def _fitness_function(self, conflict: dict, strategy: str) -> float:
        """
        Evaluates a resolution strategy using the LLM. Higher is better.
        This is a heuristic and a key part of the experimental design.
        """
        req1_id, req2_id = conflict['req1'], conflict['req2']
        doc1 = self.documents.get(req1_id, "")
        doc2 = self.documents.get(req2_id, "")

        # If docs are missing, we cannot evaluate, return minimal score
        if not doc1 or not doc2:
            return 0.0
            
        prompt = f"""
        Given two conflicting requirements ({req1_id}, {req2_id}) and a proposed resolution strategy '{strategy}',
        evaluate the quality of this strategy on a scale of 1 to 10, where 1 is a very poor strategy and 10 is an excellent one.
        Consider factors like data loss, implementation complexity, and stakeholder satisfaction.

        Strategy: {strategy} Definitions:
        - Prioritize_Req1: Fully implement Requirement 1, ignore conflicting parts of Requirement 2.
        - Prioritize_Req2: Fully implement Requirement 2, ignore conflicting parts of Requirement 1.
        - Rephrase_Req1: Rewrite Requirement 1 to align with Requirement 2.
        - Rephrase_Req2: Rewrite Requirement 2 to align with Requirement 1.
        - Merge_Reqs: Create a new, unified requirement combining ideas from both, resolving the conflict.
        - Split_Reqs: Break down one or both requirements into smaller, non-conflicting parts.
        
        Return ONLY the integer score.
        Score: 
        """
        try:
            # Using the general generate method for simple text output
            response = self.llm.generate(prompt, temperature=0.1, max_tokens=10).strip() # Small max_tokens for just a score
            return float(response)
        except Exception as e:
            print(f"Error in LLM fitness function for {req1_id}, {req2_id} with strategy {strategy}: {e}")
            return 0.0 # Failed to get a score

    def _get_llm_suggestion(self, conflict: dict, strategy: str) -> str:
        """Generates a human-readable suggestion based on the chosen strategy."""
        doc1 = self.documents.get(conflict['req1'], "")
        doc2 = self.documents.get(conflict['req2'], "")
        if not doc1 or not doc2:
            return f"Cannot generate suggestion: Documents for {conflict['req1']} or {conflict['req2']} not found."

        prompt = f"""
        **Task:** Generate a specific, actionable suggestion to resolve the conflict between two requirements using the chosen strategy.
        Be concise and focused on the technical implications.

        **Requirement 1 ({conflict['req1']}):**
        {doc1}

        **Requirement 2 ({conflict['req2']}):**
        {doc2}

        **Chosen Strategy:** {strategy} (Definitions: {', '.join(self.resolution_strategies)})

        **Suggestion:**
        """
        try:
            # Using the general generate method for text output, higher max_tokens
            return self.llm.generate(prompt, temperature=0.5, max_tokens=512)
        except Exception as e:
            print(f"Error generating LLM suggestion for {conflict['req1']} and {conflict['req2']}: {e}")
            return f"Failed to generate suggestion due to LLM error: {e}"

    # --- Hill Climbing (Existing) ---
    def resolve_with_hill_climbing(self, conflict: dict, max_iterations: int = 20) -> Dict[str, Any]:
        """Uses a Hill Climbing algorithm to find a good resolution strategy."""
        
        current_solution = random.choice(self.resolution_strategies)
        current_score = self._fitness_function(conflict, current_solution)
        
        for _ in range(max_iterations):
            neighbors = [s for s in self.resolution_strategies if s != current_solution]
            if not neighbors: break
            
            best_neighbor = max(neighbors, key=lambda s: self._fitness_function(conflict, s))
            best_neighbor_score = self._fitness_function(conflict, best_neighbor)

            if best_neighbor_score > current_score:
                current_solution = best_neighbor
                current_score = best_neighbor_score
            else:
                break # Reached a local maximum
        
        suggestion = self._get_llm_suggestion(conflict, current_solution)
        return {"chosen_strategy": current_solution, "suggestion": suggestion, "score": current_score, "method": "Hill Climbing"}

    # --- Simulated Annealing ---
    def resolve_with_simulated_annealing(self, conflict: dict,
                                        max_iterations: int = 100,
                                        initial_temperature: float = 10.0,
                                        cooling_rate: float = 0.95) -> Dict[str, Any]:
        """Uses Simulated Annealing to find a good resolution strategy."""
        current_solution = random.choice(self.resolution_strategies)
        current_score = self._fitness_function(conflict, current_solution)
        
        best_solution = current_solution
        best_score = current_score
        
        temperature = initial_temperature

        for i in range(max_iterations):
            neighbor_solution = random.choice([s for s in self.resolution_strategies if s != current_solution])
            neighbor_score = self._fitness_function(conflict, neighbor_solution)
            
            # Calculate acceptance probability
            if neighbor_score > current_score:
                # Accept better solution
                current_solution = neighbor_solution
                current_score = neighbor_score
            else:
                # Accept worse solution with a probability
                acceptance_prob = math.exp((neighbor_score - current_score) / temperature)
                if random.random() < acceptance_prob:
                    current_solution = neighbor_solution
                    current_score = neighbor_score
            
            # Update best solution found so far
            if current_score > best_score:
                best_solution = current_solution
                best_score = current_score
            
            temperature *= cooling_rate # Cool down
            if temperature < 0.1: # Minimum temperature to prevent division by zero / very small numbers
                temperature = 0.1

        suggestion = self._get_llm_suggestion(conflict, best_solution)
        return {"chosen_strategy": best_solution, "suggestion": suggestion, "score": best_score, "method": "Simulated Annealing"}

    # --- Genetic Algorithm ---
    # For a simple list of strategies, GA might be overkill, but useful for demonstration.
    # Each "individual" in population is a single strategy in this simplified version.
    def resolve_with_genetic_algorithm(self, conflict: dict,
                                       population_size: int = 5,
                                       num_generations: int = 10,
                                       mutation_rate: float = 0.2) -> Dict[str, Any]:
        """Uses a simplified Genetic Algorithm to find a good resolution strategy."""
        
        # 1. Initialize population (random strategies)
        population = [random.choice(self.resolution_strategies) for _ in range(population_size)]
        
        best_overall_solution = None
        best_overall_score = -float('inf')

        for generation in range(num_generations):
            # 2. Evaluate fitness for each individual in the population
            fitness_scores = [(strategy, self._fitness_function(conflict, strategy)) for strategy in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True) # Sort by score (fittest first)

            current_best_solution, current_best_score = fitness_scores[0]

            if current_best_score > best_overall_score:
                best_overall_score = current_best_score
                best_overall_solution = current_best_solution
            
            # 3. Selection (Elitism: keep the best, then roulette wheel or tournament)
            # For simplicity: Keep the top N% and fill the rest randomly/with mutation
            num_elite = max(1, population_size // 5)
            new_population = [s for s, _ in fitness_scores[:num_elite]]
            
            # Fill the rest with mutated versions or new random ones
            while len(new_population) < population_size:
                parent = random.choice([s for s,_ in fitness_scores]) # Select a parent (can be same as elite)
                
                # Mutation (replaces the "crossover" for discrete strategies)
                if random.random() < mutation_rate:
                    mutated_child = random.choice([s for s in self.resolution_strategies if s != parent])
                else:
                    mutated_child = parent
                
                new_population.append(mutated_child)
            
            population = new_population

        suggestion = self._get_llm_suggestion(conflict, best_overall_solution)
        return {"chosen_strategy": best_overall_solution, "suggestion": suggestion, "score": best_overall_score, "method": "Genetic Algorithm"}