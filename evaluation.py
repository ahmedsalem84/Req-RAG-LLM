# In evaluation.py
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
import networkx as nx
import time
import os,sys
import random
import numpy as np
from scipy.stats import wilcoxon
from typing import List,Dict,Any
from collections import defaultdict # Used for conflict_resolver_metrics

# --- Configuration ---
USE_LOCAL_MODEL = True
MODEL_REPO = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF" # Your Llama 3 model
MODEL_FILENAME = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

# Experiments parameters
RAG_K_VALUES = [1, 5, 10]
SCALING_DOC_COUNTS = [10, 20, 50, 100] # Number of documents for scalability study
EVAL_DOC_LIMIT = 50  # Max documents to fetch/process for robust evaluation runs. Used for experiments.

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
CORE_DIR = os.path.join(SRC_DIR, "core")
BASELINES_DIR = os.path.join(SRC_DIR, "baselines")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

for d in [RESULTS_DIR, DATA_DIR, MODELS_DIR, BASELINES_DIR, CORE_DIR]: # Ensure all necessary dirs exist
    os.makedirs(d, exist_ok=True)

# --- Import core and baseline modules ---
from src.core.data_loader import load_github_issues, process_github_issues_for_rag
from src.core.llm_handler import LLMHandler
from src.core.kg_builder import KGBuilder
from src.core.conflict_resolver import ConflictResolver
from src.core.traceability_optimizer import TraceabilityOptimizer # For ACO
from src.baselines.extractors import bm25_baseline_extractor, spacy_baseline_extractor, hf_ner_baseline_extractor, rule_based_conflict_detector, get_doc_id

# --- Dataset Definitions (ALL GitHub now) ---
ALL_GITHUB_DATASETS_CONFIG = {
    "apache_superset": {
        "repo": "apache/superset",
        "clean_path": os.path.join(DATA_DIR, "apache_superset_issues_clean.json"),
        "doc_limit": 50 # For GT annotation.
    },
    "apache_airflow": {
        "repo": "apache/airflow",
        "clean_path": os.path.join(DATA_DIR, "apache_airflow_issues_clean.json"),
        "doc_limit": 50
    },
    "tensorflow_tensorflow": {
        "repo": "tensorflow/tensorflow",
        "clean_path": os.path.join(DATA_DIR, "tensorflow_tensorflow_issues_clean.json"),
        "doc_limit": 50
    }
}
# There are no PROMISE datasets anymore, so ALL_DATASETS_CONFIG is just GitHub
ALL_DATASETS_CONFIG = ALL_GITHUB_DATASETS_CONFIG 


# --- Helper for Ground Truth Generation ---
def generate_ground_truth_for_dataset(dataset_name: str, documents: List[str], gt_output_path: str, num_records: int) -> str:
    if os.path.exists(gt_output_path):
        print(f"\n--- Ground Truth for {dataset_name} already exists at '{gt_output_path}'. Assuming it is manually annotated. ---")
        return gt_output_path
        
    print(f"\n--- Generating NEW Placeholder Ground Truth for {dataset_name} ({num_records} records) ---")
    print(f"Please manually annotate the file '{gt_output_path}' with entities, relationships, and conflicts.")

    gt_entities = []
    for i, doc in enumerate(documents[:num_records]):
        doc_id = get_doc_id(doc)
        if doc_id:
            # For GT placeholder, just Requirement ID, name (title), and a default confidence
            gt_entities.append({"id": doc_id, "type": "Requirement", "name": doc.split('\n')[0].replace("Requirement ID: ", "").strip(), "confidence": 1.0})

    ground_truth_data = {
        "entities": gt_entities,
        "relationships": [],
        "conflicts": []
    }

    with open(gt_output_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth_data, f, indent=2)
    
    print(f"NEW Placeholder Ground Truth for {dataset_name} saved to {gt_output_path}. **REQUIRES MANUAL ANNOTATION.**")
    return gt_output_path


# --- LLM Experiment Runner Functions ---
def run_llm_no_rag(docs: List[str], llm_handler: LLMHandler, kg_builder: KGBuilder) -> nx.MultiDiGraph:
    print("Running LLM Only (No RAG)")
    return kg_builder.build_graph_from_documents(docs, use_rag=False, k=0)

def run_llm_rag(docs: List[str], llm_handler: LLMHandler, kg_builder: KGBuilder, k: int = 5) -> nx.MultiDiGraph:
    print(f"Running RAG-LLM (k={k})")
    return kg_builder.build_graph_from_documents(docs, use_rag=True, k=k)

# --- Evaluation Utilities ---
def evaluate_graph(pred_G: nx.MultiDiGraph, gt_path: str) -> Dict[str, float]:
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    
    true_rels = {(r['source'], r['target'], r['type']) for r in gt.get("relationships", []) if r.get('source') and r.get('target') and r.get('type')}
    pred_rels = {(u, v, d.get("type","unknown")) for u,v,d in pred_G.edges(data=True) if u and v and d.get('type')}
    
    # --- NEW DEBUG PRINTS FOR evaluate_graph ---
    print(f"  DEBUG EVAL GRAPH: True Rels ({len(true_rels)}): {list(true_rels)[:5]}") # Show count
    print(f"  DEBUG EVAL GRAPH: Pred Rels ({len(pred_rels)}): {list(pred_rels)[:5]}") # Show count
    # --- END DEBUG PRINTS ---

    all_rels = list(true_rels.union(pred_rels))
    if not all_rels:
        return {"precision":0.0,"recall":0.0,"f1_score":0.0}
    
    y_true = [1 if r in true_rels else 0 for r in all_rels]
    y_pred = [1 if r in pred_rels else 0 for r in all_rels]
    p,r,f,_ = precision_recall_fscore_support(y_true,y_pred, average='binary', zero_division=0)
    
    return {"precision":p,"recall":r,"f1_score":f}

def evaluate_conflicts(predicted_conflicts: List[Dict], gt_path: str) -> Dict[str, float]:
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    
    true_conflict_pairs = {tuple(sorted((c['req1'], c['req2']))) for c in gt.get("conflicts", [])}
    pred_conflict_pairs = {tuple(sorted((c['req1'], c['req2']))) for c in predicted_conflicts}

    # --- NEW DEBUG PRINTS FOR evaluate_conflicts ---
    print(f"  DEBUG EVAL CONFLICTS: True Conflicts ({len(true_conflict_pairs)}): {list(true_conflict_pairs)[:5]}") # Show count
    print(f"  DEBUG EVAL CONFLICTS: Pred Conflicts ({len(pred_conflict_pairs)}): {list(pred_conflict_pairs)[:5]}") # Show count
    # --- END DEBUG PRINTS ---

    tp = len(pred_conflict_pairs.intersection(true_conflict_pairs))
    fp = len(pred_conflict_pairs.difference(true_conflict_pairs))
    fn = len(true_conflict_pairs.difference(pred_conflict_pairs))
    
    conflict_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    conflict_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    conflict_f1 = (2 * conflict_precision * conflict_recall) / (conflict_precision + conflict_recall) if (conflict_precision + conflict_recall) > 0 else 0

    return {"precision": conflict_precision, "recall": conflict_recall, "f1_score": conflict_f1}

def graph_struct_stats(G: nx.MultiDiGraph) -> Dict[str, Any]:
    if G.number_of_nodes()==0:
        return {"nodes":0,"edges":0,"avg_degree":0,"density":0,"avg_clustering":0}
    
    undirected_G = G.to_undirected() 
    
    avg_deg = sum(dict(G.degree()).values()) / G.number_of_nodes()
    
    try:
        clustering_values = nx.clustering(undirected_G)
        avg_clust = sum(clustering_values.values()) / len(clustering_values) if clustering_values else 0
    except Exception:
        avg_clust = 0
    
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "avg_degree": avg_deg,
        "density": nx.density(G),
        "avg_clustering": avg_clust
    }

# --- Main Experiment Orchestration ---
def run_all_experiments():
    all_results_df = pd.DataFrame()
    all_scaling_records = []
    
    print("\n--- Preparing Datasets ---")
    datasets = {} # Stores {'dataset_name': {'docs': [], 'gt_path': '', 'docs_map': {}}}
    
    # --- Process all GitHub datasets ---
    for ds_name_key, ds_info in ALL_GITHUB_DATASETS_CONFIG.items(): # Use ALL_GITHUB_DATASETS_CONFIG
        clean_issues_file_path = ds_info['clean_path']
        repo_owner_slash_name = ds_info['repo']
        
        print(f"Loading cleaned GitHub issues for {repo_owner_slash_name} from {clean_issues_file_path}...")
        
        clean_docs_full = []
        if os.path.exists(clean_issues_file_path) and os.path.getsize(clean_issues_file_path) > 0:
            with open(clean_issues_file_path, 'r', encoding='utf-8') as f:
                clean_docs_full = json.load(f)
            print(f"  Loaded {len(clean_docs_full)} cleaned documents for {repo_owner_slash_name}.")
        else:
            print(f"  Warning: Cleaned issues file not found or empty for {repo_owner_slash_name} at {clean_issues_file_path}. Please run clean_github_data.py first to generate it.")
            continue

        current_docs_for_experiment = clean_docs_full[:EVAL_DOC_LIMIT]
        current_docs_for_experiment = [doc for doc in current_docs_for_experiment if doc.strip()]

        datasets[ds_name_key] = {
            'docs': current_docs_for_experiment,
            'gt_path': generate_ground_truth_for_dataset(ds_name_key, current_docs_for_experiment, 
                                                         os.path.join(DATA_DIR, f"{ds_name_key.replace('/', '_')}_gt.json"),
                                                         num_records=ds_info['doc_limit']),
            'docs_map': {get_doc_id(doc): doc for doc in current_docs_for_experiment}
        }
        print(f"  {ds_name_key} prepared: {len(current_docs_for_experiment)} documents for experiments.")

    # --- No PROMISE PITS Dataset anymore, as per updated strategy ---

    # --- Initialize LLM handler and KG builder ---
    llm_handler = None
    kg_builder = None
    LLM_SUPPORTED = False
    if USE_LOCAL_MODEL:
        model_file_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
        if not os.path.exists(model_file_path):
            print("Attempting to download model via huggingface_hub (may fail if not allowed)...")
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, local_dir=MODELS_DIR, local_dir_use_symlinks=False)
                print("Model downloaded.")
            except Exception as e:
                print(f"Model download failed: {e}")
        if os.path.exists(model_file_path):
            try:
                llm_handler = LLMHandler(model_path=model_file_path)
                kg_builder = KGBuilder(llm_handler)
                LLM_SUPPORTED = True
                print("LLM handler initialized.")
            except Exception as e:
                print(f"LLM init failed: {e}")
                LLM_SUPPORTED = False
        else:
            print("Model file not present; skipping LLM experiments.")
    else:
        print("USE_LOCAL_MODEL=False -> skipping LLM experiments.")

    # Define experiment methods
    methods = [
        ("BM25 (IR Baseline)", bm25_baseline_extractor),
        ("spaCy (NLP Baseline)", spacy_baseline_extractor),
        ("BERT NER (NLP Baseline)", hf_ner_baseline_extractor)
    ]
    if LLM_SUPPORTED:
        methods += [
            ("LLM Only (No RAG)", lambda docs: run_llm_no_rag(docs, llm_handler, kg_builder)),
            ("RAG-LLM (k=1)", lambda docs: run_llm_rag(docs, llm_handler, kg_builder, k=1)),
            ("RAG-LLM (k=5)", lambda docs: run_llm_rag(docs, llm_handler, kg_builder, k=5)),
            ("RAG-LLM (k=10)", lambda docs: run_llm_rag(docs, llm_handler, kg_builder, k=10)),
            # NEW: Add different metaheuristic resolvers for conflict detection, they will use RAG-LLM (k=5) as base
            ("RAG-LLM (k=5) + Hill Climbing", lambda docs: run_llm_rag(docs, llm_handler, kg_builder, k=RAG_K_VALUES[1])),
            ("RAG-LLM (k=5) + Simulated Annealing", lambda docs: run_llm_rag(docs, llm_handler, kg_builder, k=RAG_K_VALUES[1])),
            ("RAG-LLM (k=5) + Genetic Algorithm", lambda docs: run_llm_rag(docs, llm_handler, kg_builder, k=RAG_K_VALUES[1]))
        ]

    graphs = {}
    conflict_resolver_metrics = defaultdict(list)

    # --- Main Experiment Loop (Iterate through all datasets) ---
    for ds_name, ds_data in datasets.items():
        eval_docs = ds_data['docs']
        gt_path = ds_data['gt_path']
        
        print(f"\n--- Running Main Experiments for Dataset: {ds_name} ({len(eval_docs)} documents) ---")
        current_ds_results = []
        
        for name, func in methods:
            print(f"\n  Running: {name}")
            start = time.time()
            
            G = nx.MultiDiGraph() # Initialize an empty graph for each run
            predicted_conflicts = []
            avg_resolution_score = 0 # Initialize for metaheuristic metrics

            # --- NEW: Define these flags clearly at the start of each method's run ---
            is_llm_only_or_rag = "LLM Only" in name or "RAG-LLM" in name
            is_hill_climbing = "Hill Climbing" in name
            is_simulated_annealing = "Simulated Annealing" in name
            is_genetic_algorithm = "Genetic Algorithm" in name
            is_any_llm_or_metaheuristic = is_llm_only_or_rag or is_hill_climbing or is_simulated_annealing or is_genetic_algorithm
            # --- END NEW FLAGS ---


            try:
                # Check LLM support for all LLM-based and metaheuristic methods
                if is_any_llm_or_metaheuristic and not LLM_SUPPORTED:
                    print("    LLM not supported, skipping.")
                    # Ensure G and predicted_conflicts are empty for skipped runs
                    G = nx.MultiDiGraph()
                    predicted_conflicts = []
                else:
                    # --- Logic for Metaheuristic Conflict Resolution ---
                    if is_hill_climbing or is_simulated_annealing or is_genetic_algorithm:
                        # For these, first get the base RAG-LLM(k=5) graph to find conflicts
                        print(f"Running base RAG-LLM (k={RAG_K_VALUES[1]}) for {name} graph extraction...")
                        G = run_llm_rag(eval_docs, llm_handler, kg_builder, k=RAG_K_VALUES[1])
                        
                        resolver_for_metaheuristic = ConflictResolver(G, llm_handler, eval_docs)
                        base_conflicts_to_resolve = resolver_for_metaheuristic.detect_conflicts()
                        predicted_conflicts = base_conflicts_to_resolve # F1 score is on *detected* conflicts
                        
                        resolved_conflicts_details = []
                        if base_conflicts_to_resolve:
                            for conflict in base_conflicts_to_resolve:
                                if is_hill_climbing:
                                    resolved_conflicts_details.append(resolver_for_metaheuristic.resolve_with_hill_climbing(conflict))
                                elif is_simulated_annealing:
                                    resolved_conflicts_details.append(resolver_for_metaheuristic.resolve_with_simulated_annealing(conflict))
                                elif is_genetic_algorithm:
                                    resolved_conflicts_details.append(resolver_for_metaheuristic.resolve_with_genetic_algorithm(conflict))
                            
                            avg_resolution_score = np.mean([c['score'] for c in resolved_conflicts_details]) if resolved_conflicts_details else 0
                        else:
                            avg_resolution_score = 0
                        
                        conflict_resolver_metrics[ds_name].append({
                            "Method": name, 
                            "Conflicts_Detected": len(predicted_conflicts), 
                            "Avg_Resolution_Score": avg_resolution_score
                        })
                        
                    elif is_llm_only_or_rag: # This covers "LLM Only" and "RAG-LLM (k=X)"
                        G = func(eval_docs) # Run the LLM graph extractor
                        resolver = ConflictResolver(G, llm_handler, eval_docs)
                        predicted_conflicts = resolver.detect_conflicts()
                    else: # Baselines (BM25, spaCy, BERT NER)
                        G = func(eval_docs) # Run baseline graph extractor
                        predicted_conflicts = rule_based_conflict_detector(eval_docs) # Baselines use rule-based conflict detector

            except Exception as e:
                print(f"  Experiment {name} failed for {ds_name}: {e}")
                import traceback
                traceback.print_exc()
                G = nx.MultiDiGraph()
                predicted_conflicts = []
                avg_resolution_score = 0 # Ensure this is also reset on error
            end = time.time()
            
            time_per_doc = (end-start)/len(eval_docs) if eval_docs else 0

            metrics_kg = evaluate_graph(G, gt_path)
            metrics_conflict = evaluate_conflicts(predicted_conflicts, gt_path)
            
            struct = graph_struct_stats(G)
            
            run_record = {
                "Dataset": ds_name,
                "Method": name,
                "KG_Precision": metrics_kg["precision"],
                "KG_Recall": metrics_kg["recall"],
                "KG_F1_Score": metrics_kg["f1_score"],
                "Conflict_Precision": metrics_conflict["precision"],
                "Conflict_Recall": metrics_conflict["recall"],
                "Conflict_F1_Score": metrics_conflict["f1_score"],
                "Time_Per_Doc_s": time_per_doc,
                "Nodes": struct["nodes"],
                "Edges": struct["edges"],
                "Avg_Degree": struct["avg_degree"],
                "Density": struct["density"],
                "Avg_Clustering": struct["avg_clustering"]
            }
            current_ds_results.append(run_record)
            
            graphs[f"{ds_name}_{name.replace(' ', '_')}"] = G
            print(f"  -> {name}: KG_F1={metrics_kg['f1_score']:.3f}, Conflict_F1={metrics_conflict['f1_score']:.3f}, Time/Doc={time_per_doc:.2f}s, Nodes={struct['nodes']}")
        
        all_results_df = pd.concat([all_results_df, pd.DataFrame(current_ds_results)], ignore_index=True)


    # --- Ablation: RAG k sweep (if LLM supported) ---
    if LLM_SUPPORTED:
        print("\n--- Running Ablation Study: RAG k-value ---")
        ablation_records = []
        # Use the first GitHub dataset as the primary for ablation due to its complexity and availability
        primary_ablation_scaling_dataset_name = list(ALL_GITHUB_DATASETS_CONFIG.keys())[0] # Changed to ALL_GITHUB_DATASETS_CONFIG
        ablation_docs = datasets[primary_ablation_scaling_dataset_name]['docs']
        ablation_gt_path = datasets[primary_ablation_scaling_dataset_name]['gt_path']

        for k_val in RAG_K_VALUES:
            name = f"RAG-LLM (k={k_val})"
            print(f"\n  Ablation run: {name}")
            start=time.time()
            try:
                G = run_llm_rag(ablation_docs, llm_handler, kg_builder, k=k_val)
                resolver = ConflictResolver(G, llm_handler, ablation_docs)
                predicted_conflicts = resolver.detect_conflicts()
            except Exception as e:
                print(f"  Ablation failed: {e}")
                import traceback
                traceback.print_exc()
                G = nx.MultiDiGraph(); predicted_conflicts = []
            end=time.time()
            metrics_kg = evaluate_graph(G, ablation_gt_path)
            metrics_conflict = evaluate_conflicts(predicted_conflicts, ablation_gt_path)
            
            ablation_records.append({
                "k": k_val, "Dataset": primary_ablation_scaling_dataset_name,
                "KG_F1": metrics_kg["f1_score"], "Conflict_F1": metrics_conflict["f1_score"],
                "Time_Per_Doc_s":(end-start)/len(ablation_docs) if ablation_docs else 0,
                "Nodes":G.number_of_nodes(), "Edges":G.number_of_edges()
            })
        pd.DataFrame(ablation_records).to_csv(os.path.join(RESULTS_DIR,"ablation_k_sweep.csv"), index=False)
        print("Saved ablation_k_sweep.csv")

    # --- Scalability sweep: vary doc count and measure time/F1 ---
    print("\n--- Running Scalability Study ---")
    for n_docs_count in SCALING_DOC_COUNTS:
        # Use the primary dataset for scalability
        subset = datasets[primary_ablation_scaling_dataset_name]['docs'][:min(n_docs_count, len(datasets[primary_ablation_scaling_dataset_name]['docs']))]
        if not subset: continue
        print(f"\n  Scalability run: n_docs={len(subset)}")
        
        for (name, func) in methods:
            is_llm_method = "LLM" in name or "Hill Climbing" in name or "Simulated Annealing" in name or "Genetic Algorithm" in name
            if is_llm_method and not LLM_SUPPORTED:
                print(f"    LLM not supported, skipping {name}.")
                continue

            try:
                start=time.time()
                G = func(subset)
                # For scalability, conflict detection is simplified or skipped for speed
                # We'll just evaluate KG extraction here for brevity.
                end=time.time()
            except Exception as e:
                print(f"  Scaling run {name} failed: {e}")
                import traceback
                traceback.print_exc()
                G = nx.MultiDiGraph(); end=time.time()
            
            metrics_kg = evaluate_graph(G, datasets[primary_ablation_scaling_dataset_name]['gt_path'])
            
            all_scaling_records.append({
                "Method": name, "n_docs": len(subset), "Dataset": primary_ablation_scaling_dataset_name,
                "KG_F1":metrics_kg["f1_score"],
                "Time_s": (end-start), "Time_Per_Doc_s": (end-start)/len(subset) if subset else 0
            })
    pd.DataFrame(all_scaling_records).to_csv(os.path.join(RESULTS_DIR,"scaling_results.csv"), index=False)
    print("Saved scaling_results.csv")

    # Save main results
    all_results_df.to_csv(os.path.join(RESULTS_DIR,"main_results.csv"), index=False)
    print("\n--- All experiments complete. Results saved in the /results directory. ---")
    
    # Store conflict resolver metrics
    if conflict_resolver_metrics:
        flat_conflict_metrics = []
        for ds, metrics_list in conflict_resolver_metrics.items():
            for metric_item in metrics_list:
                flat_conflict_metrics.append({**metric_item, "Dataset": ds})
        df_conflict_resolver = pd.DataFrame(flat_conflict_metrics)
        df_conflict_resolver.to_csv(os.path.join(RESULTS_DIR,"conflict_resolver_metrics.csv"), index=False)
        print("Saved conflict_resolver_metrics.csv")

    return all_results_df, graphs, datasets, LLM_SUPPORTED

# --- Stage 7: Statistical testing (Wilcoxon) ---
def run_statistical_tests(all_results_df: pd.DataFrame, all_datasets: Dict, llm_supported_status: bool):
    stat_results = {}
    if not llm_supported_status:
        print("LLM not supported; skipping Wilcoxon test.")
        return stat_results

    # Assuming we want to compare LLM-only vs RAG-LLM (k=5) across all datasets
    llm_only_f1 = all_results_df[(all_results_df['Method'] == 'LLM Only (No RAG)')]['KG_F1_Score'].tolist()
    rag_llm_f1 = all_results_df[(all_results_df['Method'] == 'RAG-LLM (k=5)')]['KG_F1_Score'].tolist()

    if len(llm_only_f1) > 1 and len(llm_only_f1) == len(rag_llm_f1):
        try:
            stat, p = wilcoxon(llm_only_f1, rag_llm_f1)
            stat_results = {"stat": float(stat), "p_value": float(p)}
            print(f"\nWilcoxon test between LLM-only and RAG-LLM (k=5) across all datasets: stat={stat:.4f}, p={p:.6f}")
        except Exception as e:
            print(f"Wilcoxon test failed: {e}. Not enough valid data points or ranks.")
    else:
        print("Not enough paired F1 scores for Wilcoxon test (LLM-only vs RAG-LLM k=5).")
    
    return stat_results

# --- Stage 8: Figures & Tables ---
from pyvis.network import Network
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots_and_tables(df_results: pd.DataFrame, all_graphs: Dict, stat_results: Dict):
    sns.set(style="whitegrid")

    if df_results.empty:
        print("No numeric results to plot.")
        return

    # --- Table: Main Results ---
    df_results.to_csv(os.path.join(RESULTS_DIR,"table_main_results.csv"), index=False)
    print("\nTable: Main Results")
    print(df_results)

    # --- Figure: F1 Score Comparison Across Methods (per dataset) ---
    for dataset in df_results['Dataset'].unique():
        plt.figure(figsize=(14,7))
        sns.barplot(y="KG_F1_Score", x="Method", data=df_results[df_results['Dataset']==dataset], palette="viridis")
        plt.xticks(rotation=45, ha='right')
        plt.title(f"KG Extraction F1 Score Comparison for {dataset}")
        plt.ylabel("F1 Score")
        plt.ylim(0,1)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR,f"figure_kg_f1_comparison_{dataset}.png"))
        plt.show()

        plt.figure(figsize=(14,7))
        sns.barplot(y="Conflict_F1_Score", x="Method", data=df_results[df_results['Dataset']==dataset], palette="magma")
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Conflict Detection F1 Score Comparison for {dataset}")
        plt.ylabel("F1 Score")
        plt.ylim(0,1)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR,f"figure_conflict_f1_comparison_{dataset}.png"))
        plt.show()

    # --- Figure: Quality vs Latency (Overall) ---
    plt.figure(figsize=(12,8))
    sns.scatterplot(data=df_results, x="Time_Per_Doc_s", y="KG_F1_Score", hue="Method", style="Dataset", s=150, alpha=0.7)
    plt.title("KG Extraction Quality (F1) vs Latency (Time/Doc)")
    plt.xlabel("Time Per Document (seconds)")
    plt.ylabel("KG F1 Score")
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,"figure_quality_vs_latency.png"))
    plt.show()

    # --- Ablation Plot (if exists) ---
    ablation_path = os.path.join(RESULTS_DIR,"ablation_k_sweep.csv")
    if os.path.exists(ablation_path):
        df_ab = pd.read_csv(ablation_path)
        plt.figure(figsize=(8,5))
        sns.lineplot(data=df_ab, x="k", y="KG_F1", marker="o", hue="Dataset")
        plt.title("Ablation: RAG k vs KG F1 Score")
        plt.xlabel("Number of Retrieved Documents (k)")
        plt.ylabel("KG F1 Score")
        plt.ylim(0,1)
        plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR,"figure_ablation_k.png")); plt.show()

    # --- Scaling Plot (if exists) ---
    scaling_path = os.path.join(RESULTS_DIR,"scaling_results.csv")
    if os.path.exists(scaling_path):
        df_scale = pd.read_csv(scaling_path)
        plt.figure(figsize=(12,7))
        sns.lineplot(data=df_scale, x="n_docs", y="Time_Per_Doc_s", hue="Method", style="Dataset", marker="o")
        plt.title("Scaling: Time/Doc vs Number of Documents")
        plt.xlabel("Number of Documents")
        plt.ylabel("Time Per Document (seconds)")
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR,"figure_scaling_time_per_doc.png")); plt.show()
        
        plt.figure(figsize=(12,7))
        sns.lineplot(data=df_scale, x="n_docs", y="KG_F1", hue="Method", style="Dataset", marker="o")
        plt.title("Scaling: KG F1 Score vs Number of Documents")
        plt.xlabel("Number of Documents")
        plt.ylabel("KG F1 Score")
        plt.ylim(0,1)
        plt.xscale('log')
        plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR,"figure_scaling_f1_score.png")); plt.show()

    # --- Conflict Resolution Metaheuristic Comparison Table/Plot ---
    conflict_resolver_metrics_path = os.path.join(RESULTS_DIR, "conflict_resolver_metrics.csv")
    if os.path.exists(conflict_resolver_metrics_path):
        df_conflict_res = pd.read_csv(conflict_resolver_metrics_path)
        print("\nTable: Conflict Resolution Metaheuristic Comparison")
        print(df_conflict_res)

        plt.figure(figsize=(12, 7))
        sns.barplot(data=df_conflict_res, x="Method", y="Avg_Resolution_Score", hue="Dataset", palette="coolwarm")
        plt.xticks(rotation=45, ha='right')
        plt.title("Average LLM-Judged Resolution Score per Metaheuristic")
        plt.ylabel("Average Resolution Score (1-10)")
        plt.ylim(0, 10)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "figure_conflict_resolution_metaheuristics.png"))
        plt.show()

    # --- Statistical Test Results ---
    if stat_results:
        print("\nWilcoxon Paired Test Results (LLM-only vs RAG-LLM k=5):")
        print(f"  Statistic: {stat_results['stat']:.4f}")
        print(f"  P-value: {stat_results['p_value']:.6f}")

# --- Stage 9: Graph visualizations ---
def save_pyvis_graphs(all_graphs: Dict, datasets: Dict):
    print("\n--- Generating Interactive Graph Visualizations ---")
    
    # Generate GT graphs first
    for ds_name, ds_data in datasets.items():
        gt_path = ds_data['gt_path']
        with open(gt_path, "r", encoding="utf-8") as f: gt_data = json.load(f)
        G_gt = nx.MultiDiGraph()
        for ent in gt_data.get("entities", []):
            if 'id' in ent: G_gt.add_node(ent["id"], **{k:v for k,v in ent.items() if k!='id'})
        for r in gt_data.get("relationships", []): # Corrected to gt_data
            if {'source','target'} <= set(r.keys()): G_gt.add_edge(r["source"], r["target"], type=r.get("type","related"))
        
        save_pyvis(G_gt, f"Ground Truth: {ds_name}", os.path.join(RESULTS_DIR, f"graph_ground_truth_{ds_name}.html"))

    # Generate graphs for each method
    for name, G in all_graphs.items():
        output_filepath = os.path.join(RESULTS_DIR, f"graph_{name}.html")
        save_pyvis(G, name.replace("_", " "), output_filepath)
        
def save_pyvis(G: nx.MultiDiGraph, title: str, output_path: str):
    net = Network(notebook=False, height="750px", width="100%", directed=True, bgcolor="#222222", font_color="white")
    net.from_nx(G)
    for n_id, n_data in G.nodes(data=True):
        pyvis_node = next((n for n in net.nodes if n['id'] == n_id), None)
        if pyvis_node:
            node_title_str = f"ID: {n_id}\\nType: {n_data.get('type', 'Unknown')}\\nName: {n_data.get('name', '')}\\nConfidence: {n_data.get('confidence', 'N/A'):.2f}"
            pyvis_node['title'] = node_title_str
            if n_data.get('type') == 'Requirement':
                pyvis_node['color'] = '#33C1FF'
            elif n_data.get('type') == 'Feature':
                pyvis_node['color'] = '#FF5733'
            elif n_data.get('type') == 'System_Component':
                pyvis_node['color'] = '#66CC66'
            elif n_data.get('type') == 'User_Role':
                pyvis_node['color'] = '#FFD700'
            elif n_data.get('type') == 'Business_Rule':
                pyvis_node['color'] = '#CC99FF'
            else:
                pyvis_node['color'] = '#CCCCCC'
    
    for edge in net.edges:
        edge['arrows'] = {'to': {'enabled': True, 'scaleFactor': 0.7}}
        edge['color'] = {'inherit': 'from'}
        edge['font'] = {'size': 10, 'color': '#AAAAAA'}
        edge['label'] = f"{edge.get('type', 'related')} (Conf: {edge.get('confidence', 'N/A'):.2f})"
        
    net.show_buttons(filter_=['physics'])
    net.save_graph(output_path)
    print(f"Saved interactive graph to {output_path}")

# --- Stage 10: Human evaluation sheet ---
def generate_human_evaluation_sheet(all_graphs: Dict, datasets: Dict):
    print("\n--- Generating Human Evaluation Sheet ---")
    human_rows = []
    
    # Target RAG-LLM (k=5) from the GitHub dataset as primary for human evaluation
    target_ds_name = list(ALL_GITHUB_DATASETS_CONFIG.keys())[0] # Use first GitHub dataset
    target_graph_key = f"{target_ds_name}_RAG-LLM (k=5)".replace(" ", "_")
    
    G_eval = all_graphs.get(target_graph_key)
    
    if G_eval and G_eval.number_of_edges() > 0:
        for u, v, d in G_eval.edges(data=True):
            # Ensure the dataset key matches the ds_name_key from GITHUB_DATASETS
            source_doc_snippet = datasets[target_ds_name]['docs_map'].get(u, 'N/A')
            target_doc_snippet = datasets[target_ds_name]['docs_map'].get(v, 'N/A')
            
            human_rows.append({
                "source_id": u,
                "target_id": v,
                "relationship_type": d.get("type", "related"),
                "confidence_score": d.get("confidence", "N/A"),
                "source_text_snippet": source_doc_snippet[:300] + '...' if len(source_doc_snippet) > 300 else source_doc_snippet,
                "target_text_snippet": target_doc_snippet[:300] + '...' if len(target_doc_snippet) > 300 else target_doc_snippet,
                "relevance_rating_1_5": "",
                "comments": ""
            })
        pd.DataFrame(human_rows).to_csv(os.path.join(RESULTS_DIR, "human_eval_sheet.csv"), index=False)
        print(f"Saved human_eval_sheet.csv with {len(human_rows)} relationships for human rating.")
    else:
        print(f"No graph found for human evaluation for {target_graph_key} or it is empty. Skipping sheet generation.")


# --- Main execution block ---
if __name__ == "__main__":
    print("\n" + "="*80)
    print("STARTING RAG-ENHANCED REQUIREMENTS ENGINEERING EXPERIMENTS")
    print("="*80)
    
    if USE_LOCAL_MODEL:
        model_full_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
        if not os.path.exists(model_full_path):
            print(f"Model {MODEL_FILENAME} not found. Attempting download...")
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, local_dir=MODELS_DIR, local_dir_use_symlinks=False)
                print("Model downloaded.")
            except Exception as e:
                print(f"Model download failed: {e}. Please download it manually to {MODELS_DIR} or set USE_LOCAL_MODEL=False.")
                sys.exit(1)

    final_results_df, all_generated_graphs, all_datasets, llm_is_supported_global = run_all_experiments()
    stat_test_output = run_statistical_tests(final_results_df, all_datasets, llm_is_supported_global)  
    
    generate_plots_and_tables(final_results_df, all_generated_graphs, stat_test_output)
    save_pyvis_graphs(all_generated_graphs, all_datasets)
    generate_human_evaluation_sheet(all_generated_graphs, all_datasets)

    print("\n" + "="*80)
    print("EXPERIMENT EXECUTION COMPLETE")
    print("Final results, plots, and figures are in the 'results/' directory.")
    print("Ground truth placeholders (requiring manual annotation) are in the 'data/' directory.")
    print("="*80)