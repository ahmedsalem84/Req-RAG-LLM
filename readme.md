# RAG-Enhanced Knowledge Graph for Dynamic Requirements Engineering

This repository contains the full experimental framework for the research paper:

**"RAG-Enhanced Knowledge Graphs with Large Language Models for Dynamic Requirements Engineering: Automated Conflict Resolution and Traceability Optimization"**

This project demonstrates a novel approach to automate and enhance key aspects of Requirements Engineering (RE) by integrating Retrieval-Augmented Generation (RAG)-Enhanced Large Language Models (LLMs) with dynamic Knowledge Graph (KG) construction, metaheuristic conflict resolution, and multi-criteria traceability optimization.

## Table of Contents

1.  [Introduction](#1-introduction)
2.  [Features](#2-features)
3.  [Project Structure](#3-project-structure)
4.  [Setup Instructions](#4-setup-instructions)
    *   [4.1. Prerequisites](#41-prerequisites)
    *   [4.2. Clone Repository & Setup Environment](#42-clone-repository--setup-environment)
    *   [4.3. Download LLM Model](#43-download-llm-model)
    *   [4.4. Pre-cache Embedding Models & NLTK Data](#44-pre-cache-embedding-models--nltk-data)
    *   [4.5. Fetch Raw Data](#45-fetch-raw-data)
    *   [4.6. Clean and Format GitHub Data](#46-clean-and-format-github-data)
    *   [4.7. Prepare Ground Truth (GT) Files](#47-prepare-ground-truth-gt-files)
5.  [Running the Experiments](#5-running-the-experiments)
6.  [Interpreting Results](#6-interpreting-results)
7.  [Interactive Web Application (Optional)](#7-interactive-web-application-optional)
8.  [Further Enhancements & Future Work](#8-further-enhancements--future-work)
9.  [Citation](#9-citation)
10. [License](#10-license)

---

## 1. Introduction

This project implements a cutting-edge framework to automate the extraction, analysis, and management of software requirements from unstructured natural language text. By combining LLMs, RAG, Knowledge Graphs, and metaheuristics, it tackles critical RE challenges such as conflict detection, conflict resolution optimization, and robust traceability analysis.

## 2. Features

*   **RAG-Enhanced KG Construction:** Dynamically builds a Knowledge Graph of entities (Requirements, Features, System Components, etc.) and relationships (implements, depends_on, conflicts_with) with LLM-generated confidence scores.
*   **Multiple LLM-based Methods:** Compares "LLM Only" vs. "RAG-LLM" with varying `k` values (1, 5, 10).
*   **Advanced Conflict Resolution:** Implements and compares Hill Climbing, Simulated Annealing, and Genetic Algorithms as metaheuristics to optimize conflict resolution strategies, using the LLM as a fitness function.
*   **Multi-Criteria Traceability:** Features an enhanced Ant Colony Optimization (ACO) algorithm for finding robust traceability paths based on length and relationship confidence.
*   **Comprehensive Baselines:** Benchmarks against traditional methods (BM25) and modern NLP baselines (spaCy, BERT NER) for KG extraction, plus a rule-based conflict detector.
*   **Multiple Real-world Datasets:** Evaluates performance across three diverse GitHub repositories (Apache Superset, Apache Airflow, TensorFlow).
*   **Ablation Studies:** Analyzes the impact of the RAG `k` parameter.
*   **Scalability Studies:** Investigates performance and efficiency trends with increasing document volumes.
*   **Statistical Analysis:** Conducts Wilcoxon signed-rank tests for significant performance comparisons.
*   **Reproducible Results:** Generates tables, plots, and interactive graph visualizations (HTML files).
*   **Interactive Web Application:** A Streamlit/FastAPI UI for qualitative exploration (optional to run).

## 3. Project Structure

rag-re/
├── data/ # Stores raw datasets, cleaned datasets, and ground truth files
│ ├── apache_superset_issues.json # Raw GitHub issues (fetched)
│ ├── apache_superset_issues_clean.json # Cleaned & formatted issues
│ ├── apache_superset_gt.json # Manual Ground Truth (for annotation)
│ ├── apache_airflow_issues.json
│ ├── apache_airflow_issues_clean.json
│ ├── apache_airflow_gt.json
│ ├── tensorflow_tensorflow_issues.json
│ ├── tensorflow_tensorflow_issues_clean.json
│ └── tensorflow_tensorflow_gt.json
├── models/ # Stores GGUF LLM model files (e.g., Llama-3-8B-Instruct.Q4_K_M.gguf)
├── results/ # Output directory for plots (.png), tables (.csv), and interactive graphs (.html)
│ ├── table_main_results.csv
│ ├── figure_kg_f1_comparison_apache_superset.png
│ ├── ... (many more plots and tables)
│ ├── graph_ground_truth_apache_superset.html
│ └── human_eval_sheet.csv
├── src/ # Main source code for the framework
│ ├── core/ # Core logic of your RAG-KG system
│ │ ├── data_loader.py # Handles dataset loading and initial parsing
│ │ ├── llm_handler.py # Abstracts LLM interaction, prompt formatting, JSON parsing
│ │ ├── kg_builder.py # RAG-enhanced Knowledge Graph construction
│ │ ├── conflict_resolver.py # Conflict detection and metaheuristic resolution (HC, SA, GA)
│ │ └── traceability_optimizer.py # Traceability optimization with ACO
│ └── baselines/ # Implementation of baseline methods
│ └── extractors.py # BM25, spaCy, BERT NER, Rule-based Conflict Detector
├── app/ # Web application for demonstration (optional)
│ ├── main_api.py # FastAPI backend
│ └── ui.py # Streamlit frontend
├── fetch_all_raw_data.py # Script to fetch raw GitHub issues
├── clean_github_data.py # Script to clean and format raw GitHub issues
├── evaluation.py # Main script to run all experiments and generate results
├── cache_models.py # Script to pre-cache Sentence-Transformers embedding model
├── .gitignore # Specifies files/directories to ignore in Git
└── requirements.txt # Python package dependencies

---

## 4. Setup Instructions

Follow these steps carefully to set up your environment and prepare the data.

### 4.1. Prerequisites

*   **Python:** Version 3.10 or newer.
*   **Git:** For cloning the repository.
*   **(Optional) VS Code:** Recommended IDE with Python extensions.
*   **GPU (Recommended):** A powerful NVIDIA GPU (e.g., RTX 30-series, 40-series) with sufficient VRAM (12GB+ is ideal for Llama-3-8B) for faster LLM inference. If no GPU, it will fall back to CPU, but LLM runs will be *extremely slow*.

### 4.2. Clone Repository & Setup Environment

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_github_repo_link_here
    cd rag-re
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    If `requirements.txt` is missing or outdated, regenerate it after initial setup:
    ```bash
    pip install "llama-cpp-python[server]" # Installs llama-cpp-python with server extras for GPU/CPU support
    pip install langchain langchain_community sentence-transformers faiss-cpu pandas matplotlib seaborn scikit-learn tqdm tiktoken huggingface_hub pyvis spacy rank_bm25 scipy
    # Download spaCy and NLTK data
    python -m spacy download en_core_web_sm
    python -m nltk.downloader stopwords
    pip freeze > requirements.txt
    ```
    **Note:** If `llama-cpp-python[server]` fails to install with GPU support, consult `llama-cpp-python` documentation for platform-specific instructions (e.g., CUDA setup). It will default to CPU otherwise.

### 4.3. Download LLM Model

The system uses `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf`.

1.  **Download:** Go to the [Llama-3-8B-Instruct GGUF page on Hugging Face](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF). Download `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf`.
2.  **Place:** Move the downloaded `.gguf` file into the `models/` directory.

### 4.4. Pre-cache Embedding Models & NLTK Data

This step ensures all necessary language models and data are downloaded once.

1.  **Run the caching script:**
    ```bash
    python cache_models.py
    ```
    This script will download the `sentence-transformers/all-MiniLM-L6-v2` embedding model and verify NLTK stopwords.

### 4.5. Fetch Raw Data

This script fetches raw GitHub issue data for all three target repositories.

1.  **Run the data fetching script:**
    ```bash
    python fetch_all_raw_data.py
    ```
    This will create `apache_superset_issues.json`, `apache_airflow_issues.json`, and `tensorflow_tensorflow_issues.json` in your `data/` directory. Be patient, as this involves API calls with `time.sleep` between requests.

### 4.6. Clean and Format GitHub Data

This script processes the raw GitHub issues, filters out automated entries (like Dependabot PRs), and formats them into a clean, readable text format.

1.  **Run the cleaning script:**
    ```bash
    python clean_github_data.py
    ```
    This will create `apache_superset_issues_clean.json`, `apache_airflow_issues_clean.json`, and `tensorflow_tensorflow_issues_clean.json` in your `data/` directory.

### 4.7. Prepare Ground Truth (GT) Files

This is the **most critical manual step** for obtaining meaningful quantitative results.

1.  **Delete any existing placeholder GT files:** Before the first evaluation run, ensure `data/*.gt.json` files are deleted if they are not your *final annotated versions*.
2.  **Run `evaluation.py` for the first time:** This will generate *new placeholder* `_gt.json` files in your `data/` directory (e.g., `apache_superset_gt.json`, `apache_airflow_gt.json`, `tensorflow_tensorflow_gt.json`).
3.  **STOP `evaluation.py` after placeholder generation.**
4.  **Manually Annotate `_gt.json` files:**
    *   Open each `_gt.json` file generated in `data/`.
    *   For the first `50` documents of each dataset (as configured by `doc_limit` in `evaluation.py`), you must **manually populate the `"relationships"` and `"conflicts"` lists** with accurate entries.
    *   Ensure all entities mentioned in your relationships/conflicts are also listed in the `"entities"` section (which is pre-populated with requirement IDs). Add `Feature`, `System_Component`, `Business_Rule`, `User_Role` entities as needed.
    *   Assign reasonable `confidence` scores (0.0-1.0) to your manual annotations.
    *   **This step is time-consuming but essential for valid research results.**
5.  **Save your annotated `_gt.json` files.**

## 5. Running the Experiments

Once all prerequisites are installed, data is fetched and cleaned, and **your ground truth files are fully annotated**, you can run the full experimental pipeline.

1.  **Activate your virtual environment (if not already active).**
2.  **Run the main evaluation script:**
    ```bash
    python evaluation.py
    ```
    This script will:
    *   Load all datasets.
    *   Initialize the LLM (takes a few minutes).
    *   Execute all baseline methods (BM25, spaCy, BERT NER).
    *   Execute LLM Only and RAG-LLM methods for different `k` values.
    *   Execute metaheuristic conflict resolution methods.
    *   Perform ablation and scalability studies.
    *   Compute all performance metrics (Precision, Recall, F1, Time/Doc, Graph Stats).
    *   Conduct statistical tests.
    *   **Save all generated plots (.png), tables (.csv), and interactive graph visualizations (.html) to the `results/` directory.**

    **Note:** LLM-based runs are computationally intensive and will take a significant amount of time (potentially hours to tens of hours depending on your hardware and `EVAL_DOC_LIMIT`). Do not interrupt the process.

## 6. Interpreting Results

After `evaluation.py` completes, navigate to the `results/` directory.

*   **Tables (`.csv` files):** Open these in a spreadsheet program (Excel, Google Sheets) to view the quantitative comparisons.
    *   `table_main_results.csv`: Core performance for all methods/datasets.
    *   `ablation_k_sweep.csv`: RAG `k` parameter impact.
    *   `scaling_results.csv`: Scalability trends.
    *   `conflict_resolver_metrics.csv`: Metaheuristic conflict resolution comparison.
    *   `human_eval_sheet.csv`: Template for human relevance rating (if you choose to do a human evaluation).
*   **Plots (`.png` files):** View these images to see graphical representations of performance trends.
*   **Interactive Graphs (`.html` files):** Open these in a web browser to explore the constructed Knowledge Graphs visually.

## 7. Interactive Web Application (Optional)

You can run a simplified version of the system locally with a web UI for qualitative exploration.

1.  **Start the FastAPI backend:**
    ```bash
    uvicorn app.main_api:app --reload
    ```
2.  **Start the Streamlit frontend (in a separate terminal):**
    ```bash
    streamlit run app/ui.py
    ```
3.  Open your browser to the URL provided by Streamlit (usually `http://localhost:8501`).
4.  Click "Load Data & Build KG" in the sidebar to initialize the LLM and build a graph for a small subset of issues. This allows you to interact with conflict detection and traceability.

## 8. Further Enhancements & Future Work

Refer to the "Future Work" section in the paper for detailed ideas on extending this framework (e.g., dynamic KG evolution, multi-modal input, human-in-the-loop feedback, advanced traceability metrics).

