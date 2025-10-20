import os

# === Define the full folder/file tree ===
structure = {
    "data": [],
    "models": [],
    "notebooks": [],
    "results": [],
    "src": {
        "core": [
            "data_loader.py",
            "llm_handler.py",
            "kg_builder.py",
            "conflict_resolver.py",
            "traceability_optimizer.py"
        ],
        "baselines": [
            "keyword_search.py",
            "rule_based_conflict.py"
        ],
    },
    "app": [
        "main_api.py",
        "ui.py"
    ]
}

# === Helper functions ===
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[+] Created directory: {path}")

def make_file(path, content=""):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[+] Created file: {path}")

# === Create folders and files recursively ===
def create_structure(base, tree):
    for name, content in tree.items():
        dir_path = os.path.join(base, name)
        make_dir(dir_path)

        # If value is dict, recurse
        if isinstance(content, dict):
            create_structure(dir_path, content)
        # If value is list of files
        elif isinstance(content, list):
            for file in content:
                file_path = os.path.join(dir_path, file)
                make_file(file_path, f"# {file}\n# Auto-generated placeholder\n")

# === Run creation ===
if __name__ == "__main__":
    root = os.getcwd()
    print(f"Initializing RAG-KG project in: {root}")

    # Create main folders
    create_structure(root, structure)

    # Create top-level files
    make_file(os.path.join(root, "evaluation.py"), "# evaluation.py\n# Script to run evaluation and generate plots\n")
    make_file(os.path.join(root, ".gitignore"),
              "# Ignore environments and large files\n"
              "venv/\n__pycache__/\n*.pyc\nmodels/\ndata/\nresults/\n.env\n")
    make_file(os.path.join(root, "requirements.txt"),
              "# Core dependencies\n"
              "fastapi\n"
              "uvicorn\n"
              "streamlit\n"
              "pandas\n"
              "matplotlib\n"
              "networkx\n"
              "faiss-cpu\n"
              "llama-cpp-python\n"
              "pydantic\n"
              "pytest\n")

    print("\n✅ Project structure created successfully!")
    print("You can now open this folder in VS Code and start building your RAG-Enhanced Requirements Engineering experiment.")
