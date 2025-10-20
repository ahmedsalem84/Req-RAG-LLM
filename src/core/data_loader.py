# In src/core/data_loader.py
import json
import pandas as pd
from typing import List, Dict, Any
import os
import requests
import zipfile
import io
import re

def load_github_issues(file_path: str) -> List[Dict]:
    """Loads GitHub issues from a JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GitHub issues file not found at: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def process_github_issues_for_rag(issues: List[Dict]) -> List[str]:
    """Processes GitHub issues into a simple text format for RAG."""
    documents = []
    for issue in issues:
        doc_content = (
            f"Requirement ID: ISSUE-{issue.get('number', 'N/A')}\n"
            f"Title: {issue.get('title', '')}\n"
            f"State: {issue.get('state', 'N/A')}\n"
            f"Author: {issue.get('user', {}).get('login', 'N/A')}\n"
            f"Labels: {[label['name'] for label in issue.get('labels', [])]}\n"
            f"Body:\n{issue.get('body', '') or ''}\n"
        )
        documents.append(doc_content.strip()) # strip to clean up extra newlines
    return documents

def download_and_extract_promise(dataset_name: str, download_url: str, target_dir: str) -> str:
    """Downloads and extracts a PROMISE dataset (usually ARFF files)."""
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, f"{dataset_name}.zip")
    extract_dir = os.path.join(target_dir, dataset_name)

    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        print(f"Dataset '{dataset_name}' already extracted at {extract_dir}.")
        return extract_dir
    
    # --- NEW: Check if ZIP file already exists locally ---
    if os.path.exists(zip_path):
        print(f"Found local ZIP file for '{dataset_name}' at {zip_path}. Skipping download.")
    else:
        print(f"Downloading {dataset_name} from {download_url}...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(download_url, stream=True, headers=headers, timeout=10) # Added timeout
            response.raise_for_status()

            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {dataset_name} to {zip_path}.")
        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to download {dataset_name} from URL: {e}")
            print(f"Please ensure '{zip_path}' exists locally or try again.")
            raise # Re-raise to stop if download failed AND file doesn't exist

    # --- Proceed to extraction ---
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file for '{dataset_name}' not found at '{zip_path}'. Cannot extract.")

    print(f"Extracting {dataset_name} from {zip_path} to {extract_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        # os.remove(zip_path) # You can remove this line if you want to keep the .zip file
        print(f"Extracted {dataset_name} to {extract_dir}.")
    except Exception as e:
        print(f"Error extracting ZIP file {zip_path}: {e}")
        raise # Re-raise extraction error

    return extract_dir

def parse_arff_to_documents(arff_file_path: str, content_field_index: int = -1, id_field_index: int = 0) -> List[str]:
    """
    Parses an ARFF file to extract document content.
    Assumes the relevant text is at content_field_index (default last)
    and an ID is at id_field_index (default first), if available.
    """
    documents = []
    in_data_section = False
    
    with open(arff_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if line.upper().startswith('@DATA'):
                in_data_section = True
                continue
            if not in_data_section or not line:
                continue

            parts = [p.strip().strip("'\"") for p in line.split(',')] # Strip quotes/whitespace from each part
            
            doc_id = None
            doc_text = ""

            if len(parts) > 0:
                # Try to get an ID from the specified index
                if len(parts) > id_field_index:
                    potential_id = parts[id_field_index]
                    # Generic ID format: alphanumeric, dashes, etc.
                    id_match = re.search(r'([A-Za-z0-9-]+)', potential_id)
                    if id_match:
                        doc_id = f"PROMISE-{id_match.group(1).upper()}"
                    else: # If no explicit ID, create one
                        doc_id = f"PROMISE-DOC-{line_num}" 
                else:
                    doc_id = f"PROMISE-DOC-{line_num}"

                # Extract content from the specified index
                if len(parts) > content_field_index:
                    doc_text = parts[content_field_index]
                else:
                    doc_text = line # Fallback to entire line if index is out of bounds


            # Ensure doc_text is not empty after stripping, if it is, use doc_id
            if not doc_text:
                doc_text = doc_id # Use ID as content if no other text is found

            documents.append(f"Requirement ID: {doc_id}\nBody:\n{doc_text.strip()}")
            
    return documents


if __name__ == '__main__':
    # Example usage for GitHub issues
    print("--- GitHub Issues Example ---")
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    data_dir = os.path.join(project_root, 'data')
    superset_issues_path = os.path.join(data_dir, 'superset_issues.json')

    if not os.path.exists(superset_issues_path):
        print(f"Please ensure {superset_issues_path} exists for this example.")
    else:
        issues = load_github_issues(superset_issues_path)
        documents = process_github_issues_for_rag(issues[:2])
        for doc in documents:
            print(doc)
            print("-" * 20)

    # Example usage for PROMISE dataset (CM1)
    print("\n--- PROMISE CM1 Example ---")
    promise_data_dir = os.path.join(data_dir, 'promise_datasets')
    cm1_url = "http://promise.site.uottawa.ca/SERepository/datasets/cm1.zip"
    cm1_extract_path = download_and_extract_promise('cm1', cm1_url, promise_data_dir)
    
    # Locate the ARFF file inside the extracted directory
    cm1_arff_file = os.path.join(cm1_extract_path, 'cm1.arff') # Assuming it's directly in there
    if os.path.exists(cm1_arff_file):
        cm1_docs = parse_arff_to_documents(cm1_arff_file)
        print(f"Parsed {len(cm1_docs)} documents from CM1.arff.")
        for i, doc in enumerate(cm1_docs[:2]):
            print(doc)
            print("-" * 20)
    else:
        print(f"CM1.arff not found in {cm1_extract_path}. Check extraction contents.")