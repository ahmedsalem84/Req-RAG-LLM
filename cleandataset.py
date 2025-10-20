# In clean_github_data.py
import json
import os
import requests # Still needed for load_github_issues, though it's local
import time # Not strictly needed here, but fine to keep for consistency
from typing import List, Dict

# --- Configuration ---
# Define all GitHub repositories to clean
GITHUB_REPOS_TO_CLEAN = {
    "apache_superset": "apache/superset",
    "apache_airflow": "apache/airflow",
    "tensorflow_tensorflow": "tensorflow/tensorflow"
}
DATA_DIR = "data"

# This function is identical to the one in src/core/data_loader.py
def load_github_issues(file_path: str) -> List[Dict]:
    """Loads GitHub issues from a JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw GitHub issues file not found at: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# This function is identical to the one in src/core/data_loader.py
def process_and_filter_issues(issues: List[Dict]) -> List[str]:
    """
    Filters out automated Dependabot PRs and processes human-authored issues
    into a readable, structured text format.
    """
    clean_documents = []
    skipped_count = 0
    processed_count = 0

    print(f"Starting to process and filter {len(issues)} raw issues...")

    for issue in issues:
        if issue.get('user', {}).get('login', '') == 'dependabot[bot]':
            skipped_count += 1
            continue
        
        issue_number = issue.get('number', 'N/A')
        issue_title = issue.get('title', '')
        issue_state = issue.get('state', 'N/A')
        issue_author = issue.get('user', {}).get('login', 'N/A')
        issue_labels = [label['name'] for label in issue.get('labels', [])]
        issue_body = issue.get('body', '') or ''

        doc_content = (
            f"Requirement ID: ISSUE-{issue_number}\n"
            f"Title: {issue_title}\n"
            f"State: {issue_state}\n"
            f"Author: {issue_author}\n"
            f"Labels: {', '.join(issue_labels)}\n"
            f"Body:\n{issue_body.strip()}\n"
        )
        clean_documents.append(doc_content.strip())
        processed_count += 1
    
    print(f"Finished processing. Skipped {skipped_count} automated issues. Processed {processed_count} human-authored issues.")
    return clean_documents

if __name__ == "__main__":
    print("--- Starting GitHub Data Cleaning ---")
    for ds_key, repo_path in GITHUB_REPOS_TO_CLEAN.items():
        RAW_ISSUES_PATH_SPECIFIC = os.path.join(DATA_DIR, f"{ds_key}_issues.json")
        CLEAN_ISSUES_PATH_SPECIFIC = os.path.join(DATA_DIR, f"{ds_key}_issues_clean.json")

        if not os.path.exists(RAW_ISSUES_PATH_SPECIFIC):
            print(f"Warning: Raw issues file for {repo_path} not found at {RAW_ISSUES_PATH_SPECIFIC}. Please run fetch_all_raw_data.py first.")
            continue

        print(f"\nLoading raw issues from: {RAW_ISSUES_PATH_SPECIFIC}")
        raw_issues_data = load_github_issues(RAW_ISSUES_PATH_SPECIFIC)
        
        clean_documents = process_and_filter_issues(raw_issues_data)
        
        print(f"Saving {len(clean_documents)} cleaned documents to: {CLEAN_ISSUES_PATH_SPECIFIC}")
        with open(CLEAN_ISSUES_PATH_SPECIFIC, 'w', encoding='utf-8') as f:
            json.dump(clean_documents, f, indent=2)
        print(f"Cleaning for {ds_key} complete. Cleaned data in {CLEAN_ISSUES_PATH_SPECIFIC}")

    print("\n--- GitHub Data Cleaning Complete ---")