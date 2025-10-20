# In fetch_all_raw_data.py
import requests
import json
import os
import time
from typing import List, Dict

# --- Configuration ---
DATA_DIR = "data" 
os.makedirs(DATA_DIR, exist_ok=True)

# Define all GitHub repositories to fetch
GITHUB_REPOS_TO_FETCH = {
    "apache_superset": "apache/superset",
    "apache_airflow": "apache/airflow",
    "tensorflow_tensorflow": "tensorflow/tensorflow"
}

def fetch_github_issues_for_repo(repo_owner_slash_name: str, output_path: str, max_pages: int = 5) -> List[Dict]:
    """
    Fetches GitHub issues for a given repository and saves them to a JSON file.
    Fetches up to max_pages * 100 issues.
    """
    print(f"\nFetching GitHub issues for {repo_owner_slash_name} (up to {max_pages*100} issues)...")
    issues = []
    base_url = f"https://api.github.com/repos/{repo_owner_slash_name}/issues?state=all&per_page=100"
    for p in range(1, max_pages + 1):
        try:
            print(f"  Fetching page {p}...")
            response = requests.get(f"{base_url}&page={p}")
            response.raise_for_status()
            page_items = response.json()
            if not page_items:
                print(f"  No more issues found after page {p-1}.")
                break
            issues.extend(page_items)
            time.sleep(1) # Be kind to the API, crucial for larger fetches
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching page {p} for {repo_owner_slash_name}: {e}")
            break
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2)
    print(f"Successfully saved {len(issues)} raw issues for {repo_owner_slash_name} to {output_path}")
    return issues

if __name__ == "__main__":
    print("--- Starting Raw GitHub Data Fetch ---")

    for ds_key, repo_path in GITHUB_REPOS_TO_FETCH.items():
        raw_output_file = os.path.join(DATA_DIR, f"{ds_key}_issues.json")
        if not os.path.exists(raw_output_file):
            fetch_github_issues_for_repo(repo_path, raw_output_file)
        else:
            print(f"\nRaw issues for {repo_path} already exist at {raw_output_file}. Skipping fetch.")

    print("\n--- Raw GitHub Data Fetch Complete ---")