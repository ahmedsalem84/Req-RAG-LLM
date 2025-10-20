# In extract_first_50_for_gt.py
import json
import os

# --- Configuration ---
DATA_DIR = "data"

# Define the cleaned GitHub datasets
CLEAN_GITHUB_DATASETS = {
    "apache_superset": os.path.join(DATA_DIR, "apache_superset_issues_clean.json"),
    "apache_airflow": os.path.join(DATA_DIR, "apache_airflow_issues_clean.json"),
    "tensorflow_tensorflow": os.path.join(DATA_DIR, "tensorflow_tensorflow_issues_clean.json")
}

NUM_RECORDS_FOR_GT = 50 # As per your doc_limit

if __name__ == "__main__":
    print("--- Extracting First 50 Records from Cleaned GitHub Datasets ---")

    for ds_name, clean_file_path in CLEAN_GITHUB_DATASETS.items():
        output_txt_path = os.path.join(DATA_DIR, f"{ds_name}_first_{NUM_RECORDS_FOR_GT}_records.txt")
        
        if not os.path.exists(clean_file_path):
            print(f"Warning: Cleaned file for '{ds_name}' not found at '{clean_file_path}'. Skipping.")
            continue
        
        try:
            with open(clean_file_path, 'r', encoding='utf-8') as f:
                all_clean_docs = json.load(f)
            
            # Take the first NUM_RECORDS_FOR_GT documents
            subset_docs = all_clean_docs[:NUM_RECORDS_FOR_GT]
            
            with open(output_txt_path, 'w', encoding='utf-8') as f_out:
                for doc in subset_docs:
                    f_out.write(doc)
                    f_out.write("\n---\n") # Add a separator for clarity between docs
            
            print(f"Extracted first {len(subset_docs)} records from '{ds_name}' to '{output_txt_path}'.")
        except Exception as e:
            print(f"Error processing {clean_file_path}: {e}")

    print("\n--- Extraction Complete. Please upload the contents of the generated .txt files. ---")



