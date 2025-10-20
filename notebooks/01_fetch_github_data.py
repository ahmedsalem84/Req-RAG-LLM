{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "81a4dfa4",
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "# In notebooks/01_fetch_github_data.ipynb\n",
    "import requests\n",
    "import json\n",
    "import pandas as pd\n",
    "import time\n",
    "\n",
    "# --- Configuration ---\n",
    "REPO = \"apache/superset\"\n",
    "OUTPUT_FILE = \"../data/superset_issues.json\"\n",
    "MAX_PAGES = 20 # Fetch 20 pages (100 issues per page = 2000 issues)\n",
    "\n",
    "# --- Fetching Logic ---\n",
    "all_issues = []\n",
    "url = f\"https://api.github.com/repos/{REPO}/issues?state=all&per_page=100\"\n",
    "\n",
    "for page in range(1, MAX_PAGES + 1):\n",
    "    print(f\"Fetching page {page}...\")\n",
    "    try:\n",
    "        response = requests.get(f\"{url}&page={page}\")\n",
    "        response.raise_for_status() # Raise an exception for bad status codes\n",
    "        issues = response.json()\n",
    "        if not issues:\n",
    "            break # No more issues\n",
    "        all_issues.extend(issues)\n",
    "        time.sleep(2) # Be a good API citizen\n",
    "    except requests.exceptions.RequestException as e:\n",
    "        print(f\"Error fetching page {page}: {e}\")\n",
    "        break\n",
    "\n",
    "print(f\"Fetched a total of {len(all_issues)} issues.\")\n",
    "\n",
    "# --- Save to a structured file ---\n",
    "with open(OUTPUT_FILE, 'w') as f:\n",
    "    json.dump(all_issues, f, indent=4)\n",
    "\n",
    "print(f\"Data saved to {OUTPUT_FILE}\")\n",
    "\n",
    "# --- Optional: Convert to DataFrame for easier inspection ---\n",
    "df = pd.DataFrame(all_issues)\n",
    "print(df[['number', 'title', 'state', 'created_at']].head())"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
