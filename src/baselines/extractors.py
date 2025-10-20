# In src/baselines/extractors.py
import re
import networkx as nx
from rank_bm25 import BM25Okapi
import spacy
from tqdm import tqdm
from typing import List, Dict, Tuple, Any
import numpy as np

# For Hugging Face NER Baseline
from transformers import pipeline

# For improved rule-based conflict detector
import nltk
from nltk.corpus import stopwords
import string

# --- Load spaCy model (ensure it's downloaded) ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spacy model 'en_core_web_sm'. This will happen once.")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- Initialize Hugging Face NER pipeline (pre-trained, general-purpose) ---
try:
    hf_ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
    print("Hugging Face NER pipeline loaded (dbmdz/bert-large-cased-finetuned-conll03-english).")
except Exception as e:
    print(f"Warning: Hugging Face NER pipeline could not be loaded: {e}. BERT NER baseline will be skipped.")
    hf_ner_pipeline = None

# --- Initialize NLTK stopwords ---
try:
    stop_words = set(stopwords.words('english'))
    # Add common punctuation to stopwords for cleaner term comparison
    stop_words.update(string.punctuation)
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stop_words.update(string.punctuation)


def get_doc_id(text: str) -> str:
    """Extracts a Requirement ID from the standard format."""
    match = re.search(r"Requirement ID: ([\w-]+)", text)
    if match:
        return match.group(1).strip()
    return f"UNKNOWN-DOC-{hash(text[:50])}" # Fallback for docs without clear ID

def bm25_baseline_extractor(documents: List[str], top_n_similar: int = 3) -> nx.MultiDiGraph:
    """
    Extracts entities (Requirements) and relationships using BM25.
    - Explicit 'depends_on' from #ISSUE-ID references.
    - Implicit 'related_to' between highly similar documents.
    """
    G = nx.MultiDiGraph()
    doc_ids = [get_doc_id(doc) for doc in documents]
    
    tokenized_docs = []
    for doc_text in documents:
        # --- LESS AGGRESSIVE TOKENIZATION FOR BM25 (Allow punctuation/symbols as part of token) ---
        # Splitting by whitespace, keeping symbols. This is better for code-heavy issue titles/bodies.
        # Example: "deck.gl/layers" becomes "deck.gl/layers" not ["deck", "gl", "layers"]
        tokens = [word.lower() for word in doc_text.split() if len(word) >= 1] # Simple whitespace split, no regex findall
        
        # Additional cleaning for specific noise in GitHub issues
        tokens = [re.sub(r'https?:\\/\\/\\S+', '', token) for token in tokens] # Remove URLs
        tokens = [re.sub(r'[`"(){}\\[\\]<>]', '', token) for token in tokens] # Remove common code/markdown symbols
        tokens = [token.strip(string.punctuation) for token in tokens if token.strip(string.punctuation)] # Strip leading/trailing punctuation
        
        if not tokens: 
            tokens = ["placeholder_bm25_token"] # Ensure non-empty list
        tokenized_docs.append(tokens)

    if all(t == ["placeholder_bm25_token"] for t in tokenized_docs):
        print("Warning: All documents tokenized to empty lists, BM25 cannot be computed. Returning empty graph.")
        return nx.MultiDiGraph()

    bm25 = BM25Okapi(tokenized_docs)

    for i, doc_text in enumerate(tqdm(documents, desc="BM25 Baseline")):
        current_doc_id = doc_ids[i]
        if not current_doc_id: continue
        
        if not G.has_node(current_doc_id):
            G.add_node(current_doc_id, type="Requirement", name=doc_text.split('\n')[0].replace("Requirement ID: ", "")) # Name is title

        # A. Explicit References (depends_on)
        refs = re.findall(r"(?:ISSUE-|#)(\\d+)", doc_text)
        for r in set(refs):
            target_id = f"ISSUE-{r}"
            # Ensure target_id is within the documents being processed or a recognized external ID
            if target_id != current_doc_id and target_id in doc_ids: 
                if not G.has_node(target_id):
                    # For referenced issues, try to get their title from processed documents if available
                    referenced_doc_text = next((d for d in documents if get_doc_id(d) == target_id), None)
                    ref_name = referenced_doc_text.split('\n')[0].replace("Requirement ID: ", "") if referenced_doc_text else f"Referenced Issue {target_id}"
                    G.add_node(target_id, type="Requirement", name=ref_name)
                G.add_edge(current_doc_id, target_id, type="depends_on", reason="Explicit reference in text (BM25).", confidence=0.9)

        # B. Implicit Similarities (related_to)
        query_tokens = tokenized_docs[i]
        
        if query_tokens != ["placeholder_bm25_token"]:
            scores = bm25.get_scores(query_tokens)
            
            # Find top_n_similar documents that are NOT the current document
            top_n_indices = np.argsort(scores)[::-1]
            
            similar_count = 0
            for idx in top_n_indices:
                if idx == i: continue
                
                similar_doc_id = doc_ids[idx]
                if similar_doc_id not in G.nodes(): # Avoid adding edges to self or already linked
                    similarity_score = scores[idx]
                    
                    if similarity_score > 2.0: # TUNE THIS THRESHOLD
                        if not G.has_node(similar_doc_id):
                            ref_name = documents[idx].split('\n')[0].replace("Requirement ID: ", "")
                            G.add_node(similar_doc_id, type="Requirement", name=ref_name)
                        confidence_val = min(1.0, max(0.1, similarity_score / 10.0))
                        G.add_edge(current_doc_id, similar_doc_id, type="related_to", reason=f"BM25 similarity (score={similarity_score:.2f}).", confidence=confidence_val)
                        similar_count += 1
                        if similar_count >= top_n_similar: break
                
    return G

def spacy_baseline_extractor(documents: List[str]) -> nx.MultiDiGraph:
    """
    Extracts entities and 'implements' relationships using spaCy's NER.
    Maps recognized entities to 'Feature' or 'System_Component'.
    """
    G = nx.MultiDiGraph()
    doc_ids = [get_doc_id(doc) for doc in documents] # Get all doc IDs for cross-referencing

    for doc_text in tqdm(documents, desc="spaCy Baseline"):
        current_doc_id = get_doc_id(doc_text)
        if not current_doc_id: continue
        
        if not G.has_node(current_doc_id):
            G.add_node(current_doc_id, type="Requirement", name=doc_text.split('\n')[0].replace("Requirement ID: ", ""))

        doc_nlp = nlp(doc_text)
        for ent in doc_nlp.ents:
            entity_type = None
            if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART", "EVENT", "FAC", "GPE", "LOC", "NORP"): # Broaden types
                entity_type = "System_Component" # More generic mapping for baselines
            elif ent.label_ == "PERSON":
                 entity_type = "User_Role"
            elif ent.label_ in ("DATE", "TIME", "MONEY", "PERCENT", "QUANTITY"): # Numbers could be business rules
                 entity_type = "Business_Rule"
            
            if entity_type:
                entity_name = ent.text.strip()
                entity_id = re.sub(r'\\W+', '_', entity_name).upper() # Create a clean ID
                if not G.has_node(entity_id):
                    G.add_node(entity_id, type=entity_type, name=entity_name)
                G.add_edge(current_doc_id, entity_id, type="implements", reason=f"spaCy NER ({ent.label_}).", confidence=0.7)
        
        # --- Add basic dependency links based on ID mentions (similar to BM25) ---
        refs = re.findall(r"(?:ISSUE-|#)(\\d+)", doc_text)
        for r in set(refs):
            target_id = f"ISSUE-{r}"
            if target_id != current_doc_id and target_id in doc_ids:
                if not G.has_node(target_id):
                    referenced_doc_text = next((d for d in documents if get_doc_id(d) == target_id), None)
                    ref_name = referenced_doc_text.split('\n')[0].replace("Requirement ID: ", "") if referenced_doc_text else f"Referenced Issue {target_id}"
                    G.add_node(target_id, type="Requirement", name=ref_name)
                G.add_edge(current_doc_id, target_id, type="depends_on", reason="Explicit reference in text (spaCy).", confidence=0.8)

    return G

def hf_ner_baseline_extractor(documents: List[str]) -> nx.MultiDiGraph:
    """
    Extracts entities using a pre-trained Hugging Face NER model (BERT-like).
    Infers basic relationships via explicit ID mentions.
    """
    if hf_ner_pipeline is None:
        print("Hugging Face NER pipeline not initialized. Returning empty graph.")
        return nx.MultiDiGraph()

    G = nx.MultiDiGraph()
    doc_ids = [get_doc_id(doc) for doc in documents] # Get all doc IDs for cross-referencing

    for doc_text in tqdm(documents, desc="BERT NER Baseline"):
        current_doc_id = get_doc_id(doc_text)
        if not current_doc_id: continue
        
        if not G.has_node(current_doc_id):
            G.add_node(current_doc_id, type="Requirement", name=doc_text.split('\n')[0].replace("Requirement ID: ", ""))

        ner_results = hf_ner_pipeline(doc_text)

        for ent_data in ner_results:
            entity_name = ent_data['word'].strip()
            entity_type = None
            # Map general NER labels to RE-specific types (this is a heuristic)
            if ent_data['entity_group'] in ("ORG", "MISC", "LOC"): # Broaden types for generic NER
                entity_type = "System_Component"
            elif ent_data['entity_group'] == "PER":
                entity_type = "User_Role"
            
            if entity_type:
                entity_id = re.sub(r'\\W+', '_', entity_name).upper()
                if not G.has_node(entity_id):
                    G.add_node(entity_id, type=entity_type, name=entity_name)
                G.add_edge(current_doc_id, entity_id, type="implements", reason=f"HF NER ({ent_data['entity_group']}).", confidence=0.7)
        
        # --- Relationship Inference for BERT Baseline (Non-LLM) ---
        refs = re.findall(r"(?:ISSUE-|#)(\\d+)", doc_text)
        for r in set(refs):
            target_id = f"ISSUE-{r}"
            if target_id != current_doc_id and target_id in doc_ids:
                if not G.has_node(target_id):
                    referenced_doc_text = next((d for d in documents if get_doc_id(d) == target_id), None)
                    ref_name = referenced_doc_text.split('\n')[0].replace("Requirement ID: ", "") if referenced_doc_text else f"Referenced Issue {target_id}"
                    G.add_node(target_id, type="Requirement", name=ref_name)
                G.add_edge(current_doc_id, target_id, type="depends_on", reason="Explicit reference in text (BERT).", confidence=0.8)

    return G
# --- Rule-Based Conflict Detection (Corrected and Improved) ---
def rule_based_conflict_detector(documents: List[str]) -> List[Dict]:
    """
    A simple rule-based detector for explicit contradictions in requirements.
    This is highly simplified and serves as a placeholder.
    """
    conflicts = []
    
    for i in tqdm(range(len(documents)), desc="Rule-Based Conflict"):
        for j in range(i + 1, len(documents)):
            doc1_id = get_doc_id(documents[i])
            doc2_id = get_doc_id(documents[j])
            
            if not doc1_id or not doc2_id:
                continue

            doc1_lower = documents[i].lower()
            doc2_lower = documents[j].lower()

            # Corrected multiline condition using parentheses
            # --- REFINED RULES ---
            # Rule 1: Direct contradiction keywords
            if (("must be enabled" in doc1_lower and "must be disabled" in doc2_lower) or
                ("allowed" in doc1_lower and "not allowed" in doc2_lower) or
                ("mandatory" in doc1_lower and "optional" in doc2_lower)):
                # This block remains the same
                
                # Try to find common nouns/verbs to imply shared context
                tokens1 = [word for word in re.findall(r'\\b(\\w+)\\b', doc1_lower) if word not in stop_words and len(word) > 2]
                tokens2 = [word for word in re.findall(r'\\b(\\w+)\\b', doc2_lower) if word not in stop_words and len(word) > 2]
                
                common_meaningful_terms = set(tokens1).intersection(set(tokens2))

                if len(common_meaningful_terms) >= 1:
                    conflicts.append({
                        "req1": doc1_id,
                        "req2": doc2_id,
                        "reason": f"Rule-based: Conflicting directives on shared terms ({', '.join(list(common_meaningful_terms)[:3])})",
                        "status": "unresolved",
                        "confidence": 0.8 # Added confidence
                    })
            # --- NEW SIMPLE RULE FOR DEMONSTRATION IF YOUR GT HAS SOMETHING LIKE THIS (FIXED INDENT AND SYNTAX) ---
            # This is a very broad, example rule; tune it to your GT.
            elif ("error" in doc1_lower and "fix" in doc2_lower and 
                  len(set(re.findall(r'\\b(\\w+)\\b', doc1_lower)).intersection(set(re.findall(r'\\b(\\w+)\\b', doc2_lower)))) >= 1):
                # The condition is now a single line or properly wrapped by outer parentheses for clarity
                
                common_tokens_general = set(re.findall(r'\\b(\\w+)\\b', doc1_lower)).intersection(set(re.findall(r'\\b(\\w+)\\b', doc2_lower)))
                common_terms_filtered = [t for t in common_tokens_general if t not in stop_words and len(t) > 2]
                if len(common_terms_filtered) >= 1:
                    conflicts.append({
                        "req1": doc1_id,
                        "req2": doc2_id,
                        "reason": f"Rule-based: Bug/Fix relation on shared terms ({', '.join(list(common_terms_filtered)[:3])})",
                        "status": "unresolved",
                        "confidence": 0.5 # Lower confidence for more generic rule
                    })
    return conflicts