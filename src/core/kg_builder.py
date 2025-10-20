# In src/core/kg_builder.py
import networkx as nx
import json
from typing import List, Dict, Any
from .llm_handler import LLMHandler
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from tqdm import tqdm
import tiktoken

class KGBuilder:
    def __init__(self, llm_handler: LLMHandler, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.llm = llm_handler
        self.graph = nx.MultiDiGraph()
        self.vector_store = None
        self.documents = []
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def setup_rag(self, documents: List[str], chunk_size: int = 1000, chunk_overlap: int = 150):
        if self.vector_store is None or self.documents != documents:
            print("Setting up RAG vector store...")
            self.documents = documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            splits = text_splitter.create_documents(documents)
            self.vector_store = FAISS.from_documents(documents=splits, embedding=self.embeddings)
            print("RAG setup complete.")

    def _get_relevant_context(self, query: str, k: int = 5) -> str:
        if not self.vector_store:
            return ""
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            return ""

    def build_graph_from_documents(self, documents: List[str], use_rag: bool, k: int = 5):
        self.graph = nx.MultiDiGraph()

        if use_rag:
            self.setup_rag(documents)

        PROMPT_BUFFER_TOKENS = 750
        MAX_MODEL_CONTEXT_TOKENS = self.llm.n_ctx
        MAX_PROMPT_TOKENS = MAX_MODEL_CONTEXT_TOKENS - PROMPT_BUFFER_TOKENS

        for doc in tqdm(documents, desc=f"Building KG (RAG={'On' if use_rag else 'Off'}, k={k if use_rag else 'N/A'})"):
            main_doc_tokens = self.tokenizer.encode(doc)
            
            final_context_str = ""
            if use_rag:
                raw_context = self._get_relevant_context(doc, k=k)
                context_tokens = self.tokenizer.encode(raw_context)

                available_space_for_context = MAX_PROMPT_TOKENS - len(main_doc_tokens)
                
                if available_space_for_context < 0:
                    truncated_main_doc_tokens = main_doc_tokens[:MAX_PROMPT_TOKENS]
                    final_doc_str = self.tokenizer.decode(truncated_main_doc_tokens)
                    final_context_str = ""
                else:
                    final_doc_str = doc
                    truncated_context_tokens = context_tokens[:available_space_for_context]
                    final_context_str = self.tokenizer.decode(truncated_context_tokens)
            else:
                final_doc_str = doc
                final_context_str = ""

            if self.tokenizer.encode(final_doc_str).__len__() > MAX_PROMPT_TOKENS:
                final_doc_str = self.tokenizer.decode(self.tokenizer.encode(final_doc_str)[:MAX_PROMPT_TOKENS])

            prompt = self._create_extraction_prompt(final_doc_str, final_context_str)
            
            try:
                extracted_data = self.llm.generate_json_response(prompt)
                # print(f"DEBUG: LLM Output for {doc.splitlines()[0]}: {extracted_data}") # Keep for specific debugging, otherwise comment out
                self._update_graph(extracted_data)
            except Exception as e:
                doc_id = doc.splitlines()[0] if doc.splitlines() else "Unknown Document"
                # print(f"\nError processing document: {doc_id}") # Comment out for cleaner eval logs
                # print(f"Error details: {e}\nSkipping...") # Comment out for cleaner eval logs
                continue

        return self.graph

    def _create_extraction_prompt(self, document: str, context: str) -> str:
        """
        Creates a robust, XML-structured prompt with a distinct example,
        now including instructions for confidence scores.
        """
        prompt = f"""
<instructions>
  <task>You are a requirements engineering expert. Your job is to extract entities and relationships from the text inside the <document_to_process> tags. Use the text inside the <context> tags to discover relationships between different documents.</task>
  
  <output_schema>
    Your response MUST be a single, valid JSON object with "entities" and "relationships" keys.
    - Entities: Each entity object MUST have "id", "type", "name", and a "confidence" score (float between 0.0 and 1.0, where 1.0 is highest).
    - Relationships: Each relationship object MUST have "source", "target", "type", and a "confidence" score (float between 0.0 and 1.0).
    - Entity types: Requirement, Feature, System_Component, User_Role, Business_Rule.
    - Relationship types: depends_on, implements, is_part_of, constrains, conflicts_with.
    - If no relevant information is found, return empty lists for entities and relationships. Do not invent data.
  </output_schema>
  
  <example>
    <input_document>
      Requirement ID: REQ-08
      Title: User Profile Page
      The system shall display the user's name and email. This is part of the new 'Account Management' feature. It depends on REQ-07. The 'Account Management' feature should also implement a 'Password Reset' rule.
    </input_document>
    <output_json>
      {{
        "entities": [
          {{"id": "REQ-08", "type": "Requirement", "name": "User Profile Page", "confidence": 0.95}},
          {{"id": "Account Management", "type": "Feature", "name": "Account Management", "confidence": 0.90}},
          {{"id": "REQ-07", "type": "Requirement", "name": "Dependency Requirement", "confidence": 0.80}},
          {{"id": "Password Reset", "type": "Business_Rule", "name": "Password Reset", "confidence": 0.85}}
        ],
        "relationships": [
          {{"source": "REQ-08", "target": "Account Management", "type": "is_part_of", "confidence": 0.92}},
          {{"source": "REQ-08", "target": "REQ-07", "type": "depends_on", "confidence": 0.78}},
          {{"source": "Account Management", "target": "Password Reset", "type": "implements", "confidence": 0.88}}
        ]
      }}
    </output_json>
  </example>
</instructions>

<context>
{context}
</context>

<document_to_process>
{document}
</document_to_process>

Now, analyze the <document_to_process> and provide the JSON output.
"""
        return prompt

    def _update_graph(self, data: Dict):
        if not isinstance(data, dict):
            return
            
        for entity in data.get("entities", []):
            if isinstance(entity, dict) and 'id' in entity and 'type' in entity:
                entity_id = entity['id']
                entity_name = entity.get('name', entity_id)
                entity_confidence = entity.get('confidence', 0.5) # Default confidence if not provided
                self.graph.add_node(entity_id, type=entity['type'], name=entity_name, confidence=entity_confidence)
        
        for rel in data.get("relationships", []):
            if isinstance(rel, dict) and 'source' in rel and 'target' in rel and 'type' in rel:
                source_id = rel['source']
                target_id = rel['target']
                relationship_type = rel['type']
                relationship_confidence = rel.get('confidence', 0.5) # Default confidence if not provided

                if self.graph.has_node(source_id) and self.graph.has_node(target_id):
                    self.graph.add_edge(source_id, target_id, type=relationship_type, confidence=relationship_confidence, reason=rel.get('reason', ''))