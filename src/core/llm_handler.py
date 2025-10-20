# In src/core/llm_handler.py
import json
from typing import Dict, List, Any
from llama_cpp import Llama

class LLMHandler:
    def __init__(self, model_path: str, n_ctx: int = 8192, n_gpu_layers: int = -1):
        self.model_path = model_path
        self.n_ctx = n_ctx
        if not model_path:
            raise ValueError("Model path must be provided for LLMHandler.")
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=self.n_ctx,
                verbose=False,
                chat_format='llama-3'
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Llama model from {model_path}: {e}")

    def generate_json_response(self, prompt: str, temperature: float = 0.2, max_tokens: int = 4096) -> Dict:
        if not self.llm:
            raise RuntimeError("LLM backend not initialized.")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides structured data. Your response MUST be only a single, valid JSON object, without any markdown formatting, comments, or extra text."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens
        )

        content = response['choices'][0]['message']['content']
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print("Warning: Direct JSON parsing failed. Attempting to clean and re-parse.")
            start_brace = content.find('{')
            end_brace = content.rfind('}')
            
            if start_brace != -1 and end_brace != -1:
                json_str = content[start_brace:end_brace+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"Fatal: Could not parse cleaned JSON. Error: {e}")
                    raise e
            else:
                raise json.JSONDecodeError("Could not find a JSON object in the LLM response.", content, 0)

    # This general generate method is kept for compatibility if needed elsewhere.
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 2048) -> str:
        if not self.llm:
            raise RuntimeError("LLM backend not initialized.")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response['choices'][0]['message']['content']

if __name__ == '__main__':
    # This block will be used for direct testing
    # Assuming 'models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf' is available for testing
    try:
        llm_handler = LLMHandler(model_path='models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf')
        
        json_prompt = "Extract the name and age from this sentence: John Doe is 30 years old."
        json_response = llm_handler.generate_json_response(json_prompt)
        print("Structured JSON Response:")
        print(json_response)

        text_response = llm_handler.generate("What is a Knowledge Graph?")
        print("\nGeneral Text Response:")
        print(text_response)
    except Exception as e:
        print(f"Error during LLMHandler example usage: {e}")