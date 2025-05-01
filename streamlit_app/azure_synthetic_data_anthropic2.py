import os
import json
import uuid
import random
import time
import httpx
from typing import List, Dict, Any, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_openai import AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import FiltrationConfig, EvolutionConfig
from deepeval.synthesizer import Evolution

class AzureOpenAIWithRetry(DeepEvalBaseLLM):
    """Custom Azure OpenAI wrapper for DeepEval with retry logic"""
    
    def __init__(self, api_key, endpoint, deployment, api_version="2024-02-01", max_retries=5):
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment = deployment
        self.api_version = api_version
        self.max_retries = max_retries
        self._model = None
    
    def load_model(self):
        if self._model is None:
            self._model = AzureChatOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint,
                azure_deployment=self.deployment,
                api_version=self.api_version,
                temperature=0.2,
                timeout=60
            )
        return self._model
    
    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.HTTPStatusError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True
    )
    def generate(self, prompt: str) -> str:
        """Generate response with retry logic"""
        try:
            chat_model = self.load_model()
            return chat_model.invoke(prompt).content
        except Exception as e:
            print(f"Error during generation (will retry): {str(e)}")
            # Add a small delay before retry
            time.sleep(2)
            raise
    
    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.HTTPStatusError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True
    )
    async def a_generate(self, prompt: str) -> str:
        """Asynchronously generate response with retry logic"""
        try:
            chat_model = self.load_model()
            res = await chat_model.ainvoke(prompt)
            return res.content
        except Exception as e:
            print(f"Error during async generation (will retry): {str(e)}")
            # Add a small delay before retry
            time.sleep(2)
            raise
    
    def get_model_name(self):
        return "Azure OpenAI Model with Retry"

def load_chunks_from_json(chunks_json_path: str) -> Dict:
    """
    Load chunks from a JSON file
    
    Args:
        chunks_json_path: Path to the JSON file with chunks
        
    Returns:
        Dictionary mapping chunk IDs to chunk content
    """
    print(f"Loading chunks from: {chunks_json_path}")
    with open(chunks_json_path, 'r', encoding='utf-8') as f:
        chunks_dict = json.load(f)
    
    print(f"Loaded {len(chunks_dict)} chunks from JSON file")
    return chunks_dict

def create_contexts_from_chunks(
    chunks_dict: Dict, 
    num_contexts: int = 20, 
    chunks_per_context: int = 3
) -> List[Tuple[List[str], List[str]]]:
    """
    Create contexts from chunks by selecting random groups
    
    Args:
        chunks_dict: Dictionary mapping chunk IDs to chunk content
        num_contexts: Number of different contexts to create
        chunks_per_context: Number of chunks per context
        
    Returns:
        List of tuples containing (chunk_contents, chunk_ids)
    """
    chunk_ids = list(chunks_dict.keys())
    
    if len(chunk_ids) < chunks_per_context:
        raise ValueError(f"Not enough chunks ({len(chunk_ids)}) to create contexts of size {chunks_per_context}")
    
    contexts = []
    for _ in range(num_contexts):
        # Select random chunks for this context
        selected_ids = random.sample(chunk_ids, chunks_per_context)
        
        # Get the contents for these chunks
        selected_contents = [chunks_dict[chunk_id]["content"] for chunk_id in selected_ids]
        
        contexts.append((selected_contents, selected_ids))
    
    print(f"Created {len(contexts)} contexts from chunks")
    return contexts

def generate_synthetic_data_from_json_chunks(
    chunks_json_path: str,
    azure_api_key: str,
    azure_endpoint: str,
    azure_deployment: str,
    num_contexts: int = 20,
    chunks_per_context: int = 3,
    max_goldens_per_context: int = 3,
    output_dir: str = "json_chunks_synthetic_data",
    start_index: int = 0  # Start from a specific index (useful for resuming)
):
    """
    Generate synthetic test data from JSON chunks using DeepEval with retry logic
    
    Args:
        chunks_json_path: Path to the JSON file with chunks
        azure_api_key: Azure OpenAI API key
        azure_endpoint: Azure OpenAI endpoint URL
        azure_deployment: Azure OpenAI deployment name
        num_contexts: Number of contexts to create
        chunks_per_context: Number of chunks per context
        max_goldens_per_context: Number of test cases to generate per context
        output_dir: Directory to save output
        start_index: Start generation from this index (for resuming interrupted runs)
    """
    # Initialize Azure OpenAI with retry
    print("Initializing Azure OpenAI with retry logic...")
    azure_openai = AzureOpenAIWithRetry(
        api_key=azure_api_key,
        endpoint=azure_endpoint,
        deployment=azure_deployment
    )
    
    # Configure DeepEval synthesizer
    print("Configuring DeepEval synthesizer...")
    evolution_config = EvolutionConfig(
        evolutions={
            Evolution.REASONING: 0.1,
            Evolution.MULTICONTEXT: 0.1,
            Evolution.CONCRETIZING: 0.1,
            Evolution.CONSTRAINED: 0.1,
            Evolution.COMPARATIVE: 0.1,
            Evolution.HYPOTHETICAL: 0.1,
            Evolution.IN_BREADTH: 0.4,
        },
        num_evolutions=2  # Reduced for faster processing
    )
    
    filtration_config = FiltrationConfig(
        critic_model=azure_openai,
        synthetic_input_quality_threshold=0.1,
        max_quality_retries=1
    )
    
    synthesizer = Synthesizer(
        model=azure_openai,
        filtration_config=filtration_config,
        evolution_config=evolution_config,
        async_mode=False  # Using synchronous mode for better error handling
    )
    
    # Load chunks from JSON file
    chunks_dict = load_chunks_from_json(chunks_json_path)
    
    # Create contexts from chunks
    contexts = create_contexts_from_chunks(
        chunks_dict=chunks_dict,
        num_contexts=num_contexts,
        chunks_per_context=chunks_per_context
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load existing data if we're resuming
    custom_format_data = []
    if start_index > 0:
        try:
            latest_file = os.path.join(output_dir, f"intermediate_results_{start_index}.json")
            if os.path.exists(latest_file):
                with open(latest_file, 'r', encoding='utf-8') as f:
                    custom_format_data = json.load(f)
                print(f"Loaded {len(custom_format_data)} existing test cases, resuming from index {start_index}")
        except Exception as e:
            print(f"Error loading existing data: {str(e)}")
            print("Starting from scratch")
            start_index = 0
    
    # Process each context with DeepEval, starting from the specified index
    for i in range(start_index, len(contexts)):
        try:
            print(f"Processing context {i+1}/{len(contexts)}")
            chunk_contents, chunk_ids = contexts[i]
            
            # Generate goldens from this context
            print(f"Generating synthetic data with DeepEval...")
            max_attempts = 3
            success = False
            
            for attempt in range(max_attempts):
                try:
                    goldens = synthesizer.generate_goldens_from_contexts(
                        contexts=[chunk_contents],
                        include_expected_output=True,
                        max_goldens_per_context=max_goldens_per_context
                    )
                    success = True
                    break
                except Exception as e:
                    print(f"Attempt {attempt+1}/{max_attempts} failed: {str(e)}")
                    time.sleep(5)  # Wait before retrying
            
            if not success:
                print(f"Failed to generate goldens for context {i+1} after {max_attempts} attempts, skipping...")
                continue
            
            # Convert to custom format
            for golden in goldens:
                custom_entry = {
                    "id": str(uuid.uuid4())[:8],
                    "question": golden.input,
                    "correct_chunks": chunk_ids,  # Use the chunk IDs from JSON
                    "correct_answer": golden.expected_output
                }
                custom_format_data.append(custom_entry)
                
            # Save intermediate results after each context
            intermediate_file = os.path.join(output_dir, f"intermediate_results_{i+1}.json")
            with open(intermediate_file, "w", encoding="utf-8") as f:
                json.dump(custom_format_data, f, indent=2)
                
            print(f"Saved intermediate results with {len(custom_format_data)} test cases so far")
            
            # Add a delay between contexts to avoid rate limiting
            time.sleep(3)
            
        except Exception as e:
            print(f"Error processing context {i+1}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Save progress so far
            error_file = os.path.join(output_dir, f"progress_before_error_{i+1}.json")
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(custom_format_data, f, indent=2)
            
            print(f"Saved progress before error to {error_file}")
            print(f"To resume, run the script with start_index={i+1}")
            continue
    
    # Save final results
    output_file = os.path.join(output_dir, "deepeval_synthetic_test_cases.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(custom_format_data, f, indent=2)
    
    print(f"Successfully generated {len(custom_format_data)} test cases")
    print(f"Saved to {output_file}")
    
    return custom_format_data

if __name__ == "__main__":
    # Path to your chunks JSON file
    CHUNKS_JSON_PATH = "deepeval/streamlit_app/extracted_chunks/all_pdf_chunks.json"
    
    # Azure OpenAI configuration
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_ENDPOINT")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT")
    
    # Generate synthetic data using DeepEval with retry logic
    test_cases = generate_synthetic_data_from_json_chunks(
        chunks_json_path=CHUNKS_JSON_PATH,
        azure_api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        num_contexts=30,  # Adjust as needed
        chunks_per_context=5,
        max_goldens_per_context=5,
        output_dir="./deepeval_synthetic_data",
        start_index=0  
    )
    
    # Print example
    if test_cases:
        print("\nSample test case:")
        sample = test_cases[0]
        print(f"ID: {sample['id']}")
        print(f"Question: {sample['question']}")
        print(f"Correct chunks: {sample['correct_chunks']}")
        print(f"Answer (first 200 chars): {sample['correct_answer'][:200]}...")