from typing import List, Optional
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseEmbeddingModel
import time
import os
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError
)
import logging
import openai
import os 
from dotenv import load_dotenv

load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureOpenAI(DeepEvalBaseLLM):
    """Custom Azure OpenAI model wrapper for DeepEval that handles schema validation"""
    
    def __init__(self, model):
        self.model = model
        
    def load_model(self):
        return self.model
    
    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM with retry logic for rate limits.
        Only accepts a prompt parameter, as per DeepEval documentation.
        """
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content
    
    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )    
    async def a_generate(self, prompt: str) -> str:
        """
        Asynchronously generate a response from the LLM with retry logic.
        Only accepts a prompt parameter, as per DeepEval documentation.
        """
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content
    
    def get_model_name(self):
        return "Azure OpenAI Model"

class AzureEmbedding(DeepEvalBaseEmbeddingModel):
    """Custom Azure OpenAI embedding model wrapper for DeepEval with rate limit handling"""
    
    def __init__(self, model):
        self.model = model
        
    def load_model(self):
        return self.model
    
    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text with rate limit retry handling"""
        embedding_model = self.load_model()
        return embedding_model.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts with batching and rate limit handling.
        Process in smaller batches to avoid rate limits.
        """
        BATCH_SIZE = 5
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(texts)-1)//BATCH_SIZE + 1}")
            
            # Use retrying for each batch
            batch_embeddings = self._embed_texts_batch(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Add a small delay between batches
            if i + BATCH_SIZE < len(texts):
                time.sleep(1)
        
        return all_embeddings
    
    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )
    def _embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """Helper method to embed a batch of texts with retries"""
        embedding_model = self.load_model()
        return embedding_model.embed_documents(texts)
    
    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )
    async def a_embed_text(self, text: str) -> List[float]:
        """Asynchronously embed a single text with rate limit retry handling"""
        embedding_model = self.load_model()
        try:
            return await embedding_model.aembed_query(text)
        except (AttributeError, NotImplementedError):
            return self.embed_text(text)
    
    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously embed multiple texts with batching and rate limit handling.
        Process in smaller batches to avoid rate limits.
        """
        BATCH_SIZE = 5
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(texts)-1)//BATCH_SIZE + 1}")
            
            # Use retrying for each batch
            try:
                batch_embeddings = await self._a_embed_texts_batch(batch)
                all_embeddings.extend(batch_embeddings)
            except RetryError as e:
                # If async fails after all retries, fall back to sync
                logger.warning(f"Async embedding failed after retries, falling back to sync: {e}")
                batch_embeddings = self._embed_texts_batch(batch)
                all_embeddings.extend(batch_embeddings)
            
            # Add a small delay between batches
            if i + BATCH_SIZE < len(texts):
                await asyncio.sleep(1)
        
        return all_embeddings
    
    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )
    async def _a_embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """Helper method to asynchronously embed a batch of texts with retries"""
        embedding_model = self.load_model()
        try:
            return await embedding_model.aembed_documents(texts)
        except (AttributeError, NotImplementedError):
            return self._embed_texts_batch(texts)
    
    def get_model_name(self):
        return "Azure OpenAI Embedding Model"


def initialize_azure_openai_models(
    azure_api_key=None, 
    azure_endpoint=None, 
    azure_deployment=None,
    azure_embedding_deployment=None,
    api_version="2024-12-01-preview"
):
    """
    Initialize and return Azure OpenAI models for completion and embedding
    
    Args:
        azure_api_key: Azure OpenAI API key
        azure_endpoint: Azure OpenAI endpoint URL
        azure_deployment: Name of the deployment for chat completion
        azure_embedding_deployment: Name of the deployment for embeddings
        api_version: API version string
        
    Returns:
        tuple: (AzureOpenAI, AzureEmbedding) models
    """
    # Use provided values or get from environment variables
    azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.environ.get("AZURE_ENDPOINT")
    azure_deployment = os.environ.get("AZURE_DEPLOYMENT")
    azure_embedding_deployment = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT")
    api_version = "2024-12-01-preview"  # Default API version

    
    if not all([azure_api_key, azure_endpoint, azure_deployment, azure_embedding_deployment]):
        raise ValueError("Missing required Azure OpenAI credentials. Please provide all required parameters.")
    
    # Initialize the chat model
    chat_model = AzureChatOpenAI(
        api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        api_version=api_version,
        temperature=0.0  # Use zero temperature for evaluation tasks
    )
    
    # Initialize the embedding model
    embedding_model = AzureOpenAIEmbeddings(
        api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_embedding_deployment,
        api_version=api_version
    )
    
    # Wrap with our DeepEval custom models
    azure_openai = AzureOpenAI(model=chat_model)
    azure_embedding = AzureEmbedding(model=embedding_model)
    
    return azure_openai, azure_embedding