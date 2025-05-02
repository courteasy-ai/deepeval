# RAG Evaluation System
A FastAPI-based system for evaluating Retrieval-Augmented Generation (RAG) pipelines with synthetic data generation.

# Overview
This system provides a complete pipeline for:

1. Processing PDF documents and extracting text chunks
2. Storing chunks in a vector database (Qdrant)
3. Generating synthetic test data for evaluation
4. Evaluating retrieval and generation performance with multiple metrics
5. Exposing all functionality through a REST API

# Components

## Main Files

1. main.py: FastAPI server that orchestrates the entire pipeline
2. azure_extract_chunks.py: Functions for extracting text chunks from PDFs
3. azure_add_extracted_chunks_to_qdrant.py: Functionality for adding chunks to Qdrant
4. azure_synthetic_data_anthropic2.py: Synthetic test data generation
5. azure_evaluation_using_synthetic_data.py: Evaluation framework with multiple metrics
6. azure_models_with_tenacity.py: Azure OpenAI client with retry logic
7. streamlit_app.py : File used to maintain streamlit app 

# Steps to start an evaluation 

1. cd streamlit_app
2. python -m venv venv
3. source venv/bin/activate
4. pip install -r requirements.txt
5. uvicorn main:app --host 0.0.0.0 --port 8000 --reload
6. streamlit run streamlit_app.py


# Future Scope

1. Modify retrival of chunks for synthetic data from Qdrant db instead of local file system.
2. Use reranker for better retrieval accuracy. 
3. Remove the depedency on local file system and migrate everything to database.