import os
import json
import shutil
import tempfile
import uuid
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime
from pathlib import Path
from fastapi import Path
# Import your module functions
from azure_extract_chunks import extract_pdf_chunks, process_first_n_pdfs
from azure_add_extracted_chunks_to_qdrant import extract_pdfs_to_qdrant
from azure_evaluation_using_synthetic_data import (
    run_evaluation, 
    setup_qdrant_client,
    load_evaluation_data,
    vector_search_retrieval,
    generate_answer
)
import traceback
from langchain_openai import AzureOpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from azure_synthetic_data_anthropic2 import generate_synthetic_data_from_json_chunks
from dotenv import load_dotenv
load_dotenv() 

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
    
# Create FastAPI app
app = FastAPI(title="RAG Evaluation API", 
              description="API for evaluating RAG systems with synthetic data")

# Add CORS middleware to allow cross-origin requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATA_DIR = os.path.join(os.getcwd(), "rag_evaluation_data")
os.makedirs(DATA_DIR, exist_ok=True)

# Directory for uploaded PDFs
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
os.makedirs(PDF_DIR, exist_ok=True)

# Directory for extracted chunks
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
os.makedirs(CHUNKS_DIR, exist_ok=True)

# Directory for evaluation data
EVAL_DIR = os.path.join(DATA_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

# Status tracking for jobs
job_status = {}

# Model classes
class QdrantConfig(BaseModel):
    qdrant_uri: str
    qdrant_api_key: str
    qdrant_collection: str

class AzureConfig(BaseModel):
    azure_api_key: str
    azure_endpoint: str
    azure_deployment: str
    azure_embedding_deployment: str

class ChunkingConfig(BaseModel):
    chunk_size: int = 1024
    chunk_overlap: int = 128
    encoding: str = "utf-8"

class SyntheticDataConfig(BaseModel):
    num_contexts: int = 20
    max_context_size: int = 3
    max_goldens_per_context: int = 2
    output_dir: str = "qdrant_deepeval_data"
    async_mode: bool = True

class EvaluationConfig(BaseModel):
    run_retrieval_eval: bool = True
    run_e2e_eval: bool = True
    run_contextual_eval: bool = True
    use_azure_for_evaluation: bool = True

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    details: Optional[Dict[str, Any]] = None

class DocumentProcessingResponse(BaseModel):
    job_id: str
    message: str

class CollectionRequest(BaseModel):
    collection_name: str

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    question_id: str
    question: str
    answer: str

@app.get("/")
async def root():
    return {"message": "RAG Evaluation API is running"}

@app.post("/ask-question/{job_id}", response_model=QuestionResponse)
async def ask_question(job_id: str = Path(...), request: QuestionRequest = None):
    """Process a question and generate an answer using the RAG pipeline"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
    
    if not request or not request.question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    try:
        # Get the collection name from job status
        collection_name = job_status[job_id].get("collection_name", "unknown_collection")
        
        # Set up Qdrant client with the collection
        db = {
            "client": QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                port=443,
                timeout=60
            ),
            "collection_name": QDRANT_COLLECTION_NAME,
            "embedding_model": setup_azure_embeddings()  # Import this from azure_evaluation_using_synthetic_data
        }
        
        # Use the generate_answer function imported from azure_evaluation_using_synthetic_data
        answer = generate_answer(request.question, db)
        
        # Generate a unique question ID
        question_id = f"q_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        return QuestionResponse(
            question_id=question_id,
            question=request.question,
            answer=answer
        )
    
    except Exception as e:

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

# Function to set up Azure embeddings (copy from azure_evaluation_using_synthetic_data)
def setup_azure_embeddings():
    """Set up and return Azure OpenAI embeddings model"""
    # Initialize the embedding model
    embedding_model = AzureOpenAIEmbeddings(
        openai_api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
        api_version="2024-12-01-preview"

    )
    
    return embedding_model

# Add this endpoint to get evaluation results
@app.get("/evaluation-results/{job_id}")
async def get_evaluation_results(job_id: str = Path(...)):
    """Get the evaluation results for a job"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
    
    # Check if evaluation has been completed
    status = job_status[job_id].get("status", "")
    if status != "evaluation_completed":
        raise HTTPException(status_code=400, detail=f"Evaluation not completed. Current status: {status}")
    
    try:
        # Get the results file path
        results_path = job_status[job_id].get("results_path", "")
        
        if not results_path or not os.path.exists(results_path):
            raise HTTPException(status_code=404, detail="Evaluation results file not found")
        
        # Load the results file
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving evaluation results: {str(e)}")

@app.post("/create-collection/", response_model=DocumentProcessingResponse)
async def create_collection(request: CollectionRequest):
    """Create a new Qdrant collection"""

    
    try:
        # Default values
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            port=443,
            timeout=60  # Longer timeout
        )
        
        # Check if collection exists and delete it
        try:
            client.delete_collection(request.collection_name)
        except Exception:
            # Ignore if collection doesn't exist
            pass
        
        # Create new collection
        client.create_collection(
            collection_name=request.collection_name,
            vectors_config=models.VectorParams(
                size=1536,  # Size for text-embedding-ada-002
                distance=models.Distance.COSINE,
            )
        )
        
        return DocumentProcessingResponse(
            job_id="collection_create",
            message=f"Collection {request.collection_name} created successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating collection: {str(e)}")

@app.post("/upload-pdf/", response_model=DocumentProcessingResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    collection_name: str = Form("document_collection"),
    chunk_size: int = Form(1024),
    chunk_overlap: int = Form(128),
    encoding: str = Form("utf-8")
):
    """Upload a PDF file to begin processing"""
    # Generate a unique job ID
    job_id = f"job_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Create job directory
    job_dir = os.path.join(DATA_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    # Create subdirectories
    pdf_dir = os.path.join(job_dir, "pdfs")
    chunks_dir = os.path.join(job_dir, "chunks")
    synthetic_dir = os.path.join(job_dir, "synthetic_data")
    eval_dir = os.path.join(job_dir, "evaluation")
    
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(synthetic_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Save the uploaded file
    file_path = os.path.join(pdf_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()
    
    # Initialize job status
    job_status[job_id] = {
        "status": "uploaded",
        "progress": 0.1,
        "file_path": file_path,
        "collection_name": QDRANT_COLLECTION_NAME,
        "chunking_config": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "encoding": encoding
        },
        "steps_completed": [],
        "current_step": "PDF uploaded",
        "job_dir": job_dir,
        "pdf_files": [file.filename],
        # "feedback": []
    }
    
    # Start background processing
    background_tasks.add_task(
        process_document, 
        job_id=job_id, 
        file_path=file_path,
        collection_name=QDRANT_COLLECTION_NAME,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding=encoding
    )
    
    return DocumentProcessingResponse(
        job_id=job_id,
        message=f"PDF uploaded successfully. Processing started with job ID: {job_id}"
    )

@app.get("/job-status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a background job"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
    
    return JobStatusResponse(
        job_id=job_id,
        status=job_status[job_id]["status"],
        progress=job_status[job_id]["progress"],
        details={
            "steps_completed": job_status[job_id].get("steps_completed", []),
            "current_step": job_status[job_id].get("current_step", ""),
            "results_path": job_status[job_id].get("results_path", ""),
            "error": job_status[job_id].get("error", ""),
            "collection_name": job_status[job_id].get("collection_name", ""),
            "pdf_files": job_status[job_id].get("pdf_files", []),
            "chunking_config": job_status[job_id].get("chunking_config", {})
        }
    )

@app.get("/jobs", response_model=List[JobStatusResponse])
async def list_jobs():
    """List all jobs"""
    return [
        JobStatusResponse(
            job_id=job_id,
            status=status["status"],
            progress=status["progress"],
            details={
                "steps_completed": status.get("steps_completed", []),
                "current_step": status.get("current_step", ""),
                "pdf_files": status.get("pdf_files", []),
                "collection_name": status.get("collection_name", "")
            }
        )
        for job_id, status in job_status.items()
    ]

@app.post("/evaluate/{job_id}", response_model=DocumentProcessingResponse)
async def run_evaluation_endpoint(
    job_id: str, 
    background_tasks: BackgroundTasks,
    run_retrieval_eval: bool = True,
    run_e2e_eval: bool = True, 
    run_contextual_eval: bool = True
):
    """Run evaluation on processed document"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
    
    if job_status[job_id]["status"] != "synthetic_data_generated":
        raise HTTPException(status_code=400, 
                           detail=f"Job is not ready for evaluation. Current status: {job_status[job_id]['status']}")
    
    # Update job status
    job_status[job_id]["status"] = "evaluation_started"
    job_status[job_id]["progress"] = 0.7
    job_status[job_id]["current_step"] = "Starting evaluation"
    
    # Start background evaluation
    background_tasks.add_task(
        perform_evaluation,
        job_id=job_id,
        run_retrieval_eval=run_retrieval_eval,
        run_e2e_eval=run_e2e_eval,
        run_contextual_eval=run_contextual_eval
    )
    
    return DocumentProcessingResponse(
        job_id=job_id,
        message=f"Evaluation started for job ID: {job_id}"
    )

@app.get("/get-synthetic-data/{job_id}")
async def get_synthetic_data(job_id: str = Path(...)):
    """Get the synthetic data for a job"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
    
    # Try to get the synthetic data file path
    synthetic_data_file = job_status[job_id].get("synthetic_data_file", "")
    
    # If not found directly, check in other potential locations
    if not synthetic_data_file:
        # Check if it might be in the details
        if "details" in job_status[job_id]:
            synthetic_data_file = job_status[job_id]["details"].get("synthetic_data_file", "")
    
    # If still not found, try to locate it in the job directory
    if not synthetic_data_file:
        job_dir = job_status[job_id].get("job_dir", "")
        if job_dir:
            # Look in synthetic_data subdirectory
            synthetic_dir = os.path.join(job_dir, "synthetic_data")
            if os.path.exists(synthetic_dir):
                # Find any JSON files that might contain synthetic data
                import glob
                json_files = glob.glob(os.path.join(synthetic_dir, "*.json"))
                if json_files:
                    synthetic_data_file = json_files[0]  # Use the first one found
    
    if not synthetic_data_file or not os.path.exists(synthetic_data_file):
        raise HTTPException(status_code=404, 
                          detail=f"Synthetic data file not found for job {job_id}")
    
    try:
        # Load the synthetic data from the file
        with open(synthetic_data_file, 'r', encoding='utf-8') as f:
            synthetic_data = json.load(f)
        
        # Verify we have actual data
        if not synthetic_data or len(synthetic_data) == 0:
            raise HTTPException(status_code=404, 
                              detail="Synthetic data file is empty")
        
        return synthetic_data
    except Exception as e:
        raise HTTPException(status_code=500, 
                          detail=f"Error loading synthetic data: {str(e)}")
        
async def process_document(
    job_id: str, 
    file_path: str, 
    collection_name: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    encoding: str = "utf-8"
):
    """Background task to process a document through the entire pipeline"""
    try:
        job_dir = job_status[job_id]["job_dir"]
        chunks_dir = os.path.join(job_dir, "chunks")
        synthetic_dir = os.path.join(job_dir, "synthetic_data")
        
        # Step 1: Extract chunks
        job_status[job_id]["status"] = "extracting_chunks"
        job_status[job_id]["progress"] = 0.2
        job_status[job_id]["current_step"] = "Extracting chunks from PDF"
        
        chunks_file = os.path.join(chunks_dir, "pdf_chunks.json")
        
        chunks_dict = extract_pdf_chunks(
            pdf_path=file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding=encoding
        )
        
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks_dict, f, indent=2)
        
        job_status[job_id]["steps_completed"].append("chunk_extraction")
        job_status[job_id]["chunks_file"] = chunks_file
        
        # Step 2: Add to Qdrant
        job_status[job_id]["status"] = "adding_to_qdrant"
        job_status[job_id]["progress"] = 0.4
        job_status[job_id]["current_step"] = "Adding chunks to Qdrant"
        
        # We'll need to create a temp directory with the PDF
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdf_path = os.path.join(temp_dir, os.path.basename(file_path))
            shutil.copy(file_path, temp_pdf_path)
            
            try:
                # Use default values for Qdrant
                
                chunks = extract_pdfs_to_qdrant(
                    directory=temp_dir,
                    qdrant_uri=QDRANT_URL,
                    qdrant_api_key=QDRANT_API_KEY,
                    collection_name=QDRANT_COLLECTION_NAME,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    output_dir=chunks_dir,
                    encoding=encoding,
                    n=1  # Just process the one file we copied
                )
                job_status[job_id]["steps_completed"].append("qdrant_upload")
            except Exception as e:
                job_status[job_id]["qdrant_error"] = str(e)
                print(f"Error uploading to Qdrant: {str(e)}")
                # Continue even if Qdrant upload fails
        
        # Step 3: Generate synthetic data
        job_status[job_id]["status"] = "generating_synthetic_data"
        job_status[job_id]["progress"] = 0.6
        job_status[job_id]["current_step"] = "Generating synthetic test data"
        
        os.makedirs(synthetic_dir, exist_ok=True)
        
        # Use default Azure values
        azure_api_key = AZURE_API_KEY
        azure_endpoint = AZURE_ENDPOINT
        azure_deployment = AZURE_OPENAI_DEPLOYMENT
        azure_embedding_deployment = AZURE_EMBEDDING_DEPLOYMENT
        
        # Modified approach: Use json chunks directly to avoid asyncio issues
        try:            
            # Generate synthetic data using the json chunks directly
            test_cases = generate_synthetic_data_from_json_chunks(
                chunks_json_path=chunks_file,
                azure_api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment,
                num_contexts=5,  # Reduced for faster processing
                chunks_per_context=2,
                max_goldens_per_context=2,
                output_dir=synthetic_dir,
                start_index=0
            )
        except Exception as e:
            print(f"Error with alternative synthetic data generation: {str(e)}")
            
            # Fallback option: generate some basic synthetic data manually
            test_cases = []
            
            # Load chunks from the file
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            
            # Create some basic test cases
            chunk_ids = list(chunk_data.keys())[:5]  # Use the first 5 chunks
            
            for i, chunk_id in enumerate(chunk_ids):
                chunk_content = chunk_data[chunk_id]['content']
                # Create a simple question from the chunk content
                test_cases.append({
                    "id": f"synthetic_{i}",
                    "question": f"What is the main topic discussed in this passage: '{chunk_content[:100]}...'?",
                    "correct_chunks": [chunk_id],
                    "correct_answer": f"The passage discusses {chunk_content[:200]}..."
                })
        
        # Save the final test cases
        synthetic_data_file = os.path.join(synthetic_dir, "synthetic_test_cases.json")
        with open(synthetic_data_file, "w", encoding="utf-8") as f:
            json.dump(test_cases, f, indent=2)
        
        job_status[job_id]["steps_completed"].append("synthetic_data_generation")
        job_status[job_id]["synthetic_data_file"] = synthetic_data_file
        job_status[job_id]["status"] = "synthetic_data_generated"
        job_status[job_id]["progress"] = 0.65
        job_status[job_id]["current_step"] = "Ready for evaluation"
        
    except Exception as e:
        job_status[job_id]["status"] = "processing_error"
        job_status[job_id]["error"] = str(e)
        job_status[job_id]["progress"] = -1
        job_status[job_id]["current_step"] = f"Error: {str(e)}"
        print(f"Error processing document: {str(e)}")
        traceback.print_exc()
        
# 1. First, the modified evaluation function that stores answers:

async def perform_evaluation(
    job_id: str, 
    run_retrieval_eval: bool = True, 
    run_e2e_eval: bool = True,
    run_contextual_eval: bool = True
):
    """Background task to run evaluation directly without using run_evaluation"""
    try:
        # Get job directory
        job_dir = job_status[job_id]["job_dir"]
        eval_dir = os.path.join(job_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Get synthetic data file path
        synthetic_data_file = job_status[job_id]["synthetic_data_file"]
        
        if not os.path.exists(synthetic_data_file):
            raise FileNotFoundError(f"Synthetic data file not found: {synthetic_data_file}")
        
        # Update status
        job_status[job_id]["status"] = "evaluating"
        job_status[job_id]["progress"] = 0.8
        job_status[job_id]["current_step"] = "Running evaluation"
        
        # Get collection name
        collection_name = job_status[job_id]["collection_name"]
        
        # Import necessary functions directly
        from azure_evaluation_using_synthetic_data import (
            setup_qdrant_client,
            load_evaluation_data,
            evaluate_retrieval, 
            vector_search_retrieval,
            evaluate_end_to_end, 
            generate_answer,
            analyze_failures
        )
        
        # Load the evaluation data
        evaluation_data = load_evaluation_data(synthetic_data_file)
        
        # Track all results
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_questions": len(evaluation_data),
                "evaluation_components": []
            }
        }
        
        # Set up the database connection
        db = setup_qdrant_client()
        
        # Run retrieval evaluation
        if run_retrieval_eval:
            try:
                results["metadata"]["evaluation_components"].append("standard_retrieval")
                
                precision, recall, mrr, f1, precisions, recalls, mrrs = evaluate_retrieval(
                    retrieval_function=vector_search_retrieval,
                    evaluation_data=evaluation_data,
                    db=db
                )
                
                results["retrieval"] = {
                    "precision": precision,
                    "recall": recall,
                    "mrr": mrr,
                    "f1": f1,
                    "individual_results": {
                        "precisions": precisions,
                        "recalls": recalls,
                        "mrrs": mrrs
                    }
                }
                
                # Save retrieval results
                retrieval_output_file = os.path.join(eval_dir, "retrieval_results.json")
                with open(retrieval_output_file, "w") as f:
                    json.dump(results["retrieval"], f, indent=2)
            except Exception as e:
                print(f"Error in retrieval evaluation: {str(e)}")
                results["retrieval"] = {"error": str(e)}
        
        # Run end-to-end evaluation
        if run_e2e_eval:
            try:
                results["metadata"]["evaluation_components"].append("end_to_end")
                
                # NEW: Store answers in a dictionary
                answers_dict = {}
                
                # We need to generate and store answers for each question
                # before running the actual evaluation to keep the answers
                for item in evaluation_data:
                    question_id = item.get("id", "")
                    question = item.get("question", "")
                    
                    # Generate answer using the function
                    answer = generate_answer(question, db)
                    answers_dict[question_id] = answer
                
                # Save answers to a file
                answers_file = os.path.join(eval_dir, "generated_answers.json")
                with open(answers_file, "w", encoding='utf-8') as f:
                    json.dump(answers_dict, f, indent=2)
                
                # Store the path in job status
                job_status[job_id]["generated_answers_file"] = answers_file
                
                # Now run the actual evaluation
                accuracy, e2e_results = evaluate_end_to_end(
                    answer_query_function=generate_answer,
                    db=db,
                    eval_data=evaluation_data
                )
                
                results["end_to_end"] = {
                    "accuracy": accuracy,
                    "individual_results": [bool(result) for result in e2e_results]
                }
                
                # Save end-to-end results
                e2e_output_file = os.path.join(eval_dir, "e2e_results.json")
                with open(e2e_output_file, "w") as f:
                    json.dump(results["end_to_end"], f, indent=2)
                
                # Analyze failures
                try:
                    analyze_failures(evaluation_data, e2e_results)
                except Exception as e:
                    print(f"Error analyzing failures: {str(e)}")
            except Exception as e:
                print(f"Error in end-to-end evaluation: {str(e)}")
                results["end_to_end"] = {"error": str(e)}
        
        # Run contextual evaluation
        if run_contextual_eval:
            try:
                results["metadata"]["evaluation_components"].append("contextual")
                
                # Import needed components
                from azure_models_with_tenacity import initialize_azure_openai_models
                from deepeval.metrics import (
                    ContextualPrecisionMetric,
                    ContextualRelevancyMetric,
                    ContextualRecallMetric
                )
                from deepeval.test_case import LLMTestCase
                
                # Initialize Azure model
                azure_model, _ = initialize_azure_openai_models()
                
                # Initialize metrics
                precision_metric = ContextualPrecisionMetric(threshold=0.5, model=azure_model, async_mode=False)
                relevancy_metric = ContextualRelevancyMetric(threshold=0.5, model=azure_model, async_mode=False)
                recall_metric = ContextualRecallMetric(threshold=0.5, model=azure_model, async_mode=False)
                
                # Scores and reasons
                precision_scores = []
                relevancy_scores = []
                recall_scores = []
                precision_reasons = []
                relevancy_reasons = []
                recall_reasons = []
                
                # Process a limited number of data points for speed
                subset_size = min(3, len(evaluation_data))
                print(f"Processing {subset_size} items for contextual evaluation")
                
                for i, item in enumerate(evaluation_data[:subset_size]):
                    try:
                        query = item['question']
                        expected_output = item.get('correct_answer', '')
                        
                        # Retrieve chunks
                        retrieved_chunks, _ = vector_search_retrieval(query, db)
                        
                        # Extract text content
                        retrieval_context = [chunk.get('page_content', '') for chunk in retrieved_chunks]
                        
                        # Create test case
                        test_case = LLMTestCase(
                            input=query,
                            actual_output="",  # Not used
                            expected_output=expected_output,
                            retrieval_context=retrieval_context
                        )
                        
                        # Measure metrics
                        precision_metric.measure(test_case)
                        relevancy_metric.measure(test_case)
                        recall_metric.measure(test_case)
                        
                        # Store scores and reasons
                        precision_scores.append(precision_metric.score)
                        relevancy_scores.append(relevancy_metric.score)
                        recall_scores.append(recall_metric.score)
                        precision_reasons.append(precision_metric.reason)
                        relevancy_reasons.append(relevancy_metric.reason)
                        recall_reasons.append(recall_metric.reason)
                    except Exception as e:
                        print(f"Error evaluating contextual metrics for item {i}: {str(e)}")
                        # Add default values
                        precision_scores.append(0)
                        relevancy_scores.append(0)
                        recall_scores.append(0)
                        precision_reasons.append(f"Error: {str(e)}")
                        relevancy_reasons.append(f"Error: {str(e)}")
                        recall_reasons.append(f"Error: {str(e)}")
                
                # Calculate averages
                avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
                avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0
                avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
                
                # Compile results
                results["contextual"] = {
                    "contextual_precision": {
                        "average_score": avg_precision,
                        "individual_scores": precision_scores,
                        "reasons": precision_reasons
                    },
                    "contextual_relevancy": {
                        "average_score": avg_relevancy,
                        "individual_scores": relevancy_scores,
                        "reasons": relevancy_reasons
                    },
                    "contextual_recall": {
                        "average_score": avg_recall,
                        "individual_scores": recall_scores,
                        "reasons": recall_reasons
                    }
                }
                
                # Save contextual results
                contextual_output_file = os.path.join(eval_dir, "contextual_results.json")
                with open(contextual_output_file, "w") as f:
                    json.dump(results["contextual"], f, indent=2)
            except Exception as e:
                print(f"Error in contextual evaluation: {str(e)}")
                results["contextual"] = {"error": str(e)}
        
        # Save comprehensive results
        results_file = os.path.join(eval_dir, "evaluation_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Update job status
        job_status[job_id]["status"] = "evaluation_completed"
        job_status[job_id]["progress"] = 1.0
        job_status[job_id]["current_step"] = "Evaluation completed"
        job_status[job_id]["results_path"] = results_file
        job_status[job_id]["steps_completed"].append("evaluation")
        
    except Exception as e:
        job_status[job_id]["status"] = "evaluation_error"
        job_status[job_id]["error"] = str(e)
        job_status[job_id]["progress"] = -1
        job_status[job_id]["current_step"] = f"Evaluation error: {str(e)}"
        print(f"Error in evaluation: {str(e)}")
        traceback.print_exc()

@app.get("/get-generated-answers/{job_id}")
async def get_generated_answers(job_id: str = Path(...)):
    """Get the answers generated during evaluation"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
    
    # Get the answers file path
    answers_file = job_status[job_id].get("generated_answers_file", "")
    
    if not answers_file or not os.path.exists(answers_file):
        raise HTTPException(status_code=404, 
                          detail=f"Generated answers file not found for job {job_id}")
    
    try:
        # Load the answers from the file
        with open(answers_file, 'r', encoding='utf-8') as f:
            answers = json.load(f)
        
        return answers
    except Exception as e:
        raise HTTPException(status_code=500, 
                          detail=f"Error loading generated answers: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)