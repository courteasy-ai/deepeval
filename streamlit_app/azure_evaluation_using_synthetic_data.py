import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Callable, Set
from tqdm import tqdm
from qdrant_client import QdrantClient
from langchain_openai import AzureOpenAIEmbeddings

# Import OpenAI and Anthropic for answer generation and evaluation
import openai
from openai import AzureOpenAI

import anthropic
from deepeval import evaluate
from deepeval.metrics import (
        ContextualPrecisionMetric,
        ContextualRelevancyMetric,
        ContextualRecallMetric
    )
from deepeval.test_case import LLMTestCase
# Import Azure models with tenacity from your module
from azure_models_with_tenacity import (
    AzureOpenAI as AzureOpenAIWrapper,
    AzureEmbedding,
    initialize_azure_openai_models
)
from deepeval.models import AzureOpenAIModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

AZURE_OPENAI_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_DEPLOYMENT")
AZURE_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT")

# Qdrant configuration - if these aren't set, they'll need to be provided another way
QDRANT_URI = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION")

# Only set environment variables if they are not None
if AZURE_OPENAI_API_KEY:
    os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
if AZURE_OPENAI_ENDPOINT:
    os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
if AZURE_OPENAI_API_VERSION:
    os.environ["OPENAI_API_VERSION"] = AZURE_OPENAI_API_VERSION

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Azure OpenAI model for evaluation
azure_model = AzureOpenAIModel(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    api_version=AZURE_OPENAI_API_VERSION
)

# Initialize Azure OpenAI client for answer generation
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

def calculate_mrr(retrieved_links: List[str], correct_links: Set[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for a single query.
    Returns the reciprocal of the rank of the first relevant item.
    """
    for i, link in enumerate(retrieved_links, 1):
        if link in correct_links:
            return 1 / i
    return 0

def evaluate_retrieval(retrieval_function: Callable, evaluation_data: List[Dict[str, Any]], db: Any) -> Tuple[float, float, float, float, List[float], List[float], List[float]]:
    """
    Evaluate a retrieval function using the provided evaluation data.
    
    Args:
        retrieval_function: Function that takes a query and database and returns chunks and scores
        evaluation_data: List of dictionaries with 'question' and 'correct_chunks' fields
        db: Database object to pass to the retrieval function
        
    Returns:
        Tuple containing:
        - average precision
        - average recall
        - average MRR
        - F1 score
        - list of all precision values
        - list of all recall values
        - list of all MRR values
    """
    precisions = []
    recalls = []
    mrrs = []
    
    for i, item in enumerate(tqdm(evaluation_data, desc="Evaluating Retrieval")):
        try:
            # Log the current query for debugging
            query_snippet = item['question'][:50] + "..." if len(item['question']) > 50 else item['question']
            logger.info(f"Query {i}: '{query_snippet}'")
            logger.info(f"Expected correct chunks: {item['correct_chunks']}")
            
            # Perform retrieval
            retrieved_chunks, scores = retrieval_function(item['question'], db)
            
            # Extract the retrieved chunk IDs
            retrieved_links = [chunk['metadata'].get('chunk_id', chunk['metadata'].get('url', '')) for chunk in retrieved_chunks]
            logger.info(f"Retrieved links: {retrieved_links}")
            logger.info(f"Scores: {scores}")
            
        except Exception as e:
            logging.error(f"Error in retrieval function for query {i}: {e}")
            logging.error(f"Query: {item['question']}")
            continue

        correct_links = set(item['correct_chunks'])
        
        true_positives = len(set(retrieved_links) & correct_links)
        precision = true_positives / len(retrieved_links) if retrieved_links else 0
        recall = true_positives / len(correct_links) if correct_links else 0
        mrr = calculate_mrr(retrieved_links, correct_links)
        
        logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, MRR: {mrr:.4f}")
        
        precisions.append(precision)
        recalls.append(recall)
        mrrs.append(mrr)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(evaluation_data)} items. Current Avg Precision: {sum(precisions) / len(precisions):.4f}, Avg Recall: {sum(recalls) / len(recalls):.4f}, Avg MRR: {sum(mrrs) / len(mrrs):.4f}")
    
    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0
    f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    return avg_precision, avg_recall, avg_mrr, f1, precisions, recalls, mrrs

def evaluate_contextual_retrieval(
    retrieval_function: Callable,
    evaluation_data: List[Dict[str, Any]],
    db: Any,
    custom_model=None,  # Can be either string or model instance
    threshold: float = 0.5,
    include_reason: bool = True,
    strict_mode: bool = False,
    async_mode: bool = True,
    verbose_mode: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a retrieval function using contextual metrics from DeepEval
    
    Args:
        retrieval_function: Function that performs retrieval
        evaluation_data: List of evaluation data points
        db: Database configuration
        custom_model: Model to use for evaluation (string or instance)
        threshold: Threshold for passing (default: 0.5)
        include_reason: Whether to include reasons in output (default: True)
        strict_mode: Whether to use strict evaluation mode (default: False)
        async_mode: Whether to use async evaluation (default: True)
        verbose_mode: Whether to print detailed logs (default: False)
        
    Returns:
        Dictionary containing the evaluation results
    """
    # Initialize metrics
    if custom_model is None:
        from azure_models_with_tenacity import initialize_azure_openai_models
        azure_model, _ = initialize_azure_openai_models()
        custom_model = azure_model  # Use your Azure model wrapper
        
    precision_metric = ContextualPrecisionMetric(
        threshold=threshold,
        model=custom_model,
        include_reason=include_reason,
        strict_mode=strict_mode,
        async_mode=async_mode,
        verbose_mode=verbose_mode
    )
    
    relevancy_metric = ContextualRelevancyMetric(
        threshold=threshold,
        model=custom_model,
        include_reason=include_reason,
        strict_mode=strict_mode,
        async_mode=async_mode,
        verbose_mode=verbose_mode
    )
    
    recall_metric = ContextualRecallMetric(
        threshold=threshold,
        model=custom_model,
        include_reason=include_reason,
        strict_mode=strict_mode,
        async_mode=async_mode,
        verbose_mode=verbose_mode
    )
    
    # Metrics results
    precision_scores = []
    relevancy_scores = []
    recall_scores = []
    precision_reasons = []
    relevancy_reasons = []
    recall_reasons = []
    
    for i, item in enumerate(tqdm(evaluation_data, desc="Evaluating Contextual Retrieval Metrics")):
        try:
            # Get query from the item
            query = item['question']
            expected_output = item.get('correct_answer', '')
            
            # Retrieve chunks
            retrieved_chunks, _ = retrieval_function(query, db)
                        
            # Extract the text content from retrieved chunks
            retrieval_context = [chunk.get('page_content', '') for chunk in retrieved_chunks]
            
            # Create test case for evaluation - using LLMTestCase for text-based metrics
            test_case = LLMTestCase(
                input=query,
                actual_output="",  # Placeholder - not used for metrics calculation
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
            
            if include_reason:
                precision_reasons.append(precision_metric.reason)
                relevancy_reasons.append(relevancy_metric.reason)
                recall_reasons.append(recall_metric.reason)
            
            # Log progress
            if (i + 1) % 5 == 0 or verbose_mode:
                logger.info(f"Processed {i + 1}/{len(evaluation_data)} items")
                logger.info(f"Current Avg Precision: {sum(precision_scores) / len(precision_scores):.4f}")
                logger.info(f"Current Avg Relevancy: {sum(relevancy_scores) / len(relevancy_scores):.4f}")
                logger.info(f"Current Avg Recall: {sum(recall_scores) / len(recall_scores):.4f}")
                
        except Exception as e:
            logger.error(f"Error evaluating item {i}: {e}")
            # Add default values for failed evaluations
            precision_scores.append(0)
            relevancy_scores.append(0)
            recall_scores.append(0)
            if include_reason:
                error_reason = f"Evaluation failed due to: {str(e)}"
                precision_reasons.append(error_reason)
                relevancy_reasons.append(error_reason)
                recall_reasons.append(error_reason)
    
    # Calculate averages
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    
    # Compile results
    results = {
        "contextual_precision": {
            "average_score": avg_precision,
            "individual_scores": precision_scores,
            "reasons": precision_reasons if include_reason else None
        },
        "contextual_relevancy": {
            "average_score": avg_relevancy,
            "individual_scores": relevancy_scores,
            "reasons": relevancy_reasons if include_reason else None
        },
        "contextual_recall": {
            "average_score": avg_recall,
            "individual_scores": recall_scores,
            "reasons": recall_reasons if include_reason else None
        }
    }
    
    return results


def generate_answer(query: str, db: Dict) -> str:
    """
    Generate an answer to a query using document retrieval and Azure OpenAI
    
    Args:
        query: The user's question
        db: Dictionary with Qdrant client configuration
        
    Returns:
        Generated answer
    """
    try:
        # Step 1: Retrieve relevant chunks
        chunks, _ = vector_search_retrieval(query, db)
        
        # Step 2: Create a context from the chunks
        context = "\n\n".join([chunk["page_content"] for chunk in chunks])
        
        # Step 3: Generate answer using Azure OpenAI
        prompt = f"""
            You have been tasked with helping us to answer the following query: 
            <query>
            {query}
            </query>
            You have access to the following documents which are meant to provide context as you answer the query:
            <documents>
            {context}
            </documents>
            Please remain faithful to the underlying context, and only deviate from it if you are 100% sure that you know the answer already. 
            Answer the question now, and avoid providing preamble such as 'Here is the answer', etc
        """
        
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a helpful legal assistant specialized in Indian law."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "I apologize, but I encountered an error while trying to answer your question."

def evaluate_end_to_end(answer_query_function: Callable, db: Dict, eval_data: List[Dict]) -> Tuple[float, List[bool]]:
    """
    Evaluate the end-to-end performance of the question answering system using Azure OpenAI
    
    Args:
        answer_query_function: Function that generates answers to queries
        db: Database configuration
        eval_data: Evaluation dataset containing questions and correct answers
        
    Returns:
        Tuple of (accuracy, list of results for each question)
    """
    correct_answers = 0
    results = []
    total_questions = len(eval_data)
    
    for i, item in enumerate(tqdm(eval_data, desc="Evaluating End-to-End")):
        query = item['question']
        correct_answer = item['correct_answer']
        
        # Generate answer
        try:
            generated_answer = answer_query_function(query, db)
        except Exception as e:
            logger.error(f"Error generating answer for question {i}: {e}")
            results.append(False)
            continue
        
        # Create evaluation prompt
        system_message = "You are an AI assistant tasked with evaluating the correctness of answers to questions about legal documentation."
        
        evaluation_prompt = f"""
        Question: {query}
        
        Correct Answer: {correct_answer}
        
        Generated Answer: {generated_answer}
        
        Is the Generated Answer correct based on the Correct Answer? You should pay attention to the substance of the answer, and ignore minute details that may differ. 
        
        Small differences or changes in wording don't matter. If the generated answer and correct answer are saying essentially the same thing then that generated answer should be marked correct. 
        
        However, if there is any critical piece of information which is missing from the generated answer in comparison to the correct answer, then we should mark this as incorrect. 
        
        Finally, if there are any direct contradictions between the correct answer and generated answer, we should deem the generated answer to be incorrect.
        
        Respond in the following XML format:
        <evaluation>
        <content>
        <explanation>Your explanation here</explanation>
        <is_correct>true/false</is_correct>
        </content>
        </evaluation>
        """
        
        try:
            # Use Azure OpenAI to evaluate the answer
            response = openai_client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0,
                response_format={"type": "text"}
            )
            
            response_text = response.choices[0].message.content
            
            # Parse the XML
            try:
                # Try parsing with the content tags
                evaluation = ET.fromstring(response_text)
                explanation = evaluation.find('.//explanation').text
                is_correct_text = evaluation.find('.//is_correct').text.lower()
                is_correct = is_correct_text == 'true'
            except ET.ParseError:
                # If the response doesn't include the full XML structure, try adding tags
                try:
                    # Some models might omit the outer tags
                    full_xml = response_text
                    if not response_text.startswith('<evaluation>'):
                        full_xml = "<evaluation>" + response_text
                    if not response_text.endswith('</evaluation>'):
                        full_xml = full_xml + "</evaluation>"
                    
                    evaluation = ET.fromstring(full_xml)
                    explanation = evaluation.find('.//explanation').text
                    is_correct_text = evaluation.find('.//is_correct').text.lower()
                    is_correct = is_correct_text == 'true'
                except ET.ParseError as inner_e:
                    # If still not able to parse properly, use regex as a fallback
                    logger.warning(f"Could not parse XML response for question {i}, using regex fallback")
                    import re
                    is_correct_match = re.search(r'<is_correct>(true|false)</is_correct>', response_text, re.IGNORECASE)
                    if is_correct_match:
                        is_correct = is_correct_match.group(1).lower() == 'true'
                    else:
                        # Last resort: check if "true" appears prominently in the response
                        is_correct = 'true' in response_text.lower() and 'false' not in response_text.lower()
                    
                    explanation_match = re.search(r'<explanation>(.*?)</explanation>', response_text, re.IGNORECASE | re.DOTALL)
                    explanation = explanation_match.group(1) if explanation_match else "No explanation provided"
            
            if is_correct:
                correct_answers += 1
            results.append(is_correct)
            
            # Log the result
            logger.info(f"Question {i + 1}/{total_questions}: {is_correct}")
            logger.debug(f"Explanation: {explanation}")
            
        except Exception as e:
            logger.error(f"Unexpected error for question {i}: {e}")
            logger.error(f"Response text: {response_text if 'response_text' in locals() else 'N/A'}")
            results.append(False)
        
        # Report progress
        if (i + 1) % 5 == 0:
            current_accuracy = correct_answers / (i + 1)
            print(f"Processed {i + 1}/{total_questions} questions. Current Accuracy: {current_accuracy:.4f}")
        
        # Rate limit to avoid API throttling
        time.sleep(1)
    
    accuracy = correct_answers / total_questions
    return accuracy, results

def load_evaluation_data(file_path: str) -> List[Dict[str, Any]]:
    """Load the synthetic test dataset generated earlier and perform basic validation"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Check for variety in questions
    questions = [item['question'] for item in data]
    unique_questions = set(questions)
    logger.info(f"Loaded {len(data)} items with {len(unique_questions)} unique questions")
    
    if len(unique_questions) < len(questions):
        logger.warning("Some questions are duplicated in the evaluation data!")
    
    # Check for variety in correct chunks
    correct_chunks_counts = [len(item['correct_chunks']) for item in data]
    avg_correct = sum(correct_chunks_counts) / len(correct_chunks_counts)
    min_correct = min(correct_chunks_counts)
    max_correct = max(correct_chunks_counts)
    logger.info(f"Correct chunks per question: min={min_correct}, avg={avg_correct:.2f}, max={max_correct}")
    
    # Log a couple of example items for verification
    if len(data) > 0:
        logger.info("Example evaluation item:")
        logger.info(json.dumps(data[0], indent=2))
    
    if len(data) > 1:
        logger.info("Another example evaluation item:")
        logger.info(json.dumps(data[1], indent=2))
    
    return data

def setup_qdrant_client():
    """Set up and return a Qdrant client"""
    client = QdrantClient(
        url=QDRANT_URI,
        api_key=QDRANT_API_KEY,
        port=443, timeout=10.0
    )
    
    return {"client": client, "collection_name": QDRANT_COLLECTION}

def setup_azure_embeddings():
    """Set up and return Azure OpenAI embeddings model"""
    # Initialize the embedding model
    embedding_model = AzureOpenAIEmbeddings(
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
        api_version="2023-05-15"
    )
    
    return embedding_model

def check_qdrant_collection(db: Dict) -> bool:
    """Check if the Qdrant collection has enough varied data"""
    client = db["client"]
    collection_name = db["collection_name"]
    
    try:
        # Get collection info
        collection_info = client.get_collection(collection_name=collection_name)
        logger.info(f"Collection '{collection_name}' size: {collection_info.points_count} points")
        
        # Sample a few random points if the collection is not empty
        if collection_info.points_count > 0:
            # Get some random IDs to sample
            sample_size = min(5, collection_info.points_count)
            if sample_size > 0:
                # This uses the scroll method to get a few random points
                sample_points = client.scroll(
                    collection_name=collection_name,
                    limit=sample_size,
                    with_payload=True
                )[0]
                
                logger.info(f"Sample point payload keys:")
                for point in sample_points:
                    logger.info(f"ID {point.id}: {list(point.payload.keys())}")
                    # Show a small snippet of content if available
                    if 'content' in point.payload:
                        content_preview = point.payload['content'][:100] + "..." if len(point.payload['content']) > 100 else point.payload['content']
                        logger.info(f"Content preview: {content_preview}")
        
        return True
    except Exception as e:
        logger.error(f"Error checking Qdrant collection: {e}")
        return False

def vector_search_retrieval(query: str, db: Dict) -> Tuple[List[Dict], List[float]]:
    """
    Perform vector search retrieval using Qdrant
    
    Args:
        query: The search query
        db: Dictionary with Qdrant client, collection name, and embedding model
        
    Returns:
        Tuple of (retrieved chunks, scores)
    """
    client = db["client"]
    collection_name = db["collection_name"]
    # Use the embedding model from db if available, otherwise create a new one
    embedding_model = db.get("embedding_model", setup_azure_embeddings())
    
    # Log the query for debugging
    logger.info(f"Processing query: {query}")
    
    # Get query embedding
    query_vector = embedding_model.embed_query(query)
    
    # Debug: log a few values from the embedding to verify they're different
    logger.info(f"Query embedding sample (first 3 values): {query_vector[:3]}")
    
    # Search Qdrant
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5  # Retrieve top 5 chunks
    )
    
    # Debug: log the IDs of returned results
    result_ids = [point.id for point in search_result]
    logger.info(f"Retrieved point IDs: {result_ids}")
    logger.info(f"Number of results returned: {len(search_result)}")
    
    # Format results for evaluation
    chunks = []
    scores = []
    
    for point in search_result:
        # Get the full payload for the point
        point_details = client.retrieve(
            collection_name=collection_name,
            ids=[point.id]
        )[0]
        
        chunk = {
            "page_content": point_details.payload.get("content", ""),
            "metadata": {
                "chunk_id": point_details.payload.get("chunk_id", ""),
                "source": point_details.payload.get("source", "")
            }
        }
        
        chunks.append(chunk)
        scores.append(point.score)
    
    return chunks, scores

def analyze_failures(evaluation_data, results):
    """
    Analyze cases where the end-to-end evaluation failed
    
    Args:
        evaluation_data: The evaluation dataset
        results: Boolean list indicating success/failure for each question
    """
    failures = []
    for i, (item, is_correct) in enumerate(zip(evaluation_data, results)):
        if not is_correct:
            failures.append({
                "question_id": item.get("id", str(i)),
                "question": item["question"],
                "correct_chunks": item["correct_chunks"],
                "correct_answer": item["correct_answer"]
            })
    
    # Save failures for detailed analysis
    with open("evaluation_failures.json", "w") as f:
        json.dump(failures, f, indent=2)
    
    logger.info(f"Found {len(failures)} failures out of {len(results)} questions")
    logger.info(f"Failure rate: {len(failures)/len(results):.2%}")
    logger.info(f"Detailed failure information saved to evaluation_failures.json")

def run_evaluation(
    run_retrieval_eval=True, 
    run_e2e_eval=True, 
    run_contextual_eval=True,
    use_azure_for_evaluation=True
):
    """
    Run comprehensive evaluation with standard and contextual metrics
    
    Args:
        run_retrieval_eval: Whether to run retrieval evaluation
        run_e2e_eval: Whether to run end-to-end evaluation
        run_contextual_eval: Whether to run contextual evaluation
        use_azure_for_evaluation: Whether to use Azure OpenAI for evaluation
    """
    # Create results directory
    os.makedirs("evaluation_results", exist_ok=True)
    
    # 1. Load the synthetic test dataset
    eval_data_path = "./qdrant_deepeval_data/qdrant_deepeval_test_cases.json"
    
    if not os.path.exists(eval_data_path):
        logger.error(f"Evaluation data file not found: {eval_data_path}")
        return
    
    logger.info(f"Loading evaluation data from {eval_data_path}")
    evaluation_data = load_evaluation_data(eval_data_path)
    logger.info(f"Loaded {len(evaluation_data)} evaluation items")
    
    # 2. Set up the database connection
    db = setup_qdrant_client()
    
    # 3. Initialize Azure models for evaluation if requested
    local_azure_model = None
    if run_contextual_eval and use_azure_for_evaluation:
        try:
            # Use the initialize_azure_openai_models function from your module
            local_azure_model, _ = initialize_azure_openai_models()
            logger.info("Successfully initialized Azure model for evaluation")
        except Exception as e:
            logger.error(f"Failed to initialize Azure model for evaluation: {e}")
            logger.info("Falling back to default GPT-4o model")
            local_azure_model = None
    
    # Track all results
    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_questions": len(evaluation_data),
            "evaluation_components": [],
            "using_azure_for_evaluation": use_azure_for_evaluation and local_azure_model is not None
        }
    }
    
    # 4. Run retrieval evaluation
    if run_retrieval_eval:
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
        retrieval_output_file = "evaluation_results/retrieval_results.json"
        with open(retrieval_output_file, "w") as f:
            json.dump(results["retrieval"], f, indent=2)
        
        logger.info(f"Retrieval evaluation complete. Results saved to {retrieval_output_file}")
    
    # 5. Run end-to-end evaluation
    if run_e2e_eval:
        logger.info("Starting end-to-end evaluation...")
        results["metadata"]["evaluation_components"].append("end_to_end")
        
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
        e2e_output_file = "evaluation_results/e2e_results_without_reranking.json"
        with open(e2e_output_file, "w") as f:
            json.dump(results["end_to_end"], f, indent=2)
        
        logger.info(f"End-to-end evaluation complete. Results saved to {e2e_output_file}")
        
        # Analyze failures
        analyze_failures(evaluation_data, e2e_results)
    
    # 6. Run contextual evaluation
    if run_contextual_eval:
        logger.info("Starting contextual evaluation...")
        results["metadata"]["evaluation_components"].append("contextual")
        
        # Determine which model to use
        model_for_evaluation = local_azure_model if local_azure_model is not None else azure_model
        model_type = "Azure OpenAI with tenacity retry" if local_azure_model is not None else "Default Azure OpenAI model"
        logger.info(f"Using {model_type} for contextual evaluation")
        
        # Run contextual evaluation with the selected model
        contextual_results = evaluate_contextual_retrieval(
            retrieval_function=vector_search_retrieval,
            evaluation_data=evaluation_data,
            db=db,
            custom_model=model_for_evaluation,
            threshold=0.5,
            include_reason=True,
            strict_mode=False,
            async_mode=True,
            verbose_mode=False
        )
        
        results["contextual"] = contextual_results
        
        # Save contextual results
        contextual_output_file = "evaluation_results/contextual_results.json"
        with open(contextual_output_file, "w") as f:
            json.dump(results["contextual"], f, indent=2)
        
        logger.info(f"Contextual evaluation complete. Results saved to {contextual_output_file}")
    
    # 7. Save comprehensive results
    comprehensive_output_file = "evaluation_results/comprehensive_results.json"
    with open(comprehensive_output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # 8. Print summary
    logger.info("=" * 70)
    logger.info("COMPREHENSIVE EVALUATION RESULTS")
    logger.info("=" * 70)
    
    if run_retrieval_eval:
        logger.info("Standard Retrieval Metrics:")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"MRR: {mrr:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
    
    if run_e2e_eval:
        logger.info("\nEnd-to-End Metrics:")
        logger.info(f"Accuracy: {accuracy:.4f}")
    
    if run_contextual_eval:
        logger.info("\nContextual Metrics:")
        logger.info(f"Contextual Precision: {results['contextual']['contextual_precision']['average_score']:.4f}")
        logger.info(f"Contextual Relevancy: {results['contextual']['contextual_relevancy']['average_score']:.4f}")
        logger.info(f"Contextual Recall: {results['contextual']['contextual_recall']['average_score']:.4f}")
    
    logger.info(f"\nDetailed results saved to {comprehensive_output_file}")
    
    return results

if __name__ == "__main__":
    # Run the comprehensive evaluation - both retrieval and end-to-end
    run_evaluation(run_retrieval_eval=True, run_e2e_eval=True, run_contextual_eval=True)