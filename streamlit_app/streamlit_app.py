import streamlit as st
import requests
import json
import time
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
from datetime import datetime

# Constants
API_URL = "http://localhost:8000"  # FastAPI server URL

# Page configuration
st.set_page_config(
    page_title="RAG Evaluation Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session states
if "active_job" not in st.session_state:
    st.session_state.active_job = None
if "jobs" not in st.session_state:
    st.session_state.jobs = []
if "refresh_jobs" not in st.session_state:
    st.session_state.refresh_jobs = True
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Upload"
if "questions" not in st.session_state:
    st.session_state.questions = {}
if "ratings" not in st.session_state:
    st.session_state.ratings = {}
if "question_feedback" not in st.session_state:
    st.session_state.question_feedback = {}
if "question_feedback_after_eval" not in st.session_state:
    st.session_state.question_feedback_after_eval = {}

def create_sidebar():
    """Create the sidebar with app description and navigation"""
    with st.sidebar:
        st.title("📑 RAG Evaluation App")
        st.markdown("---")
        
        st.markdown("""
        ### About
        This app allows you to:
        - Upload PDF documents
        - Extract and process text
        - Generate synthetic test data
        - Evaluate retrieval performance
        - Analyze question-answering capabilities
        """)
        
        st.markdown("---")
        
        # Job selection section
        st.subheader("Active Jobs")
        
        # Refresh jobs button
        if st.button("🔄 Refresh Jobs"):
            st.session_state.refresh_jobs = True
            
        # Fetch jobs if needed
        if st.session_state.refresh_jobs:
            try:
                response = requests.get(f"{API_URL}/jobs")
                if response.status_code == 200:
                    st.session_state.jobs = response.json()
                    st.session_state.refresh_jobs = False
            except Exception as e:
                st.error(f"Error fetching jobs: {str(e)}")
        
        # Display job selection
        if st.session_state.jobs:
            job_options = {}
            for job in st.session_state.jobs:
                # Create a status indicator
                status = job["status"]
                if status == "evaluation_completed":
                    status_icon = "✅"
                elif status.endswith("error"):
                    status_icon = "❌"
                elif status in ["uploaded", "extracting_chunks", "adding_to_qdrant", "generating_synthetic_data"]:
                    status_icon = "🔄"
                elif status == "synthetic_data_generated":
                    status_icon = "⏳"
                elif status == "evaluating":
                    status_icon = "📊"
                else:
                    status_icon = "❓"
                
                # Get PDF filename
                pdf_files = job["details"].get("pdf_files", ["Unknown"])
                pdf_name = pdf_files[0] if pdf_files else "Unknown"
                
                # Format option text
                option_text = f"{status_icon} {pdf_name} ({job['job_id'][-6:]})"
                job_options[option_text] = job["job_id"]
            
            # Add a "Start New" option
            job_options["➕ Start New Upload"] = "new_upload"
            
            selected_job_text = st.selectbox(
                "Select Job",
                options=list(job_options.keys()),
                index=0
            )
            
            selected_job_id = job_options[selected_job_text]
            
            if selected_job_id == "new_upload":
                st.session_state.active_job = None
                st.session_state.current_tab = "Upload"
            elif selected_job_id != st.session_state.active_job:
                st.session_state.active_job = selected_job_id
                
                # If job is in evaluation completed state, switch to Results tab
                for job in st.session_state.jobs:
                    if job["job_id"] == selected_job_id:
                        if job["status"] == "evaluation_completed":
                            st.session_state.current_tab = "Results"
                        elif job["status"] == "synthetic_data_generated":
                            st.session_state.current_tab = "Question Feedback"
                        break
        else:
            st.info("No jobs available. Use the upload tab to create one.")
        
        # Navigation
        st.markdown("---")
        st.subheader("Navigation")
        
        tabs = {
            "Upload": "📤 Upload Document",
            "Process": "⚙️ Process Document",
            "Evaluate": "📊 Evaluate RAG",
            "Results": "📈 View Results"
        }
        
        for tab_id, tab_name in tabs.items():
            if st.button(tab_name, key=f"nav_{tab_id}"):
                st.session_state.current_tab = tab_id
        st.markdown("---")
        st.markdown("RAG Evaluation Pipeline | Powered by DeepEval and Qdrant")

# Helper function to check job status
def check_job_status(job_id):
    try:
        response = requests.get(f"{API_URL}/job-status/{job_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error checking job status: {str(e)}")
        return None

# Helper function to display task progress
def display_task_progress(job_id, completion_message, auto_refresh=True):
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Check job status until completed or failed
    task_complete = False
    start_time = time.time()
    
    while not task_complete:
        status_data = check_job_status(job_id)
        
        if status_data:
            status = status_data.get("status", "")
            progress = status_data.get("progress", 0)
            details = status_data.get("details", {})
            current_step = details.get("current_step", "")
            
            # Update progress bar - ensure it's between 0 and 1
            if progress >= 0:
                # Scale to 0-1 range for progress bar
                progress_bar.progress(min(progress, 1.0))
            
            if status in ["evaluation_completed", "synthetic_data_generated"]:
                progress_bar.progress(1.0)
                status_text.success(f"{completion_message}: {current_step}")
                task_complete = True
                return status_data
            elif status.endswith("error"):
                progress_bar.progress(1.0)
                error_msg = details.get("error", "Unknown error")
                status_text.error(f"Task failed: {error_msg}")
                task_complete = True
                return status_data
            else:
                # Show current progress
                status_text.info(f"Status: {status} - {current_step}")
                
                # For long-running tasks, don't block indefinitely
                if not auto_refresh:
                    break
        
        time.sleep(2)  # Check every 2 seconds
    
    return None

# Function to display results in a nice format
def display_evaluation_results(results):
    if not results:
        st.warning("No results available.")
        return
    
    # Create tabs for different result types
    tabs = st.tabs(["Overview", "Retrieval Metrics", "End-to-End Metrics", "Contextual Metrics"])
    
    with tabs[0]:
        st.subheader("Evaluation Overview")
        metadata = results.get("metadata", {})
        
        # Create two columns for metadata display
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Number of Questions", metadata.get("num_questions", "N/A"))
            st.write("**Evaluation Components:**")
            components = metadata.get("evaluation_components", [])
            for component in components:
                st.write(f"- {component}")
        
        with col2:
            st.write("**Timestamp:**", metadata.get("timestamp", "N/A"))
            st.write("**Using Azure for Evaluation:**", 
                     "Yes" if metadata.get("using_azure_for_evaluation", False) else "No")
    
    with tabs[1]:
        st.subheader("Retrieval Metrics")
        if "retrieval" in results:
            retrieval = results["retrieval"]
            
            # Display metrics in a nice grid
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Precision", f"{retrieval.get('precision', 0):.4f}")
            col2.metric("Recall", f"{retrieval.get('recall', 0):.4f}")
            col3.metric("MRR", f"{retrieval.get('mrr', 0):.4f}")
            col4.metric("F1 Score", f"{retrieval.get('f1', 0):.4f}")
            
            # Display individual results if available
            if "individual_results" in retrieval:
                individual = retrieval["individual_results"]
                
                # Create dataframe for visualization
                if "precisions" in individual and "recalls" in individual and "mrrs" in individual:
                    df = pd.DataFrame({
                        "Query": [f"Q{i+1}" for i in range(len(individual["precisions"]))],
                        "Precision": individual["precisions"],
                        "Recall": individual["recalls"],
                        "MRR": individual["mrrs"]
                    })
                    
                    # Plot metrics
                    fig = px.bar(
                        df, x="Query", y=["Precision", "Recall", "MRR"],
                        title="Metrics per Query",
                        labels={"value": "Score", "variable": "Metric"},
                        barmode="group"  # This groups the bars by query
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    # Show data table
                    st.dataframe(df)
        else:
            st.info("No retrieval metrics available.")
    
    with tabs[2]:
        st.subheader("End-to-End Metrics")
        if "end_to_end" in results:
            e2e = results["end_to_end"]
            
            # Display accuracy metric
            st.metric("Answer Accuracy", f"{e2e.get('accuracy', 0):.4f}")
            
            # Display individual results if available
            if "individual_results" in e2e:
                # Create a visualization
                results_list = e2e["individual_results"]
                correct = sum(1 for r in results_list if r)
                incorrect = len(results_list) - correct
                
                # Create pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=['Correct', 'Incorrect'],
                    values=[correct, incorrect],
                    hole=.4,
                    marker_colors=['#4CAF50', '#F44336']
                )])
                fig.update_layout(title_text="Correct vs Incorrect Answers")
                st.plotly_chart(fig, use_container_width=True)
                
                # Show per-question results
                df = pd.DataFrame({
                    "Question": [f"Q{i+1}" for i in range(len(results_list))],
                    "Correct": ["✓" if r else "✗" for r in results_list]
                })
                st.dataframe(df)
        else:
            st.info("No end-to-end metrics available.")
    
    with tabs[3]:
        st.subheader("Contextual Metrics")
        if "contextual" in results:
            contextual = results["contextual"]
            
            # Display metrics in a grid
            col1, col2, col3 = st.columns(3)
            
            # Contextual Precision
            if "contextual_precision" in contextual:
                precision = contextual["contextual_precision"]
                col1.metric("Contextual Precision", f"{precision.get('average_score', 0):.4f}")
            
            # Contextual Relevancy
            if "contextual_relevancy" in contextual:
                relevancy = contextual["contextual_relevancy"]
                col2.metric("Contextual Relevancy", f"{relevancy.get('average_score', 0):.4f}")
            
            # Contextual Recall
            if "contextual_recall" in contextual:
                recall = contextual["contextual_recall"]
                col3.metric("Contextual Recall", f"{recall.get('average_score', 0):.4f}")
            
            # Create a combined visualization if possible
            if all(k in contextual for k in ["contextual_precision", "contextual_relevancy", "contextual_recall"]):
                try:
                    precision_scores = contextual["contextual_precision"]["individual_scores"]
                    relevancy_scores = contextual["contextual_relevancy"]["individual_scores"]
                    recall_scores = contextual["contextual_recall"]["individual_scores"]
                    
                    if len(precision_scores) == len(relevancy_scores) == len(recall_scores):
                        df = pd.DataFrame({
                            "Query": [f"Q{i+1}" for i in range(len(precision_scores))],
                            "Precision": precision_scores,
                            "Relevancy": relevancy_scores,
                            "Recall": recall_scores
                        })
                        
                        # Plot radar chart for overall
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatterpolar(
                            r=[
                                contextual["contextual_precision"]["average_score"],
                                contextual["contextual_relevancy"]["average_score"],
                                contextual["contextual_recall"]["average_score"]
                            ],
                            theta=['Precision', 'Relevancy', 'Recall'],
                            fill='toself',
                            name='Contextual Metrics'
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )),
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Also show a line chart
                        # Show a bar chart
                        fig2 = px.bar(
                            df, x="Query", y=["Precision", "Relevancy", "Recall"],
                            title="Contextual Metrics per Query",
                            labels={"value": "Score", "variable": "Metric"},
                            barmode="group"  # This groups the bars by query
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Show data table
                        st.dataframe(df)
                        
                        # Option to view reasons
                        if st.checkbox("Show Evaluation Reasons"):
                            st.subheader("Evaluation Reasoning")
                            
                            # Create expandable sections for each question
                            for i in range(len(precision_scores)):
                                with st.expander(f"Question {i+1} Reasoning"):
                                    st.write("**Precision Reasoning:**")
                                    st.write(contextual["contextual_precision"]["reasons"][i])
                                    
                                    st.write("**Relevancy Reasoning:**")
                                    st.write(contextual["contextual_relevancy"]["reasons"][i])
                                    
                                    st.write("**Recall Reasoning:**")
                                    st.write(contextual["contextual_recall"]["reasons"][i])
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
        else:
            st.info("No contextual metrics available.")

# Upload Document page
def upload_document_page():
    st.title("📤 Upload Document")
    
    with st.form("upload_document_form"):
        st.subheader("Upload PDF File")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        
        st.subheader("Processing Configuration")
        
        # Collection name
        collection_name = "test_streamlit_app"
        
        # Chunking configuration
        col1, col2, col3 = st.columns(3)
        with col1:
            chunk_size = st.number_input(
                "Chunk Size (tokens)", 
                min_value=128, 
                max_value=4096, 
                value=1024,
                help="Size of text chunks in tokens"
            )
        with col2:
            chunk_overlap = st.number_input(
                "Chunk Overlap (tokens)", 
                min_value=0, 
                max_value=512, 
                value=128,
                help="Overlap between consecutive chunks in tokens"
            )
        with col3:
            encoding = st.selectbox(
                "Text Encoding", 
                options=["utf-8", "utf-16", "ascii"],
                help="Character encoding for text extraction"
            )
        
        submit_button = st.form_submit_button("Upload & Process")
    
    if submit_button:
        if not uploaded_file:
            st.error("Please upload a PDF file.")
        else:
            try:
                # First create the collection
                response = requests.post(
                    f"{API_URL}/create-collection/",
                    json={"collection_name": collection_name}
                )
                
                if response.status_code != 200:
                    st.error(f"Error creating collection: {response.text}")
                    return
                
                # Now upload the PDF
                files = {"file": uploaded_file}
                data = {
                    "collection_name": "test_streamlit_app",
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "encoding": encoding
                }
                
                with st.spinner("Uploading document..."):
                    response = requests.post(
                        f"{API_URL}/upload-pdf/",
                        files=files,
                        data=data
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    job_id = result["job_id"]
                    
                    st.success(f"Document uploaded! Job ID: {job_id}")
                    
                    # Set as active job
                    st.session_state.active_job = job_id
                    st.session_state.refresh_jobs = True
                    
                    # Show processing progress
                    st.subheader("Processing Progress")
                    st.info("The document is now being processed. This will take a few minutes.")
                    
                    status_data = display_task_progress(job_id, "Processing completed", auto_refresh=False)
                    if status_data:
                        st.session_state.current_tab = "Process"
                        st.rerun()
                else:
                    st.error(f"Error uploading document: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

def process_document_page():
    st.title("⚙️ Process Document")
    
    if not st.session_state.active_job:
        st.warning("No active job selected. Please upload a document or select a job.")
        return
    
    job_id = st.session_state.active_job
    
    # Get job status
    status_data = check_job_status(job_id)
    
    if not status_data:
        st.error("Failed to fetch job status. Please try again.")
        return
    
    # Display current status
    status = status_data["status"]
    progress = status_data["progress"]
    details = status_data["details"]
    
    # Show job details
    st.subheader("Job Details")
    st.write(f"**Job ID:** {job_id}")
    st.write(f"**Status:** {status}")
    st.write(f"**Current Step:** {details.get('current_step', 'Unknown')}")
    
    # Show PDF files
    st.write(f"**Document:** {', '.join(details.get('pdf_files', ['Unknown']))}")
    
    # Show chunking configuration if available
    chunking_config = details.get("chunking_config", {})
    if chunking_config:
        st.write(f"**Chunk Size:** {chunking_config.get('chunk_size', 'Unknown')} tokens")
        st.write(f"**Chunk Overlap:** {chunking_config.get('chunk_overlap', 'Unknown')} tokens")
        st.write(f"**Encoding:** {chunking_config.get('encoding', 'Unknown')}")
    
    # Show collection name
    st.write(f"**Collection Name:** {details.get('collection_name', 'Unknown')}")
    
    # Display progress bar
    st.subheader("Processing Progress")
    
    # Create a progress bar
    if progress >= 0:
        st.progress(min(progress, 1.0))
    else:
        st.progress(0)
    
    # Display status message
    if status.endswith("error"):
        st.error(f"Error: {details.get('error', 'Unknown error')}")
    elif status == "evaluation_completed":
        st.success("Processing and evaluation completed successfully!")
    elif status == "synthetic_data_generated":
        st.success("Processing completed successfully! Ready for question feedback and evaluation.")
    else:
        st.info(f"Current status: {status} - {details.get('current_step', '')}")
    
    # Steps completed
    steps_completed = details.get("steps_completed", [])
    
    # Create a step tracker
    st.subheader("Processing Steps")
    
    steps = [
        ("PDF Upload", "uploaded" in status or len(steps_completed) > 0),
        ("Chunk Extraction", "chunk_extraction" in steps_completed),
        ("Qdrant Upload", "qdrant_upload" in steps_completed),
        ("Synthetic Data Generation", "synthetic_data_generation" in steps_completed),
        ("Evaluation", "evaluation" in steps_completed)
    ]
    
    for step_name, is_completed in steps:
        if is_completed:
            st.markdown(f"✅ **{step_name}**")
        else:
            st.markdown(f"⏳ {step_name}")
    
    # Add refresh button
    if st.button("Refresh Status"):
        st.session_state.refresh_jobs = True
        st.rerun()
    
    # Navigation options based on status
    st.subheader("Next Steps")
    
    # Add error details if available
    if "error" in details and details["error"]:
        st.error(f"Error details: {details['error']}")
    
    # If synthetic data is generated, show sample questions right on this page
    if status == "synthetic_data_generated":
        st.subheader("Review Generated Questions")
        
        # Initialize session state for question feedback if needed
        if "question_feedback" not in st.session_state:
            st.session_state.question_feedback = {}
            
        # Try to get synthetic data
        got_synthetic_data = False
        try:
            # First check if synthetic_data_file is directly in the status_data
            synthetic_data_file = details.get("synthetic_data_file", "")
            
            # If no synthetic data file path is found, try the API method instead
            if not synthetic_data_file:
                try:
                    # Try to get synthetic data directly from API
                    with st.spinner("Fetching synthetic questions from server..."):
                        response = requests.get(f"{API_URL}/get-synthetic-data/{job_id}")
                        
                        if response.status_code == 200:
                            synthetic_data = response.json()
                            
                            # If successful, we can proceed using this data
                            if synthetic_data and len(synthetic_data) > 0:
                                # Sample 30% of questions if not already sampled
                                if job_id not in st.session_state.question_feedback:
                                    # Sample 30% of questions, but at least 1 and at most 10
                                    sample_size = max(1, min(10, int(len(synthetic_data) * 0.3)))
                                    sampled_questions = random.sample(synthetic_data, sample_size)
                                    
                                    # Store in session state
                                    st.session_state.question_feedback[job_id] = {
                                        "questions": sampled_questions,
                                        "feedback": {}
                                    }
                                
                                # Display questions and collect feedback
                                display_question_feedback(job_id)
                                got_synthetic_data = True
                            else:
                                st.error("No synthetic data available from API.")
                        else:
                            st.error(f"Failed to retrieve synthetic questions: {response.text}")
                except Exception as e:
                    st.error(f"Error fetching synthetic data from API: {str(e)}")
            else:
                # We have a file path, try to load the file
                synthetic_data = load_synthetic_data(synthetic_data_file)
                
                if not synthetic_data or len(synthetic_data) == 0:
                    st.error("No synthetic data available in the file.")
                else:
                    # Sample 30% of questions if not already sampled
                    if job_id not in st.session_state.question_feedback:
                        # Sample 30% of questions, but at least 1 and at most 10
                        sample_size = max(1, min(10, int(len(synthetic_data) * 0.3)))
                        sampled_questions = random.sample(synthetic_data, sample_size)
                        
                        # Store in session state
                        st.session_state.question_feedback[job_id] = {
                            "questions": sampled_questions,
                            "feedback": {}
                        }
                    
                    # Display questions and collect feedback
                    display_question_feedback(job_id)
                    got_synthetic_data = True
                
        except Exception as e:
            st.error(f"Error processing synthetic data: {str(e)}")
            import traceback
            st.write(traceback.format_exc())
        
        # If we couldn't get synthetic data, provide clear message and options
        if not got_synthetic_data:
            st.warning("⚠️ Unable to retrieve synthetic questions for review.")
            st.info("You can still proceed to evaluation without reviewing questions.")
        
        # Button to proceed to evaluation
        if st.button("Proceed to Evaluation"):
            st.session_state.current_tab = "Evaluate"
            st.rerun()
            
    elif status == "evaluation_completed":
        if st.button("View Results"):
            st.session_state.current_tab = "Results"
            st.rerun()

# Helper function to display questions and collect feedback
# Helper function to display questions and collect feedback
def display_question_feedback(job_id):
    """Display questions from session state and collect feedback"""
    if job_id not in st.session_state.question_feedback:
        st.warning("No questions available for feedback.")
        return
    
    sampled_data = st.session_state.question_feedback[job_id]["questions"]
    
    st.write(f"Please review these {len(sampled_data)} sample questions (30% of total):")
    
    for i, item in enumerate(sampled_data):
        question_id = item.get("id", f"q_{i}")
        
        with st.expander(f"Question {i+1}: {item['question'][:100]}...", expanded=i==0):
            st.write("**Question:**")
            st.write(item["question"])
            
            # Display expected answer if available
            if "correct_answer" in item:
                # Instead of using a nested expander, just show the expected answer with a header
                st.write("**Expected Answer:**")
                st.write(item["correct_answer"])
            
            # Feedback options
            col1, col2 = st.columns(2)
            
            feedback_key = f"pre_feedback_{question_id}"
            current_feedback = st.session_state.question_feedback[job_id]["feedback"].get(question_id, None)
            
            # Check if feedback already given
            if current_feedback is not None:
                if current_feedback:
                    st.success("✅ You marked this question as relevant")
                else:
                    st.error("❌ You marked this question as not relevant")
            else:
                with col1:
                    if st.button("👍 Good Question", key=f"pre_good_{question_id}"):
                        st.session_state.question_feedback[job_id]["feedback"][question_id] = True
                        st.rerun()
                
                with col2:
                    if st.button("👎 Bad Question", key=f"pre_bad_{question_id}"):
                        st.session_state.question_feedback[job_id]["feedback"][question_id] = False
                        st.rerun()
    
    # Calculate and display feedback stats
    feedback_data = st.session_state.question_feedback[job_id]["feedback"]
    if feedback_data:
        st.subheader("Question Feedback Summary")
        total = len(feedback_data)
        good = sum(1 for x in feedback_data.values() if x)
        
        col1, col2 = st.columns(2)
        col1.metric("Questions Reviewed", f"{total}/{len(sampled_data)}")
        col2.metric("Good Questions", f"{good}/{total}")
        
        # If all questions have feedback, show button to proceed to evaluation
        if total == len(sampled_data):
            st.success("You've reviewed all the questions! You can now proceed to evaluation.")

# Helper function to load synthetic data
def load_synthetic_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading synthetic data: {str(e)}")
        return []

# Question Feedback page (this is the new page we're adding)
def question_feedback_page():
    st.title("💬 Question Feedback")
    
    if not st.session_state.active_job:
        st.warning("No active job selected. Please upload a document or select a job.")
        return
    
    job_id = st.session_state.active_job
    
    # Get job status
    status_data = check_job_status(job_id)
    
    if not status_data:
        st.error("Failed to fetch job status. Please try again.")
        return
    
    status = status_data["status"]
    details = status_data["details"]
    
    # Initialize session state for questions feedback if not already done
    if "question_feedback" not in st.session_state:
        st.session_state.question_feedback = {}
    
    if "question_feedback_after_eval" not in st.session_state:
        st.session_state.question_feedback_after_eval = {}
    
    # After synthetic data generation, before evaluation
    if status == "synthetic_data_generated":
        st.subheader("Pre-Evaluation Question Review")
        
        # Load synthetic data file
        synthetic_data_file = status_data["details"].get("synthetic_data_file", "")
        
        if not synthetic_data_file:
            st.error("Synthetic data file path not found in job status.")
            return
        
        try:
            # Load the synthetic data
            synthetic_data = load_synthetic_data(synthetic_data_file)
            
            if not synthetic_data:
                st.error("No synthetic data available.")
                return
            
            # Sample 30% of questions if not already sampled
            if job_id not in st.session_state.question_feedback:
                # Sample 30% of questions, but at least 1 and at most 10
                sample_size = max(1, min(10, int(len(synthetic_data) * 0.3)))
                sampled_questions = random.sample(synthetic_data, sample_size)
                
                # Store in session state
                st.session_state.question_feedback[job_id] = {
                    "questions": sampled_questions,
                    "feedback": {}
                }
            
            # Display the sampled questions for feedback
            sampled_data = st.session_state.question_feedback[job_id]["questions"]
            
            st.write(f"Please review these {len(sampled_data)} questions before evaluation:")
            
            for i, item in enumerate(sampled_data):
                question_id = item.get("id", f"q_{i}")
                
                with st.expander(f"Question {i+1}: {item['question'][:100]}...", expanded=False):
                    st.write("**Question:**")
                    st.write(item["question"])
                    
                    # Display expected answer if available
                    if "correct_answer" in item:
                        with st.expander("View Expected Answer"):
                            st.write(item["correct_answer"])
                    
                    # Feedback options
                    col1, col2 = st.columns(2)
                    
                    feedback_key = f"pre_feedback_{question_id}"
                    current_feedback = st.session_state.question_feedback[job_id]["feedback"].get(question_id, None)
                    
                    # Check if feedback already given
                    if current_feedback is not None:
                        if current_feedback:
                            st.success("✅ You marked this question as relevant")
                        else:
                            st.error("❌ You marked this question as not relevant")
                    else:
                        with col1:
                            if st.button("👍 Good Question", key=f"pre_good_{question_id}"):
                                st.session_state.question_feedback[job_id]["feedback"][question_id] = True
                                st.rerun()
                        
                        with col2:
                            if st.button("👎 Bad Question", key=f"pre_bad_{question_id}"):
                                st.session_state.question_feedback[job_id]["feedback"][question_id] = False
                                st.rerun()
            
            # Calculate and display feedback stats
            feedback_data = st.session_state.question_feedback[job_id]["feedback"]
            if feedback_data:
                st.subheader("Feedback Summary")
                total = len(feedback_data)
                good = sum(1 for x in feedback_data.values() if x)
                
                col1, col2 = st.columns(2)
                col1.metric("Questions Reviewed", f"{total}/{len(sampled_data)}")
                col2.metric("Good Questions", f"{good}/{total}")
                
                # If all questions have feedback, show button to proceed to evaluation
                if total == len(sampled_data):
                    st.success("You've reviewed all the questions! You can now proceed to evaluation.")
                    
                    if st.button("Proceed to Evaluation"):
                        st.session_state.current_tab = "Evaluate"
                        st.rerun()
            
        except Exception as e:
            st.error(f"Error loading synthetic data: {str(e)}")
            st.exception(e)
    
    # After evaluation completed
    elif status == "evaluation_completed":
        st.subheader("Post-Evaluation Answer Review")
        
        # Make sure we have the pre-evaluation questions
        if job_id not in st.session_state.question_feedback:
            st.error("Pre-evaluation question feedback not found. Please go back to the Question Feedback tab before evaluation.")
            return
        
        # Get the same questions we sampled before
        sampled_questions = st.session_state.question_feedback[job_id]["questions"]
        
        # If we haven't fetched answers yet, do it now
        if job_id not in st.session_state.question_feedback_after_eval:
            # Initialize
            st.session_state.question_feedback_after_eval[job_id] = {
                "answers": {},
                "feedback": {}
            }
            
            # Get the already generated answers from the API
            with st.spinner("Fetching answers generated during evaluation..."):
                try:
                    response = requests.get(f"{API_URL}/get-generated-answers/{job_id}")
                    
                    if response.status_code == 200:
                        generated_answers = response.json()
                        
                        # Store the answers in session state
                        for item in sampled_questions:
                            question_id = item.get("id", "")
                            if question_id in generated_answers:
                                st.session_state.question_feedback_after_eval[job_id]["answers"][question_id] = generated_answers[question_id]
                            else:
                                st.session_state.question_feedback_after_eval[job_id]["answers"][question_id] = "Answer not available from evaluation"
                    else:
                        st.error(f"Failed to fetch generated answers: {response.text}")
                        # Fallback to showing a message
                        for item in sampled_questions:
                            st.session_state.question_feedback_after_eval[job_id]["answers"][item.get("id", "")] = "Failed to retrieve answer from evaluation"
                except Exception as e:
                    st.error(f"Error fetching generated answers: {str(e)}")
                    # Fallback message
                    for item in sampled_questions:
                        st.session_state.question_feedback_after_eval[job_id]["answers"][item.get("id", "")] = f"Error retrieving answer: {str(e)}"
        
        # Display questions with generated answers
        st.write(f"Please review the {len(sampled_questions)} questions with their generated answers:")
        
        for i, item in enumerate(sampled_questions):
            question_id = item.get("id", f"q_{i}")
            
            with st.expander(f"Question {i+1}: {item['question'][:100]}...", expanded=False):
                # Question
                st.write("**Question:**")
                st.write(item["question"])
                
                # Generated answer
                st.write("**Generated Answer:**")
                answer = st.session_state.question_feedback_after_eval[job_id]["answers"].get(question_id, "Answer not available")
                st.write(answer)
                
                # Expected answer
                if "correct_answer" in item:
                    with st.expander("View Expected Answer"):
                        st.write(item["correct_answer"])
                
                # Feedback options
                col1, col2 = st.columns(2)
                
                feedback_key = f"post_feedback_{question_id}"
                current_feedback = st.session_state.question_feedback_after_eval[job_id]["feedback"].get(question_id, None)
                
                # Check if feedback already given
                if current_feedback is not None:
                    if current_feedback:
                        st.success("✅ You marked this answer as helpful")
                    else:
                        st.error("❌ You marked this answer as not helpful")
                else:
                    with col1:
                        if st.button("👍 Good Answer", key=f"post_good_{question_id}"):
                            st.session_state.question_feedback_after_eval[job_id]["feedback"][question_id] = True
                            st.rerun()
                    
                    with col2:
                        if st.button("👎 Bad Answer", key=f"post_bad_{question_id}"):
                            st.session_state.question_feedback_after_eval[job_id]["feedback"][question_id] = False
                            st.rerun()
        
        # Calculate and display feedback stats
        feedback_data = st.session_state.question_feedback_after_eval[job_id]["feedback"]
        if feedback_data:
            st.subheader("Feedback Summary")
            total = len(feedback_data)
            good = sum(1 for x in feedback_data.values() if x)
            
            col1, col2 = st.columns(2)
            col1.metric("Answers Reviewed", f"{total}/{len(sampled_questions)}")
            col2.metric("Good Answers", f"{good}/{total}")
            
            # Compare with pre-evaluation feedback
            pre_feedback = st.session_state.question_feedback[job_id]["feedback"]
            
            if pre_feedback and total == len(sampled_questions):
                st.subheader("Question vs. Answer Quality Comparison")
                
                # Create comparison dataframe
                comparison_data = []
                
                for item in sampled_questions:
                    q_id = item.get("id", "")
                    if q_id in pre_feedback and q_id in feedback_data:
                        comparison_data.append({
                            "Question ID": q_id,
                            "Question Text": item["question"][:50] + "...",
                            "Question Quality": "Good" if pre_feedback[q_id] else "Poor",
                            "Answer Quality": "Good" if feedback_data[q_id] else "Poor",
                            "Match": pre_feedback[q_id] == feedback_data[q_id]
                        })
                
                if comparison_data:
                    df = pd.DataFrame(comparison_data)
                    st.dataframe(df)
                    
                    # Calculate concordance
                    matches = sum(1 for row in comparison_data if row["Match"])
                    concordance = (matches / len(comparison_data)) * 100
                    
                    st.metric("Question-Answer Quality Concordance", f"{concordance:.1f}%")
    
    else:
        st.info(f"Current job status is '{status}'. Feedback is available after synthetic data generation or evaluation completion.")

# Evaluate page
def evaluate_page():
    st.title("📊 Evaluate RAG System")
    
    if not st.session_state.active_job:
        st.warning("No active job selected. Please upload a document or select a job.")
        return
    
    job_id = st.session_state.active_job
    
    # Get job status
    status_data = check_job_status(job_id)
    
    if not status_data:
        st.error("Failed to fetch job status. Please try again.")
        return
    
    status = status_data["status"]
    
    if status != "synthetic_data_generated" and not status.startswith("evaluation"):
        st.warning(f"Job is not ready for evaluation. Current status: {status}")
        
        # Add a refresh button
        if st.button("Refresh Status"):
            st.session_state.refresh_jobs = True
            st.rerun()
        
        return
    
    if status == "synthetic_data_generated":
        st.success("Synthetic test data has been generated. Ready to run evaluation!")
        
        # Check if user has provided question feedback
        has_feedback = job_id in st.session_state.question_feedback and st.session_state.question_feedback[job_id]["feedback"]
        
        if has_feedback:
            # Display feedback summary
            feedback_data = st.session_state.question_feedback[job_id]["feedback"]
            total = len(feedback_data)
            good = sum(1 for x in feedback_data.values() if x)
            
            st.info(f"You've reviewed {total} questions, with {good} marked as good quality.")
        else:
            st.warning("You haven't reviewed any questions yet. It's recommended to go back to the Process tab to review some sample questions before evaluation.")
        
        st.subheader("Evaluation Configuration")
        
        # Evaluation options
        col1, col2, col3 = st.columns(3)
        with col1:
            run_retrieval = st.checkbox("Retrieval Evaluation", value=True, 
                                      help="Evaluate retrieval performance (precision, recall, MRR)")
        with col2:
            run_e2e = st.checkbox("End-to-End Evaluation", value=True,
                                 help="Evaluate answer generation accuracy")
        with col3:
            run_contextual = st.checkbox("Contextual Evaluation", value=True,
                                         help="Apply contextual relevancy metrics")
        
        if st.button("Start Evaluation"):
            with st.spinner("Starting evaluation..."):
                response = requests.post(
                    f"{API_URL}/evaluate/{job_id}",
                    params={
                        "run_retrieval_eval": run_retrieval,
                        "run_e2e_eval": run_e2e,
                        "run_contextual_eval": run_contextual
                    }
                )
                
                if response.status_code == 200:
                    st.success("Evaluation started!")
                    
                    # Display evaluation progress
                    st.subheader("Evaluation Progress")
                    status_data = display_task_progress(job_id, "Evaluation completed", auto_refresh=False)
                    
                    # Refresh to show latest status
                    st.session_state.refresh_jobs = True
                    st.rerun()
                else:
                    st.error(f"Error starting evaluation: {response.text}")
    
    elif status == "evaluation_started" or status == "evaluating":
        st.info("Evaluation is in progress...")
        
        # Display evaluation progress
        st.subheader("Evaluation Progress")
        status_data = display_task_progress(job_id, "Evaluation completed", auto_refresh=False)
        
        # Add a refresh button
        if st.button("Refresh Status"):
            st.session_state.refresh_jobs = True
            st.rerun()
    
    elif status == "evaluation_completed":
        st.success("Evaluation has been completed!")
        
        if st.button("View Results"):
            st.session_state.current_tab = "Results"
            st.rerun()
    
    elif status == "evaluation_error":
        st.error(f"Evaluation failed: {status_data['details'].get('error', 'Unknown error')}")
        
        if st.button("Try Again"):
            st.session_state.refresh_jobs = True
            st.rerun()

def results_page():
    st.title("📈 Evaluation Results")
    
    if not st.session_state.active_job:
        st.warning("No active job selected. Please upload a document or select a job.")
        return
    
    job_id = st.session_state.active_job
    
    # Get job status
    status_data = check_job_status(job_id)
    
    if not status_data:
        st.error("Failed to fetch job status. Please try again.")
        return
    
    status = status_data["status"]
    
    if status != "evaluation_completed":
        st.warning(f"Evaluation not completed. Current status: {status}")
        return
    
    # Fetch evaluation results
    try:
        response = requests.get(f"{API_URL}/evaluation-results/{job_id}")
        
        if response.status_code == 200:
            results = response.json()
            
            # Display results
            display_evaluation_results(results)
            
            # Show question answers and feedback directly on this page
            st.subheader("Questions with Generated Answers")
            
            # Check if we have pre-evaluation questions
            if job_id not in st.session_state.question_feedback:
                st.warning("No pre-evaluation question feedback found.")
            else:
                # Get the same questions we sampled before
                sampled_questions = st.session_state.question_feedback[job_id]["questions"]
                
                # Initialize post-evaluation feedback if needed
                if "question_feedback_after_eval" not in st.session_state:
                    st.session_state.question_feedback_after_eval = {}
                
                # If we haven't fetched answers yet, do it now
                if job_id not in st.session_state.question_feedback_after_eval:
                    # Initialize
                    st.session_state.question_feedback_after_eval[job_id] = {
                        "answers": {},
                        "feedback": {}
                    }
                    
                    # Get the already generated answers from the API
                    with st.spinner("Fetching answers generated during evaluation..."):
                        try:
                            response = requests.get(f"{API_URL}/get-generated-answers/{job_id}")
                            
                            if response.status_code == 200:
                                generated_answers = response.json()
                                
                                # Store the answers in session state
                                for item in sampled_questions:
                                    question_id = item.get("id", "")
                                    if question_id in generated_answers:
                                        st.session_state.question_feedback_after_eval[job_id]["answers"][question_id] = generated_answers[question_id]
                                    else:
                                        st.session_state.question_feedback_after_eval[job_id]["answers"][question_id] = "Answer not available from evaluation"
                            else:
                                st.error(f"Failed to fetch generated answers: {response.text}")
                                # Fallback to showing a message
                                for item in sampled_questions:
                                    st.session_state.question_feedback_after_eval[job_id]["answers"][item.get("id", "")] = "Failed to retrieve answer from evaluation"
                        except Exception as e:
                            st.error(f"Error fetching generated answers: {str(e)}")
                            # Fallback message
                            for item in sampled_questions:
                                st.session_state.question_feedback_after_eval[job_id]["answers"][item.get("id", "")] = f"Error retrieving answer: {str(e)}"
                
                # Display questions with generated answers
                st.write(f"Please review the same {len(sampled_questions)} questions with their generated answers:")
                
                for i, item in enumerate(sampled_questions):
                    question_id = item.get("id", f"q_{i}")
                    
                    with st.expander(f"Question {i+1}: {item['question'][:100]}...", expanded=i==0):
                        # Question
                        st.write("**Question:**")
                        st.write(item["question"])
                        
                        # Generated answer
                        st.write("**Generated Answer:**")
                        answer = st.session_state.question_feedback_after_eval[job_id]["answers"].get(question_id, "Answer not available")
                        st.write(answer)
                        
                        # Expected answer - Fix: No nested expander
                        if "correct_answer" in item:
                            st.write("**Expected Answer:**")
                            st.write(item["correct_answer"])
                        
                        # Feedback options
                        col1, col2 = st.columns(2)
                        
                        feedback_key = f"post_feedback_{question_id}"
                        current_feedback = st.session_state.question_feedback_after_eval[job_id]["feedback"].get(question_id, None)
                        
                        # Check if feedback already given
                        if current_feedback is not None:
                            if current_feedback:
                                st.success("✅ You marked this answer as helpful")
                            else:
                                st.error("❌ You marked this answer as not helpful")
                        else:
                            with col1:
                                if st.button("👍 Good Answer", key=f"post_good_{question_id}"):
                                    st.session_state.question_feedback_after_eval[job_id]["feedback"][question_id] = True
                                    st.rerun()
                            
                            with col2:
                                if st.button("👎 Bad Answer", key=f"post_bad_{question_id}"):
                                    st.session_state.question_feedback_after_eval[job_id]["feedback"][question_id] = False
                                    st.rerun()
                
                # Calculate and display feedback stats
                feedback_data = st.session_state.question_feedback_after_eval[job_id]["feedback"]
                if feedback_data:
                    st.subheader("Answer Feedback Summary")
                    total = len(feedback_data)
                    good = sum(1 for x in feedback_data.values() if x)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Answers Reviewed", f"{total}/{len(sampled_questions)}")
                    col2.metric("Good Answers", f"{good}/{total}")
                    
                    # Compare with pre-evaluation feedback
                    pre_feedback = st.session_state.question_feedback[job_id]["feedback"]
                    
                    if pre_feedback and total == len(sampled_questions):
                        st.subheader("Question vs. Answer Quality Comparison")
                        
                        # Create comparison dataframe
                        comparison_data = []
                        
                        for item in sampled_questions:
                            q_id = item.get("id", "")
                            if q_id in pre_feedback and q_id in feedback_data:
                                comparison_data.append({
                                    "Question ID": q_id,
                                    "Question Text": item["question"][:50] + "...",
                                    "Question Quality": "Good" if pre_feedback[q_id] else "Poor",
                                    "Answer Quality": "Good" if feedback_data[q_id] else "Poor",
                                    "Match": pre_feedback[q_id] == feedback_data[q_id]
                                })
                        
                        if comparison_data:
                            df = pd.DataFrame(comparison_data)
                            st.dataframe(df)
                            
                            # Calculate concordance
                            matches = sum(1 for row in comparison_data if row["Match"])
                            concordance = (matches / len(comparison_data)) * 100
                            
                            st.metric("Question-Answer Quality Concordance", f"{concordance:.1f}%")
                            
                            # Create visualization of the comparison
                            good_q_good_a = sum(1 for row in comparison_data if row["Question Quality"] == "Good" and row["Answer Quality"] == "Good")
                            good_q_poor_a = sum(1 for row in comparison_data if row["Question Quality"] == "Good" and row["Answer Quality"] == "Poor")
                            poor_q_good_a = sum(1 for row in comparison_data if row["Question Quality"] == "Poor" and row["Answer Quality"] == "Good")
                            poor_q_poor_a = sum(1 for row in comparison_data if row["Question Quality"] == "Poor" and row["Answer Quality"] == "Poor")
                            
                            fig = go.Figure(data=[go.Pie(
                                labels=["Good Q + Good A", "Good Q + Poor A", "Poor Q + Good A", "Poor Q + Poor A"],
                                values=[good_q_good_a, good_q_poor_a, poor_q_good_a, poor_q_poor_a],
                                hole=.3
                            )])
                            fig.update_layout(title_text="Question-Answer Quality Distribution")
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Failed to fetch evaluation results: {response.text}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.write(traceback.format_exc())

def main():
    # Create sidebar
    create_sidebar()
    
    # Main content based on selected tab
    if st.session_state.current_tab == "Upload":
        upload_document_page()
    elif st.session_state.current_tab == "Process":
        process_document_page()
    elif st.session_state.current_tab == "Evaluate":
        evaluate_page()
    elif st.session_state.current_tab == "Results":
        results_page()

if __name__ == "__main__":
    main()