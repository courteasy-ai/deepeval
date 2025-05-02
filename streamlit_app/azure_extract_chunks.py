import os
import json
from typing import Dict, Any
from deepeval.synthesizer.chunking.doc_chunker import DocumentChunker
from azure_models_with_tenacity import initialize_azure_openai_models
from langchain_text_splitters import TokenTextSplitter

def extract_pdf_chunks(pdf_path: str, 
                       chunk_size: int = 1024, 
                       chunk_overlap: int = 0, encoding: str = "utf-8") -> Dict[str, Any]:
    """
    Extract chunks from a PDF file
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Size of each chunk in tokens
        chunk_overlap: Overlap between chunks in tokens
    
    Returns:
        Dictionary of extracted chunks
    """
    print(f"Extracting chunks from: {pdf_path}")
    
    # Initialize Azure OpenAI embedding model
    print("Initializing Azure OpenAI embedding model...")
    _, azure_embedding = initialize_azure_openai_models()
    
    # Create document chunker
    print("Creating document chunker...")
    doc_chunker = DocumentChunker(embedder=azure_embedding)
    
    # Load the document
    print("Loading document...")
    doc_chunker.load_doc(pdf_path, encoding="utf-8")
    
    # Get the document sections directly
    print("Processing document sections...")
    if doc_chunker.sections is None:
        raise ValueError("Failed to load document sections")
    
    # Use the text splitter directly
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(doc_chunker.sections)
    
    print(f"Split document into {len(chunks)} chunks")
    
    # Get the document name for unique IDs
    doc_name = os.path.basename(pdf_path).split('.')[0]
    
    # Create a dictionary mapping chunk IDs to chunk content
    chunks_dict = {}
    for i, chunk in enumerate(chunks):
        # Create a unique ID that includes the document name
        unique_id = f"{doc_name}_chunk_{i}"
        chunks_dict[unique_id] = {
            "content": chunk.page_content,
            "metadata": {
                "source": chunk.metadata.get("source", pdf_path),
                "page": chunk.metadata.get("page", None),
                "document_name": doc_name
            }
        }
    
    print(f"Extracted {len(chunks_dict)} chunks")
    
    return chunks_dict

def process_first_n_pdfs(
    directory: str, 
    n: int = 1, 
    output_file: str = "extracted_chunks/all_pdf_chunks.json"
):
    """
    Process the first N PDF files in a given directory and store in a single JSON file
    
    Args:
        directory: Path to the directory containing PDF files
        n: Number of PDFs to process
        output_file: Path to save the combined JSON file
    """
    # Get list of PDF files, sorted to ensure consistent processing
    pdf_files = sorted([f for f in os.listdir(directory) if f.endswith('.pdf')])
    
    # Limit to first n files
    pdf_files = pdf_files[:n]
    
    # Combined dictionary to store all chunks
    all_chunks = {}
    
    # Process each PDF
    for pdf_filename in pdf_files:
        pdf_path = os.path.join(directory, pdf_filename)
        
        try:
            # Extract chunks
            chunks = extract_pdf_chunks(pdf_path)
            
            # Add to combined chunks
            all_chunks.update(chunks)
            
            # Optional: Print sample chunks
            print("\nSample chunks for", pdf_filename)
            for i, (chunk_id, chunk_data) in enumerate(list(chunks.items())[:3]):
                print(f"\nChunk ID: {chunk_id}")
                print(f"Content (first 200 chars): {chunk_data['content'][:200]}...")
                print(f"Metadata: {chunk_data['metadata']}")
        
        except Exception as e:
            print(f"Error processing {pdf_filename}: {e}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save all chunks to a single JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)
    
    print(f"\nTotal chunks extracted: {len(all_chunks)}")
    print(f"Saved all chunks to: {output_file}")
    
    return all_chunks

if __name__ == "__main__":
    # Directory containing PDF files
    pdf_directory = "test_pdfs"
    
    # Output JSON file path
    output_json_file = "extracted_chunks/pdf_chunks.json"
    
    # Process first 5 PDFs and save to single JSON
    chunks = process_first_n_pdfs(
        directory=pdf_directory, 
        n=20,  # Number of PDFs to process
        output_file=output_json_file
    )