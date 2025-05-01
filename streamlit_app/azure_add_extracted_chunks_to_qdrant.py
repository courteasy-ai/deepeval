import os
import json
from deepeval.synthesizer.chunking.doc_chunker import DocumentChunker
from azure_models_with_tenacity import initialize_azure_openai_models
from langchain_text_splitters import TokenTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv()
# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_pdfs_to_qdrant(
    directory: str, 
    qdrant_uri: str, 
    qdrant_api_key: str, 
    collection_name: str = "test_evaluation_chunking",
    chunk_size: int = 1024, 
    chunk_overlap: int = 128, 
    output_dir: str = "extracted_chunks",
    encoding: str = "utf-8",
    n: int = 20
):
    """
    Extract chunks from multiple PDF files and store them in Qdrant
    
    Args:
        directory: Directory containing PDF files
        qdrant_uri: URI for Qdrant database
        qdrant_api_key: API key for Qdrant
        collection_name: Name of the Qdrant collection
        chunk_size: Size of each chunk in tokens
        chunk_overlap: Overlap between chunks in tokens
        output_dir: Directory to save the extracted chunks locally
        encoding: Text encoding to use when loading documents
        n: Number of PDFs to process
    
    Returns:
        Dictionary of all extracted chunks
    """
    # Initialize Azure OpenAI embedding model
    print("Initializing Azure OpenAI embedding model...")
    _, azure_embedding = initialize_azure_openai_models()
    
    # Initialize Qdrant client
    print(f"Connecting to Qdrant at {qdrant_uri}")
    client = QdrantClient(
        url=qdrant_uri,
        api_key=qdrant_api_key,
        port=443, timeout=10.0
    )
    
    # Create collection if it doesn't exist
    try:
        # Attempt to delete existing collection to start fresh
        client.delete_collection(collection_name)
    except:
        pass
    
    # Create new collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1536,  # Size for text-embedding-ada-002
            distance=models.Distance.COSINE,
        )
    )
    
    # Get list of PDF files
    pdf_files = [
        f for f in os.listdir(directory) 
        if f.endswith('.pdf')
    ][:n]  # Limit to first n PDFs
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Combined dictionary to store all chunks
    all_chunks = {}
    
    # Process each PDF
    for pdf_filename in pdf_files:
        pdf_path = os.path.join(directory, pdf_filename)
        print(f"\nProcessing: {pdf_filename}")
        
        try:
            # Create document chunker
            doc_chunker = DocumentChunker(embedder=azure_embedding)
            
            # Load the document - add encoding parameter
            doc_chunker.load_doc(pdf_path, encoding=encoding)
            
            # Check document sections
            if doc_chunker.sections is None:
                print(f"Failed to load sections for {pdf_filename}")
                continue
            
            # Use the text splitter directly
            text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = text_splitter.split_documents(doc_chunker.sections)
            
            # Get the document name for unique IDs
            doc_name = os.path.basename(pdf_path).split('.')[0]
            
            # Generate embeddings for all chunks
            contents = [chunk.page_content for chunk in chunks]
            embeddings = azure_embedding.embed_texts(contents)
            
            # Prepare points for Qdrant
            points = []
            doc_chunks = {}
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create a unique ID that includes the document name
                unique_id = f"{doc_name}_chunk_{i}"
                
                # Create point for Qdrant
                point = models.PointStruct(
                    id=len(all_chunks) + i,  # Unique numeric ID across all documents
                    vector=embedding,
                    payload={
                        "chunk_id": unique_id,
                        "content": chunk.page_content,
                        "source": chunk.metadata.get("source", pdf_path),
                        "page": chunk.metadata.get("page", None),
                        "document_name": doc_name
                    }
                )
                points.append(point)
                
                # Store in document-specific and combined chunks
                doc_chunks[unique_id] = {
                    "content": chunk.page_content,
                    "metadata": {
                        "source": chunk.metadata.get("source", pdf_path),
                        "page": chunk.metadata.get("page", None),
                        "document_name": doc_name
                    }
                }
                all_chunks[unique_id] = doc_chunks[unique_id]
            
            # Upload to Qdrant in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                print(f"Uploading batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} to Qdrant")
                client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
            
            # Save document-specific chunks to JSON
            doc_output_file = os.path.join(output_dir, f"{doc_name}_chunks.json")
            with open(doc_output_file, "w", encoding="utf-8") as f:
                json.dump(doc_chunks, f, indent=2)
            
            print(f"Extracted {len(doc_chunks)} chunks from {pdf_filename}")
        
        except Exception as e:
            print(f"Error processing {pdf_filename}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save all chunks to a combined JSON file
    combined_output_file = os.path.join(output_dir, "all_chunks.json")
    with open(combined_output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)
    
    print(f"\nTotal chunks extracted: {len(all_chunks)}")
    print(f"Saved all chunks to: {combined_output_file}")
    
    return all_chunks

if __name__ == "__main__":
    # Directory containing PDF files
    PDF_DIRECTORY = " " # Specify the directory containing your PDF files
    
    # Qdrant configuration
    QDRANT_URI = os.getenv("QDRANT_URI")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
    
    # Extract chunks and store in Qdrant
    chunks = extract_pdfs_to_qdrant(
        directory=PDF_DIRECTORY,
        qdrant_uri=QDRANT_URI,
        qdrant_api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION_NAME,
        n=20  # Process first 20 PDFs
    )
    
    # Print sample chunks
    print("\nSample Chunks:")
    for i, (chunk_id, chunk_data) in enumerate(list(chunks.items())[:3]):
        print(f"\nChunk ID: {chunk_id}")
        print(f"Content (first 200 chars): {chunk_data['content'][:200]}...")
        print(f"Metadata: {chunk_data['metadata']}")