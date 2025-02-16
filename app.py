system_message = """
    You are a Helpful Legal Data Analyst specializing in legal document analysis. Your task is to do the following:
    
    - Your Primary Focus should Always be on how to protect and save the prime accused.
    - Focus on delivering answers that are directly related to the Question, ensuring that the response is as specific and actionable as possible based on the context you retrieve.
    - If the answer is not present in the context or if the query is ambiguous or vague, state that the information cannot be determined or is not available from the document.
    - Always provide any additional insights from the fetched contexts that may be relevant to the question, such as anomalies, contradictions, or key points that could aid in an investigation or analysis.
    - Note: Sometimes Contents of the same entity such as Tables can span over multiple consecutive pages. Your task is to identify the pages in order and consolidate them accordingly from the provided contexts.
    - The contexts you receive are outputs from OCR so expect them to have mistakes. Rely on your Intelligence to make corrections to the text as appropriate when formulating and presenting answers.
    - IMPORTANT: The Law is a Precise Endeavour. Never make up information you do not find in the contexts or provide your opinion on things unless explicitly asked.
    - TIP: Always provide long and detailed Answers.
    - Never use Double Quotes in your Answer. Use Backticks to highlight if necessary.
    """

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import tempfile
import json
import pickle
import fitz  # PyMuPDF
import faiss
import numpy as np
import boto3
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Any
from tavily import TavilyClient
import requests

app = FastAPI()

# Define valid users in a dictionary
def load_dict_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

# Load the MPNet model
mpnet_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
secrets_file = "../secrets.json"

SECRETS = load_dict_from_json(secrets_file)

# Replace these with your actual AWS credentials
aws_access_key_id = SECRETS["aws_access_key_id"]
aws_secret_access_key = SECRETS["aws_secret_access_key"]
INFERENCE_PROFILE_ARN = SECRETS["INFERENCE_PROFILE_ARN"]
REGION = SECRETS["REGION"]
GPT_ENDPOINT = SECRETS["GPT_ENDPOINT"]
GPT_API = SECRETS["GPT_API"]
TAVILY_API = SECRETS["TAVILY_API"]

# Paths for saving index and metadata
FAISS_INDEX_PATH = SECRETS["FAISS_INDEX_PATH"]
METADATA_STORE_PATH = SECRETS["METADATA_STORE_PATH"]

# AWS S3 setup
s3_bucket_name = SECRETS["s3_bucket_name"]
# s3_bucket_name = "gstdoststorage"

# Create a Bedrock Runtime client
bedrock_runtime = boto3.client('bedrock-runtime', region_name=REGION,
                              aws_access_key_id=aws_access_key_id,
                              aws_secret_access_key=aws_secret_access_key)

# Create a Textract Runtime client for document analysis
textract_client = boto3.client('textract', region_name=REGION,
                              aws_access_key_id=aws_access_key_id,
                              aws_secret_access_key=aws_secret_access_key)

# Create an S3 client for storage
s3_client = boto3.client('s3', region_name=REGION,
                         aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key)

# FAISS Setup
dimension = 768  # Embedding dimension for text embeddings v3
faiss_index = faiss.IndexFlatL2(dimension)
metadata_store = []

# Paths for saving index and metadata
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_STORE_PATH = "metadata_store.pkl"

def file_exists_in_blob(file_name: str) -> bool:
    """Check if a file exists in S3."""
    try:
        s3_client.head_object(Bucket=s3_bucket_name, Key=file_name)
        return True
    except Exception as e:
        if hasattr(e, 'response') and e.response['Error']['Code'] == '404':
            return False
        else:
            raise e

def upload_to_s3(local_file_path: str, file_name: str):
    """Uploads a file to S3."""
    try:
        s3_client.upload_file(local_file_path, s3_bucket_name, file_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file to S3: {str(e)}")

# Function to download file from S3
def download_from_s3(file_name: str, local_file_path: str) -> bool:
    try:
        s3_client.download_file(s3_bucket_name, file_name, local_file_path)
        return True
    except Exception as e:
        if hasattr(e, 'response') and e.response['Error']['Code'] == '404':
            return False
        else:
            raise HTTPException(status_code=500, detail=f"Failed to download {file_name}: {str(e)}")

# Function to generate embeddings
def generate_titan_embeddings(text: str) -> np.ndarray:
    try:
        embedding = mpnet_model.encode(text, normalize_embeddings=True)
        return np.array(embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {e}")

def extract_text_from_pdf(file_path: str) -> List[Tuple[int, str]]:
    """Extracts text from a PDF using AWS Textract."""
    doc = fitz.open(file_path)
    pages_text = []
    
    for page_num in range(len(doc)):
        try:
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            temp_image_path = os.path.join(tempfile.gettempdir(), f"page_{page_num}.png")
            pix.save(temp_image_path)
            
            with open(temp_image_path, "rb") as image_file:
                image_bytes = image_file.read()
            
            response = textract_client.detect_document_text(Document={'Bytes': image_bytes})
            text_lines = [block['Text'] for block in response.get('Blocks', []) if 'Text' in block]
            pages_text.append((page_num + 1, "\n".join(text_lines)))
            
            os.remove(temp_image_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing page {page_num + 1}: {str(e)}")
    
    return pages_text

def add_pdf_to_index(file_name: str):
    """Processes and indexes the given PDF file from S3."""
    temp_pdf_path = os.path.join(tempfile.gettempdir(), file_name)
    
    # if not download_from_s3(file_name, temp_pdf_path):
    #     raise HTTPException(status_code=400, detail="File not found in S3.")
    faiss_index, metadata_store = load_index_and_metadata()
    file_name_base = os.path.basename(file_name)
    upload_to_s3(temp_pdf_path, file_name_base)
    
    pages_text = extract_text_from_pdf(temp_pdf_path)
    
    for page_num, text in pages_text:
        embedding = generate_titan_embeddings(text)
        faiss_index.add(embedding.reshape(1, -1))
        metadata_store.append({"filename": file_name, "page": page_num, "text": text})
    
    save_index_and_metadata()
    try:
        os.remove(temp_pdf_path)
    except:
        print("Temp file Not Deleted")

def save_index_and_metadata():
    # Save files locally
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(METADATA_STORE_PATH, "wb") as f:
        pickle.dump(metadata_store, f)

    # Upload files to Blob Storage
    try:
        upload_to_s3(FAISS_INDEX_PATH, os.path.basename(FAISS_INDEX_PATH))
        upload_to_s3(METADATA_STORE_PATH, os.path.basename(METADATA_STORE_PATH))
    except Exception as e:
        print(f"Error uploading index or metadata to Blob Storage: {str(e)}")

def load_index_and_metadata():
    global faiss_index, metadata_store

    index_blob_name = os.path.basename(FAISS_INDEX_PATH)
    metadata_blob_name = os.path.basename(METADATA_STORE_PATH)

    # Download files from Blob Storage if available
    index_downloaded = download_from_s3(index_blob_name, FAISS_INDEX_PATH)
    metadata_downloaded = download_from_s3(metadata_blob_name, METADATA_STORE_PATH)

    if index_downloaded and metadata_downloaded:
        # Load FAISS index and metadata store
        try:
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_STORE_PATH, "rb") as f:
                metadata_store = pickle.load(f)
            print("Index and metadata loaded from Storage.")
        except Exception as e:
            print(f"Failed to load index or metadata: {str(e)}")
            # Initialize empty index and metadata if loading fails
            faiss_index = faiss.IndexFlatL2(dimension)
            metadata_store = []
    else:
        print("Index or metadata not found in Blob Storage. Initializing new.")
        # Initialize empty index and metadata
        faiss_index = faiss.IndexFlatL2(dimension)
        metadata_store = []
    
    return faiss_index, metadata_store

def call_gpt_api(system_message, user_query):
    url = GPT_ENDPOINT
    headers = {  
        "Content-Type": "application/json",  
        "api-key": GPT_API
    }  
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query}
    ]
    payload = {  
        "messages": messages,  
        "temperature": 0.7,  
        "max_tokens": 16384   
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()  
    return response.json()["choices"][0]["message"]["content"]

def call_llm_api(system_message, user_query):
    # Combine system and user messages
    messages = system_message + user_query

    # Prepare the request payload
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 16384,
        "messages": [
            {
                "role": "user",
                "content": messages
            }
        ]
    }

    try:
        # Invoke the model (Claude)
        response = bedrock_runtime.invoke_model(
            modelId=INFERENCE_PROFILE_ARN,  # Use the ARN for your inference profile
            contentType='application/json',
            accept='application/json',
            body=json.dumps(payload)
        )

        # Parse and return the response
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']

    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.post("/upload_pdf")
def upload_pdf(file_name: str):
    """API endpoint to process and index a single PDF file from S3."""
    try:
        if file_exists_in_blob(file_name):
            return {"filename": file_name, "message": "File already exists. Skipping upload."}
        
        add_pdf_to_index(file_name)
        return {"filename": file_name, "message": "PDF processed and indexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/query_documents_with_page_range")
def query_documents_with_page_range(selected_files: str, selected_page_ranges: str, prompt: str, top_k: str, web_search: str, llm_model: str):
    try:
        selected_files = json.loads(selected_files)
        
        selected_page_ranges = json.loads(selected_page_ranges)
        print(selected_files)
        print(selected_page_ranges)

        qp_prompt = {
            "system_message": "You are an intelligent query refiner. Your job is to take a user's original query (which may contain poor grammar or informal language) and generate two well-formed prompts: one for a semantic search over a FAISS index and another for a web search. The semantic search prompt should improve the user's provided query, incorporating the last 5 messages for better contextual understanding. The web search prompt should refine the query further to fetch relevant legal resources online. Output only a JSON object with 'semantic_search_prompt' and 'web_search_prompt' as keys.",
            "user_query": f"User Query: {prompt}\n\\n\nGenerate the JSON output with the two improved prompts."
        }

        op_format = '''
        # Output Format:
        
        ```json
        {
            "semantic_search_prompt": "Refined user query using proper grammar and context from last 5 messages, optimized for FAISS index retrieval.",
            "web_search_prompt": "Further refined query designed to fetch relevant legal resources from the web."
        }
        ```
        '''

        prompts = call_llm_api(qp_prompt["system_message"], qp_prompt["user_query"]+op_format)
        try:
            # return json.loads(answer[7:-3])
            prompt_op = json.loads(prompts.split("```json")[1].split("```")[0])
        except:
            # return json.loads(answer[3:-3])
            prompt_op = json.loads(prompts.split("```")[1].split("```")[0])
        print(prompt_op)
        query = prompt_op["semantic_search_prompt"]

        faiss_index, metadata_store = load_index_and_metadata()

        query_embedding = generate_titan_embeddings(query).reshape(1, -1)
        if faiss_index.ntotal == 0:
            return [], "No data available to query."

        # Fetch all metadata for the given query
        k = faiss_index.ntotal  # Initial broad search
        distances, indices = faiss_index.search(query_embedding, k)
        
        filtered_results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(metadata_store):
                metadata = metadata_store[idx]
                if metadata['filename'] in selected_files:
                    min_page, max_page = selected_page_ranges.get(metadata['filename'], (None, None))
                    if min_page and max_page and min_page <= metadata['page'] <= max_page:
                        filtered_results.append((dist, idx))
        
        # Limit to topK after filtering
        top_k_results = sorted(filtered_results, key=lambda x: x[0])[:int(top_k)]
        top_k_metadata = [metadata_store[idx] for _, idx in top_k_results]

        user_query = f"""
        You are required to provide a structured response to the following question, based on the context retrieved from the provided documents.

        # User Query:
        <<<{prompt}>>>

        # The top K most relevant contexts fetched from the documents are as follows:
        {json.dumps(top_k_metadata, indent=4)}

        # Always Approach the Task Step by Step.
        * Read and Understand the Provided Contexts.
        * Identify Relevant sections from those and Arrange them as necessary
        * Then Formulate your Answer Adhering to the Guidelines.
        """
        
        ws_response = ""

        if web_search=="True":
            ws_query = prompt_op["web_search_prompt"]
            # Call the LLM API to get the answer
            # To install, run: pip install tavily-python


            client = TavilyClient(api_key="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")

            ws_response = client.search(
                query=ws_query,
                search_depth="advanced",
                include_raw_content=True
            )

            print(ws_response)

            wsp = f"""
            # Feel free to use the Web Search Results for Additional Context as well:

            {json.dumps(ws_response)}
            """
            if llm_model=="Claude 3.5 Sonnet":
                answer = call_llm_api(system_message, user_query+wsp)
            elif llm_model=="GPT 4o":
                answer = call_gpt_api(system_message, user_query+wsp)
        else:
            if llm_model=="Claude 3.5 Sonnet":
                answer = call_llm_api(system_message, user_query)
            elif llm_model=="GPT 4o":
                answer = call_gpt_api(system_message, user_query)

        return {"answer": answer, "top_k_metadata": top_k_metadata, "web_search_results": ws_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))