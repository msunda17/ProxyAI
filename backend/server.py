from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from typing import Optional
import os
import wandb
import weave
import uvicorn

app = FastAPI()

# Retrieve API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
wandb_key = os.getenv("WANDB_KEY")

# Ensure API keys are provided
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")
if not wandb_key:
    raise ValueError("Missing WANDB_KEY environment variable.")

# Initialize Weights & Biases
wandb.login(key=wandb_key)
wandb.init(project="proxyai")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to specific origins for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to fetch files from a GitHub repository
GITHUB_REPO_URL = "https://github.com/msunda17/blob/main/ProxyAI/backend/"

def fetch_github_file(file_name):
    url = GITHUB_REPO_URL + file_name
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        if file_name.endswith(".json"):
            return response.json()
        return response.text
    except Exception as e:
        raise ValueError(f"Error fetching {file_name}: {str(e)}")

# Function to fetch embeddings from GitHub if not found locally
def fetch_github_embedding(index_name):
    local_path = f"{index_name}/"
    if not os.path.exists(local_path):
        try:
            zip_url = GITHUB_REPO_RAW_URL + f"{index_name}.zip"
            zip_path = f"/tmp/{index_name}.zip"
            response = requests.get(zip_url, timeout=10, stream=True)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            shutil.unpack_archive(zip_path, local_path)
            print(f"✅ Downloaded and extracted {index_name} embeddings from GitHub.")
        except Exception as e:
            raise ValueError(f"Error fetching {index_name} embeddings: {str(e)}")

# Load structured data from GitHub
choice_data = fetch_github_file("choice_data.json")
json_format = fetch_github_file("json_format.json")
system_prompt = fetch_github_file("system_prompt.txt")
collaboratory_form = fetch_github_file("collaboratory_activity_form.json")

# Load embeddings, fetching from GitHub if necessary
embeddings = OpenAIEmbeddings()
for index in ["choice_data_index", "json_format_index", "system_prompt_index", "collaboratory_activity_form_index"]:
    fetch_github_embedding(index)

# Initialize OpenAI Model
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)

# Function to scrape text from a URL
def scrape_url_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        article_text = "\n".join([para.get_text() for para in paragraphs])
        return article_text if article_text else "No content extracted from URL."
    except Exception as e:
        return f"Error extracting content: {str(e)}"

# Function to extract JSON from AI response
def extract_json_from_string(response_text):
    try:
        match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if match:
            json_text = match.group(1)
            return json.loads(json_text)
    except json.JSONDecodeError as e:
        return {"error": "Failed to parse JSON response", "details": str(e)}
    return {"error": "No valid JSON found in response"}

# Define a proper chain using FAISS retrievers
def retrieve_text(query, retriever):
    docs = retriever.similarity_search(query, k=3)
    return " ".join([doc.page_content for doc in docs])

class InputData(BaseModel):
    url: Optional[str] = None
    file: Optional[UploadFile] = None

@app.post("/generate_activity")
@weave.op()
def generate_activity(input_data: InputData):
    """Process user input and generate structured Collaboratory activity data."""
    wandb.log({"request_received": input_data.dict()})
    
    input_text = extract_text(input_data.url, input_data.file)
    
    # Retrieve relevant texts
    choice_text = retrieve_text(input_text, choice_retriever)
    json_text = retrieve_text(input_text, json_retriever)
    system_text = retrieve_text(input_text, system_retriever)
    collaboratory_text = retrieve_text(input_text, collaboratory_retriever)
    
    # Combine retrieved content
    full_context = f"{input_text} \n {choice_text} \n {json_text} \n {system_text} \n {collaboratory_text}"
    structured_response = llm.invoke(full_context)
    
    # Print AI Message in logs
    print("AI Message:", structured_response.content)
    
    # Extract JSON from response
    extracted_json = extract_json_from_string(structured_response.content)
    
    wandb.log({"ai_message": structured_response.content, "structured_response": extracted_json})
    weave.log({"ai_message": structured_response.content, "structured_response": extracted_json})
    
    return {"ai_message": structured_response.content, "structured_response": extracted_json}

def extract_text(url, file):
    if url:
        return scrape_url_content(url)
    elif file:
        return process_file(file)
    return ""

def process_file(file):
    df = pd.read_csv(file.file) if file.filename.endswith(".csv") else pd.read_excel(file.file)
    return df.to_json()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)