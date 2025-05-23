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
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from urllib.parse import urljoin

app = FastAPI()
load_dotenv()

# Retrieve API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
wandb_key = os.getenv("WANDB_KEY")
env = os.getenv("ENV", "prod")

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

# Define local file paths
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")

# Load FAISS embeddings from local storage
embeddings = OpenAIEmbeddings()
choice_retriever = FAISS.load_local("data/choice_data_enums_index", embeddings, allow_dangerous_deserialization=True)
model_retriever = FAISS.load_local("data/pydantic_model_index", embeddings, allow_dangerous_deserialization=True)
system_retriever = FAISS.load_local("data/system_prompt_index", embeddings, allow_dangerous_deserialization=True)
collaboratory_retriever = FAISS.load_local("data/collaboratory_activity_form_index", embeddings, allow_dangerous_deserialization=True)
user_retriever = FAISS.load_local("data/user_prompt_index", embeddings, allow_dangerous_deserialization=True)

# Initialize OpenAI Model
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)

def get_profile_info(link):
    response = requests.get(link, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    name_tag = soup.find("h1")
    name = name_tag.get_text(strip=True) if name_tag else "Name not found"

    email_tag = soup.find("a", href=re.compile(r"mailto:"))
    email = email_tag.get_text(strip=True) if email_tag else "Email not found"

    phone_tag = soup.find(string=re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"))
    phone = phone_tag.strip() if phone_tag else "Phone not found"
 
    text=name + "\n" + email + "\n" + phone
    return text

all_tags=[]

def scrape_url_content(url):
    global actualUrl
    try:
        actualUrl=url
        response = requests.get(url, timeout=10)
        response.raise_for_status()          

        # Check if the URL points to a PDF file
        if "application/pdf" in response.headers.get("Content-Type", ""):
            # Save the PDF locally
            pdf_path = "temp.pdf"
            with open(pdf_path, "wb") as pdf_file:
                pdf_file.write(response.content)

            # Extract text from the PDF
            pdf_text = extract_text_from_pdf(pdf_path)

            # Clean up the temporary file
            os.remove(pdf_path)

            return pdf_text if pdf_text else "No content extracted from PDF."
        else:
            # Handle non-PDF content (e.g., HTML)
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            article_text = "\n".join([para.get_text() for para in paragraphs])
        time_tag = soup.find('time')
        if time_tag:
            date_text = time_tag.get_text(strip=True)
            article_text += f"Activity Published on Date: {date_text}\n"
        else:
            print("Date not found")
        
        if ".asu.edu" in url:
            links = []
            for para in paragraphs:
                for a in para.find_all("a", href=True):
                    href = urljoin(url, a['href'])  # Convert to absolute URL
                    if "search.asu.edu" in href:
                        links.append(href)           
            if links:
                for link in links:
                    article_text += get_profile_info(link)
            
        tag_elements = soup.select("div.node-body-categories .view-content a.btn-tag") 
        tags = [tag.get_text(strip=True) for tag in tag_elements]
        all_tags.extend(tags)  # Store all tags for later use
        article_text += "\nTags: " + ", ".join(tags)
        article_text += "\nActivity Website: " + url
        return article_text if article_text else "No content extracted from URL."

    except Exception as e:
        return f"Error extracting content: {str(e)}"

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

# Function to extract JSON from AI response
def extract_json_from_string(response_text):
    try:
        match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if match:
            json_text = match.group(1)
            complete_json=json.loads(json_text)
            return complete_json
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
    model_text = retrieve_text(input_text, model_retriever)
    system_text = retrieve_text(input_text, system_retriever)
    collaboratory_text = retrieve_text(input_text, collaboratory_retriever)
    user_text = retrieve_text(input_text, user_retriever)
    
    # Combine retrieved content
    full_context = f"{input_text} \n {choice_text} \n {model_text} \n {user_text} \n {collaboratory_text}"
    structured_response = llm.invoke([
        {"role": "system", "content": system_text},
        {"role": "user", "content": full_context}
    ])
    
    # Print AI Message in logs
    print("AI Message:", structured_response.content)
    
    # Extract JSON from response
    extracted_json = extract_json_from_string(structured_response.content)

    wandb.log({"ai_message": structured_response.content, "structured_response": extracted_json})
    
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
    if env == "prod":
        uvicorn.run(app, host="0.0.0.0", port=8888, ssl_keyfile="proxyai.pem", ssl_certfile="proxyai.crt")
    else:
        uvicorn.run(app, host="localhost", port=8888)
