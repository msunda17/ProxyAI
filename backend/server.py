from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import json
import pandas as pd
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
import tiktoken
import asyncio
import aiohttp

app = FastAPI()
load_dotenv()

# Retrieve API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
wandb_key = os.getenv("WANDB_KEY")
env = os.getenv("ENV", "prod")

def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

async def log_token_usage_async(input_text, choice_text, model_text, user_text, collaboratory_text, system_text, final_response, activity_type):
    try:
        # Calculate token usage for monitoring
        input_tokens = count_tokens(input_text)
        choice_tokens = count_tokens(choice_text)
        model_tokens = count_tokens(model_text)
        user_tokens = count_tokens(user_text)
        collab_tokens = count_tokens(collaboratory_text)
        system_tokens = count_tokens(system_text)
        
        # Use final_response for token counting
        if isinstance(final_response, dict) and "error" not in final_response:
            response_tokens = count_tokens(str(final_response))
        else:
            response_tokens = 0
        
        total_llm_tokens = system_tokens + response_tokens
        rag_tokens = input_tokens + choice_tokens + model_tokens + user_tokens + collab_tokens
        total_pipeline_tokens = total_llm_tokens + rag_tokens

        print(f"\nToken Usage Analysis (Async):")
        print(f"Input Tokens: {input_tokens}")
        print(f"Choice Tokens: {choice_tokens}")
        print(f"Model Tokens: {model_tokens}")
        print(f"User Tokens: {user_tokens}")
        print(f"Collaboratory Tokens: {collab_tokens}")
        print(f"System Prompt Tokens: {system_tokens}")
        print(f"Response Token Count: {response_tokens}")
        print(f"Total LLM Tokens (Prompt + Response): {total_llm_tokens}")
        print(f"Total Pipeline Tokens (Input + Retrieval + LLM): {total_pipeline_tokens}")

        # Log to Weights & Biases with token metrics
        wandb.log({
            "token_analysis": {
                "input_tokens": input_tokens,
                "choice_tokens": choice_tokens,
                "model_tokens": model_tokens,
                "user_tokens": user_tokens,
                "collaboratory_tokens": collab_tokens,
                "system_tokens": system_tokens,
                "response_tokens": response_tokens,
                "total_llm_tokens": total_llm_tokens,
                "rag_tokens": rag_tokens,
                "total_pipeline_tokens": total_pipeline_tokens
            },
            "activity_type": activity_type
        })
        
    except Exception as e:
        print(f"Error in token counting: {e}")

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


# Load FAISS embeddings from local storage
embeddings = OpenAIEmbeddings()
choice_retriever = FAISS.load_local("data/choice_data_enums_index", embeddings, allow_dangerous_deserialization=True)
model_retriever = FAISS.load_local("data/pydantic_model_index", embeddings, allow_dangerous_deserialization=True)
system_retriever = FAISS.load_local("data/system_prompt_index", embeddings, allow_dangerous_deserialization=True)
collaboratory_retriever = FAISS.load_local("data/collaboratory_activity_form_index", embeddings, allow_dangerous_deserialization=True)
user_retriever = FAISS.load_local("data/user_prompt_index", embeddings, allow_dangerous_deserialization=True)

# Initialize OpenAI Models for parallel processing
llm_classification = ChatOpenAI(model="gpt-4.1", temperature=0, openai_api_key=openai_api_key)
llm_extraction = ChatOpenAI(model="gpt-4.1", temperature=0, openai_api_key=openai_api_key)

async def get_profile_info_async(link):
    """Async version of profile info extraction"""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(link) as response:
                response.raise_for_status()
                content = await response.read()
                soup = BeautifulSoup(content, "html.parser")
                name_tag = soup.find("h1")
                name = name_tag.get_text(strip=True) if name_tag else "Name not found"

                email_tag = soup.find("a", href=re.compile(r"mailto:"))
                email = email_tag.get_text(strip=True) if email_tag else "Email not found"

                phone_tag = soup.find(string=re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"))
                phone = phone_tag.strip() if phone_tag else "Phone not found"
         
                text = name + "\n" + email + "\n" + phone
                return text
    except Exception as e:
        return f"Error extracting profile info: {str(e)}"

async def scrape_url_content_async(url):
    """Async version of URL scraping for parallel processing"""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                content = await response.read()
                
                # Check if the URL points to a PDF file
                if "application/pdf" in response.headers.get("Content-Type", ""):
                    # Save the PDF locally
                    pdf_path = f"temp_{hash(url)}.pdf"
                    with open(pdf_path, "wb") as pdf_file:
                        pdf_file.write(content)

                    # Extract text from the PDF
                    pdf_text = extract_text_from_pdf(pdf_path)

                    # Clean up the temporary file
                    os.remove(pdf_path)

                    return pdf_text if pdf_text else "No content extracted from PDF.", []
                else:
                    # Handle non-PDF content (e.g., HTML)
                    soup = BeautifulSoup(content, "html.parser")
                    paragraphs = soup.find_all("p")
                    article_text = "\n".join([para.get_text() for para in paragraphs])
                    
                    time_tag = soup.find('time')
                    if time_tag:
                        date_text = time_tag.get_text(strip=True)
                        article_text += f"Activity Published on Date: {date_text}\n"
                    else:
                        print("Date not found")
                    
                    tags = []
                    if ".asu.edu" in url:
                        links = []
                        for para in paragraphs:
                            for a in para.find_all("a", href=True):
                                href = urljoin(url, a['href'])  # Convert to absolute URL
                                if "search.asu.edu" in href:
                                    links.append(href)           
                        if links:
                            for link in links:
                                article_text += await get_profile_info_async(link)
                        
                    tag_elements = soup.select("div.node-body-categories .view-content a.btn-tag") 
                    tags = [tag.get_text(strip=True) for tag in tag_elements]
                    article_text += "\nTags: " + ", ".join(tags)
                    article_text += "\nActivity Website: " + url
                    return article_text if article_text else "No content extracted from URL.", tags

    except Exception as e:
        return f"Error extracting content: {str(e)}", []

def extract_sdg_number(text):
    match = re.search(r"SDG\s*0*(\d+)", text, re.IGNORECASE)
    return f"SDG {int(match.group(1))}" if match else None

def make_complete_json(json_text, all_tags, actual_url):
    """Process JSON with tags and URL context passed as parameters"""
    try:
        # Check if actualUrl contains .asu.edu
        if actual_url and ".asu.edu" in actual_url:
            seen_sdg_numbers = set()
            unique_programs = []
            for item in json_text.get("programsOrInitiatives", []):
                sdg_key = extract_sdg_number(item)
                if sdg_key:
                    if sdg_key.lower() not in seen_sdg_numbers:
                        seen_sdg_numbers.add(sdg_key.lower())
                        unique_programs.append(item)
                    else:
                        continue 
                else:
                    unique_programs.append(item)

            for tag in all_tags:
                if tag.lower().startswith("sdg"):
                    sdg_key = extract_sdg_number(tag)
                    if sdg_key and sdg_key.lower() not in seen_sdg_numbers:
                        unique_programs.append(tag) 
                        seen_sdg_numbers.add(sdg_key.lower())

            json_text["programsOrInitiatives"] = unique_programs

            for tag in all_tags:
                if tag.lower() in ["public service", "community engagement"]:
                    json_text["activityType"] = tag
                    break
        # No specific processing for non-ASU URLs

    except Exception as e:
        return {"error": "Failed to update JSON response", "details": str(e)}

    return json_text

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def load_classification_prompt():
    """Load the classification prompt for activity type determination"""
    prompt_path = "data/activity_type_classification_prompt.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_content = f.read().strip()
    return prompt_content

def extract_json_from_string(response_text, all_tags=None, actual_url=None):
    try:
        match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if match:
            json_text = match.group(1)
            complete_json = make_complete_json(json.loads(json_text), all_tags or [], actual_url)
            return complete_json
        
    except json.JSONDecodeError as e:
        return {"error": "Failed to parse JSON response", "details": str(e)}
    return {"error": "No valid JSON found in response"}

# Define a proper chain using FAISS retrievers
def retrieve_text(query, retriever):
    docs = retriever.similarity_search(query, k=3)
    return " ".join([doc.page_content for doc in docs])

async def classify_activity_type(input_text):
    """Fast classification call - determines activity type only"""
    try:        
        classification_prompt = load_classification_prompt()
        classification_context = f"""
        ACTIVITY DESCRIPTION:
        {input_text}
        
        CLASSIFICATION TASK:
        {classification_prompt}
        """        
        response = await llm_classification.ainvoke([
            {"role": "system", "content": classification_prompt},
            {"role": "user", "content": classification_context}
        ])
        
        # Extract activity type from response
        content = response.content
        
        # Robust extraction logic - look for exact matches first
        if "Community Engagement" in content:
            result = "Community Engagement"
        elif "Public Service" in content:
            result = "Public Service"
        return result
            
    except Exception as e:
        print(f"❌ Classification error: {e}")
        if any(word in input_text.lower() for word in ["partnership", "collaboration", "co-", "shared", "joint", "together"]):
            return "Community Engagement"
        else:
            return "Public Service"

async def extract_full_data(input_text, choice_text, model_text, system_text, user_text, collaboratory_text):
    """Full data extraction call - processes all fields with RAG"""
    try:
        # Combine retrieved content for full extraction
        full_context = f"{input_text} \n {choice_text} \n {model_text} \n {user_text} \n {collaboratory_text}"
        
        response = await llm_extraction.ainvoke([
            {"role": "system", "content": system_text},
            {"role": "user", "content": full_context}
        ])
        
        return response.content
        
    except Exception as e:
        print(f"Extraction error: {e}")
        return None

def merge_responses(activity_type, extraction_response, all_tags=None, actual_url=None):
    """Merge classification and extraction responses"""
    try:
        # Extract JSON from extraction response
        extracted_json = extract_json_from_string(extraction_response, all_tags, actual_url)
        
        if "error" in extracted_json:
            return extracted_json
        
        # Ensure activityType is set correctly
        extracted_json["activityType"] = activity_type
        
        return extracted_json
        
    except Exception as e:
        print(f"Merge error: {e}")
        return {"error": "Failed to merge responses", "details": str(e)}

class InputData(BaseModel):
    url: Optional[str] = None
    urls: Optional[list[str]] = None
    file: Optional[UploadFile] = None

@app.post("/generate_activity")
@weave.op()
async def generate_activity(input_data: InputData):
    import time
    start_time = time.time()
    
    wandb.log({"request_received": input_data.dict()})
    
    # Handle multiple URLs with parallel processing
    all_tags = []
    actual_url = None
    
    if input_data.urls:
        # Process multiple URLs in parallel
        valid_urls = [url for url in input_data.urls if url.strip()]
        if not valid_urls:
            return {"error": "No valid URLs provided"}
        
        # Scrape all URLs concurrently
        scraping_tasks = [scrape_url_content_async(url) for url in valid_urls]
        scraping_results = await asyncio.gather(*scraping_tasks, return_exceptions=True)
        
        # Process results and collect tags
        all_texts = []
        for i, result in enumerate(scraping_results):
            if isinstance(result, Exception):
                print(f"Error scraping URL {valid_urls[i]}: {result}")
                all_texts.append(f"Error extracting content from {valid_urls[i]}: {str(result)}")
            else:
                text, tags = result
                all_texts.append(text)
                all_tags.extend(tags)
        
        # Use first URL as actual_url for ASU processing
        actual_url = valid_urls[0] if valid_urls else None
        
        # Combine all URL content with separators
        input_text = "\n\n---\n\n".join(all_texts)
    else:
        # Handle single URL (backward compatibility)
        input_text = await extract_text(input_data.url, input_data.file)
        actual_url = input_data.url
    
    # Retrieve relevant texts
    choice_text = retrieve_text(input_text, choice_retriever)
    model_text = retrieve_text(input_text, model_retriever)
    system_text = retrieve_text(input_text, system_retriever)
    collaboratory_text = retrieve_text(input_text, collaboratory_retriever)
    user_text = retrieve_text(input_text, user_retriever)

    try:
        # Execute both tasks concurrently using asyncio
        classification_task = classify_activity_type(input_text)
        extraction_task = extract_full_data(input_text, choice_text, model_text, system_text, user_text, collaboratory_text)
        
        # Wait for both tasks to complete concurrently
        activity_type, extraction_response = await asyncio.gather(classification_task, extraction_task)
        # Combine classification and extraction responses
        final_response = merge_responses(activity_type, extraction_response, all_tags, actual_url)
    except Exception as e:
        print(f"Parallel processing failed, falling back to sequential: {e}")
        
        # Fallback to sequential processing
        activity_type = await classify_activity_type(input_text)
        extraction_response = await extract_full_data(input_text, choice_text, model_text, system_text, user_text, collaboratory_text)
        final_response = merge_responses(activity_type, extraction_response, all_tags, actual_url)

    # Log basic metrics to Weights & Biases immediately
    wandb.log({
        "ai_message": final_response,
        "structured_response": final_response,
        "parallel_processing": True,
        "activity_type": activity_type
    })
    
    # Start token counting asynchronously (non-blocking)
    asyncio.create_task(log_token_usage_async(
        input_text, choice_text, model_text, user_text, 
        collaboratory_text, system_text, final_response, activity_type
    ))
    
    return {"ai_message": final_response, "structured_response": final_response}

async def extract_text(url, file):
    if url:
        text, _ = await scrape_url_content_async(url)
        return text
    elif file:
        return process_file(file)  
    return ""

def process_file(file):
    df = pd.read_csv(file.file) if file.filename.endswith(".csv") else pd.read_excel(file.file)
    return df.to_json()

if __name__ == "__main__":
    if env == "prod":
        uvicorn.run(app, host="0.0.0.0", port=8001, ssl_keyfile="proxyai.pem", ssl_certfile="proxyai.crt")
    else:
        uvicorn.run(app, host="localhost", port=8001)
