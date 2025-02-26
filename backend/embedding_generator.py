import json
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Define paths to retrieval materials
CHOICE_DATA_PATH = "choice_data.json"
JSON_FORMAT_PATH = "json_format.json"
SYSTEM_PROMPT_PATH = "system_prompt.txt"
COLLABORATORY_FORM_PATH = "collaboratory_activity_form.json"

# Define embedding storage paths
CHOICE_INDEX_PATH = "choice_data_index"
JSON_INDEX_PATH = "json_format_index"
SYSTEM_INDEX_PATH = "system_prompt_index"
COLLABORATORY_INDEX_PATH = "collaboratory_activity_form_index"

# Load OpenAI Embeddings
openai_api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

def load_text_file(file_path):
    """Load text content from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def load_json_file(file_path):
    """Load JSON content from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def generate_and_save_faiss_index(data, index_path):
    """Generate FAISS index from text data and save it."""
    vectorstore = FAISS.from_texts(data, embeddings)
    vectorstore.save_local(index_path)
    print(f"✅ FAISS index saved at: {index_path}")

def update_embeddings():
    """Regenerates FAISS embeddings whenever retrieval materials are updated."""
    print("🔄 Updating FAISS embeddings...")
    
    # Load retrieval materials
    choice_data = load_json_file(CHOICE_DATA_PATH)
    json_format = load_json_file(JSON_FORMAT_PATH)
    system_prompt = load_text_file(SYSTEM_PROMPT_PATH)
    collaboratory_form = load_json_file(COLLABORATORY_FORM_PATH)
    
    # Prepare text for embedding
    choice_texts = [str(item) for item in choice_data]  # Convert JSON objects to strings
    json_texts = [json.dumps(json_format, indent=2)]  # Convert JSON format to a structured string
    system_texts = [system_prompt]  # System prompt is already text
    collaboratory_texts = [json.dumps(collaboratory_form, indent=2)]  # Convert JSON to structured text
    
    # Generate and save FAISS indexes
    generate_and_save_faiss_index(choice_texts, CHOICE_INDEX_PATH)
    generate_and_save_faiss_index(json_texts, JSON_INDEX_PATH)
    generate_and_save_faiss_index(system_texts, SYSTEM_INDEX_PATH)
    generate_and_save_faiss_index(collaboratory_texts, COLLABORATORY_INDEX_PATH)
    
    print("✅ Embeddings updated successfully!")

if __name__ == "__main__":
    update_embeddings()