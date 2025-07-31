import os
import json
import shutil
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from data.choice_data_enums import (
    SustainableDevelopmentGoal,
    TargetPopulation,
    FocusAreaCategory,
    GoalOutput,
    GoalInstitutionalOutcome,
    GoalCommunityImpact,
    FocusAreaCategoryArtsAndCulture,
    FocusAreaCategoryCommunityAndEconomicDevelopment,
    FocusAreaCategoryEducation,
    FocusAreaCategoryEnvironmentalSustainability,
    FocusAreaCategoryGovernmentAndPublicSafety,
    FocusAreaCategoryHealthandWellness,
    FocusAreaCategorySocialIssues,
)
from data.activity_record_model import ActivityRecord

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")

# Define ChromaDB collection names
CHOICE_COLLECTION_NAME = "choice_data_enums"
PYDANTIC_MODEL_COLLECTION_NAME = "pydantic_model"
SYSTEM_COLLECTION_NAME = "system_prompt"
USER_PROMPT_COLLECTION_NAME = "user_prompt"
COLLABORATORY_COLLECTION_NAME = "collaboratory_activity_form"

# Define source files
SYSTEM_PROMPT_PATH = "data/system_prompt.txt"
USER_PROMPT_PATH = "data/user_prompt.txt"
COLLABORATORY_FORM_PATH = "data/collaboratory_activity_form.json"

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def delete_collection_if_exists(collection_name):
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./data/chroma_db")
        client.delete_collection(collection_name)
        print(f"🧹 Deleted existing ChromaDB collection: {collection_name}")
    except Exception as e:
        print(f"ℹ️ Collection {collection_name} doesn't exist or already deleted: {e}")

def generate_and_save_chroma_collection(texts, collection_name, metadata_list=None):
    if metadata_list is None:
        metadata_list = [{"source": collection_name} for _ in texts]
    
    # Create documents with IDs
    documents = [f"doc_{i}" for i in range(len(texts))]
    
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./data/chroma_db",
        metadatas=metadata_list
    )
    vectorstore.persist()
    print(f"✅ ChromaDB collection saved: {collection_name}")

def extract_enum_strings(*enum_classes):
    return [member.value for enum_class in enum_classes for member in enum_class]

def extract_pydantic_schema_description(model_class):
    return json.dumps(ActivityRecord.model_json_schema(), indent=2)

def update_embeddings():
    print("🔄 Updating ChromaDB embeddings...")

    # Delete old collections
    delete_collection_if_exists(CHOICE_COLLECTION_NAME)
    delete_collection_if_exists(PYDANTIC_MODEL_COLLECTION_NAME)
    delete_collection_if_exists(SYSTEM_COLLECTION_NAME)
    delete_collection_if_exists(USER_PROMPT_COLLECTION_NAME)
    delete_collection_if_exists(COLLABORATORY_COLLECTION_NAME)

    # Extract enum strings
    choice_texts = extract_enum_strings(
        SustainableDevelopmentGoal,
        TargetPopulation,
        FocusAreaCategory,
        GoalOutput,
        GoalInstitutionalOutcome,
        GoalCommunityImpact,
        FocusAreaCategoryArtsAndCulture,
        FocusAreaCategoryCommunityAndEconomicDevelopment,
        FocusAreaCategoryEducation,
        FocusAreaCategoryEnvironmentalSustainability,
        FocusAreaCategoryGovernmentAndPublicSafety,
        FocusAreaCategoryHealthandWellness,
        FocusAreaCategorySocialIssues,
    )

    # Extract Pydantic model schema
    pydantic_model_schema = [extract_pydantic_schema_description(ActivityRecord)]

    # Load additional materials
    system_prompt_text = [load_text_file(SYSTEM_PROMPT_PATH)]
    user_prompt_text = [load_text_file(USER_PROMPT_PATH)]
    collaboratory_form_text = [load_text_file(COLLABORATORY_FORM_PATH)]

    # Generate embeddings with metadata
    choice_metadata = [{"source": "enum", "type": "choice_data"} for _ in choice_texts]
    pydantic_metadata = [{"source": "schema", "type": "pydantic_model"}]
    system_metadata = [{"source": "prompt", "type": "system_prompt"}]
    user_metadata = [{"source": "prompt", "type": "user_prompt"}]
    collaboratory_metadata = [{"source": "form", "type": "collaboratory_form"}]

    # Generate embeddings
    generate_and_save_chroma_collection(choice_texts, CHOICE_COLLECTION_NAME, choice_metadata)
    generate_and_save_chroma_collection(pydantic_model_schema, PYDANTIC_MODEL_COLLECTION_NAME, pydantic_metadata)
    generate_and_save_chroma_collection(system_prompt_text, SYSTEM_COLLECTION_NAME, system_metadata)
    generate_and_save_chroma_collection(collaboratory_form_text, COLLABORATORY_COLLECTION_NAME, collaboratory_metadata)
    generate_and_save_chroma_collection(user_prompt_text, USER_PROMPT_COLLECTION_NAME, user_metadata)
    
    print("✅ All ChromaDB embeddings updated successfully!")

if __name__ == "__main__":
    update_embeddings() 