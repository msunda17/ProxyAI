import os
import json
import shutil
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
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

# Define embedding save paths
CHOICE_INDEX_PATH = "data/choice_data_enums_index"
PYDANTIC_MODEL_INDEX_PATH = "data/pydantic_model_index"
SYSTEM_INDEX_PATH = "data/system_prompt_index"
USER_PROMPT_INDEX_PATH = "data/user_prompt_index"
COLLABORATORY_INDEX_PATH = "data/collaboratory_activity_form_index"

# Define source files
SYSTEM_PROMPT_PATH = "data/system_prompt.txt"
USER_PROMPT_PATH = "data/user_prompt.txt"
COLLABORATORY_FORM_PATH = "data/collaboratory_activity_form.json"

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def delete_index_if_exists(index_path):
    if os.path.exists(index_path):
        shutil.rmtree(index_path)
        print(f"🧹 Deleted existing FAISS index at: {index_path}")

def generate_and_save_faiss_index(texts, index_path):
    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local(index_path)
    print(f"✅ FAISS index saved at: {index_path}")

def extract_enum_strings(*enum_classes):
    return [member.value for enum_class in enum_classes for member in enum_class]

def extract_pydantic_schema_description(model_class):
    return json.dumps(ActivityRecord.model_json_schema(), indent=2)

def update_embeddings():
    print("🔄 Updating FAISS embeddings...")

    # Delete old indexes
    delete_index_if_exists(CHOICE_INDEX_PATH)
    delete_index_if_exists(PYDANTIC_MODEL_INDEX_PATH)
    delete_index_if_exists(SYSTEM_INDEX_PATH)
    delete_index_if_exists(USER_PROMPT_INDEX_PATH)
    delete_index_if_exists(COLLABORATORY_INDEX_PATH)

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

    # Generate embeddings
    generate_and_save_faiss_index(choice_texts, CHOICE_INDEX_PATH)
    generate_and_save_faiss_index(pydantic_model_schema, PYDANTIC_MODEL_INDEX_PATH)
    generate_and_save_faiss_index(system_prompt_text, SYSTEM_INDEX_PATH)
    generate_and_save_faiss_index(collaboratory_form_text, COLLABORATORY_INDEX_PATH)
    generate_and_save_faiss_index(user_prompt_text, USER_PROMPT_INDEX_PATH)

    print("✅ All embeddings updated successfully!")

if __name__ == "__main__":
    update_embeddings()