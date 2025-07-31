#!/usr/bin/env python3
"""
Script to check ChromaDB collections and help debug loading issues
"""

import chromadb
from pathlib import Path

def check_chroma_collections():
    """Check what collections exist in ChromaDB"""
    try:
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path="./data/chroma_db")
        
        # List all collections
        collections = client.list_collections()
        
        print("🔍 ChromaDB Collections Found:")
        print("=" * 50)
        
        if not collections:
            print("❌ No collections found!")
            return False
            
        for collection in collections:
            print(f"📁 Collection: {collection.name}")
            print(f"   Count: {collection.count()}")
            print(f"   Metadata: {collection.metadata}")
            print()
            
        return True
        
    except Exception as e:
        print(f"❌ Error checking ChromaDB: {e}")
        return False

def check_expected_collections():
    """Check if expected collections exist"""
    expected_collections = [
        "choice_data_enums",
        "pydantic_model", 
        "system_prompt",
        "user_prompt",
        "collaboratory_activity_form"
    ]
    
    try:
        client = chromadb.PersistentClient(path="./data/chroma_db")
        existing_collections = [col.name for col in client.list_collections()]
        
        print("🎯 Expected vs Actual Collections:")
        print("=" * 50)
        
        for expected in expected_collections:
            if expected in existing_collections:
                print(f"✅ {expected}")
            else:
                print(f"❌ {expected} - MISSING")
                
        return all(expected in existing_collections for expected in expected_collections)
        
    except Exception as e:
        print(f"❌ Error checking expected collections: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Checking ChromaDB Collections...")
    print("=" * 50)
    
    # Check what collections exist
    collections_exist = check_chroma_collections()
    
    if collections_exist:
        print("\n🎯 Checking Expected Collections...")
        all_expected_exist = check_expected_collections()
        
        if all_expected_exist:
            print("\n✅ All expected collections found!")
            print("🚀 You can now run: python server_chroma.py")
        else:
            print("\n❌ Some expected collections are missing!")
            print("🔄 Please run: python embedding_generator_chroma.py")
    else:
        print("\n❌ No ChromaDB collections found!")
        print("🔄 Please run: python embedding_generator_chroma.py") 