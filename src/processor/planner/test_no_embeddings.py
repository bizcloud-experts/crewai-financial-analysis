#!/usr/bin/env python3
"""
Test Weaviate without embeddings (manual vectors)
"""
import requests
import json

WEAVIATE_URL = "http://ec2-54-162-122-172.compute-1.amazonaws.com:8080"
WEAVIATE_API_KEY = "your-secret-api-key"

def create_simple_schema():
    """Create schema without vectorizer"""
    
    schema = {
        "class": "SimpleFinancialDoc",
        "description": "Simple financial documents without auto-vectorization",
        "vectorizer": "none",  # No automatic vectorization
        "properties": [
            {
                "name": "title",
                "dataType": ["text"],
                "description": "Document title"
            },
            {
                "name": "content",
                "dataType": ["text"],
                "description": "Full document content"
            },
            {
                "name": "document_type",
                "dataType": ["text"],
                "description": "Type of document"
            }
        ]
    }
    
    try:
        headers = {
            "Authorization": f"Bearer {WEAVIATE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Check if class exists
        response = requests.get(f"{WEAVIATE_URL}/v1/schema", headers=headers)
        response.raise_for_status()
        
        existing_classes = response.json().get("classes", [])
        class_names = [cls.get("class") for cls in existing_classes]
        
        if "SimpleFinancialDoc" in class_names:
            print("ℹ️  SimpleFinancialDoc class already exists")
            return True
        
        # Create class
        response = requests.post(f"{WEAVIATE_URL}/v1/schema", 
                               headers=headers, 
                               json=schema)
        response.raise_for_status()
        
        print("✅ Created SimpleFinancialDoc schema (no embeddings)")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create schema: {e}")
        return False

def add_simple_document():
    """Add a document without automatic vectorization"""
    
    sample_doc = {
        "title": "Test Financial Report",
        "content": "This is a test financial document to verify Weaviate storage works.",
        "document_type": "test_report"
    }
    
    try:
        headers = {
            "Authorization": f"Bearer {WEAVIATE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Add document without vector
        response = requests.post(f"{WEAVIATE_URL}/v1/objects", 
                               headers=headers, 
                               json={"class": "SimpleFinancialDoc", "properties": sample_doc})
        response.raise_for_status()
        
        result = response.json()
        doc_id = result.get("id")
        
        print(f"✅ Added simple document with ID: {doc_id}")
        return doc_id
        
    except Exception as e:
        print(f"❌ Failed to add document: {e}")
        print(f"Response: {e.response.text if hasattr(e, 'response') else 'No response'}")
        return None

def list_documents():
    """List all documents"""
    
    try:
        headers = {
            "Authorization": f"Bearer {WEAVIATE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        query = {
            "query": """
            {
                Get {
                    SimpleFinancialDoc {
                        title
                        document_type
                        content
                    }
                }
            }
            """
        }
        
        response = requests.post(f"{WEAVIATE_URL}/v1/graphql", 
                               headers=headers, 
                               json=query)
        response.raise_for_status()
        
        result = response.json()
        documents = result.get("data", {}).get("Get", {}).get("SimpleFinancialDoc", [])
        
        if documents:
            print("✅ Documents in Weaviate:")
            for doc in documents:
                print(f"  📄 {doc.get('title', 'Untitled')}")
                print(f"     Type: {doc.get('document_type', 'Unknown')}")
        else:
            print("ℹ️  No documents found")
            
    except Exception as e:
        print(f"❌ Failed to list documents: {e}")

def main():
    print("🚀 Testing Weaviate basic functionality...")
    
    # Create simple schema
    if not create_simple_schema():
        return
    
    # Add document
    doc_id = add_simple_document()
    if doc_id:
        # List documents
        list_documents()
    
    print("\n✅ Basic Weaviate functionality test complete!")

if __name__ == "__main__":
    main()