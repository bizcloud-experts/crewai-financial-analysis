#!/usr/bin/env python3
"""
Simple test script for Weaviate using HTTP-only connection
"""
import requests
import json

# Weaviate configuration
WEAVIATE_URL = "http://ec2-18-232-170-162.compute-1.amazonaws.com:8080"
WEAVIATE_API_KEY = "your-secret-api-key"

def test_connection():
    """Test basic connection to Weaviate via HTTP"""
    try:
        headers = {
            "Authorization": f"Bearer {WEAVIATE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Test meta endpoint
        response = requests.get(f"{WEAVIATE_URL}/v1/meta", headers=headers)
        response.raise_for_status()
        
        meta = response.json()
        print("✅ Connected to Weaviate successfully!")
        print(f"Version: {meta.get('version', 'Unknown')}")
        print(f"Modules: {list(meta.get('modules', {}).keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Weaviate: {e}")
        return False

def create_financial_schema():
    """Create schema for financial documents"""
    
    schema = {
        "class": "FinancialDocument",
        "description": "Financial documents, reports, and policies",
        "vectorizer": "text2vec-aws",
        "moduleConfig": {
            "text2vec-aws": {
                "model": "amazon.titan-embed-text-v1",
                "region": "us-east-1"
            }
        },
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
            },
            {
                "name": "date",
                "dataType": ["date"],
                "description": "Document date"
            },
            {
                "name": "summary",
                "dataType": ["text"],
                "description": "Document summary"
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
        
        if "FinancialDocument" in class_names:
            print("ℹ️  FinancialDocument class already exists")
            return True
        
        # Create class
        response = requests.post(f"{WEAVIATE_URL}/v1/schema", 
                               headers=headers, 
                               json=schema)
        response.raise_for_status()
        
        print("✅ Created FinancialDocument schema")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create schema: {e}")
        return False

def add_sample_document():
    """Add a sample financial document"""
    
    sample_doc = {
        "title": "Q4 2024 Revenue Report",
        "content": "Our Q4 2024 revenue reached $2.5M, representing a 15% increase from Q3. Key drivers included increased customer acquisition in the moving services sector and expansion of our premium service offerings. Operating expenses remained controlled at $1.8M, resulting in a healthy profit margin of 28%.",
        "document_type": "quarterly_report",
        "date": "2024-12-31T00:00:00Z",
        "summary": "Strong Q4 performance with revenue growth and controlled expenses"
    }
    
    try:
        headers = {
            "Authorization": f"Bearer {WEAVIATE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Add document
        response = requests.post(f"{WEAVIATE_URL}/v1/objects", 
                               headers=headers, 
                               json={"class": "FinancialDocument", "properties": sample_doc})
        response.raise_for_status()
        
        result = response.json()
        doc_id = result.get("id")
        
        print(f"✅ Added sample document with ID: {doc_id}")
        return doc_id
        
    except Exception as e:
        print(f"❌ Failed to add sample document: {e}")
        return None

def test_search():
    """Test semantic search"""
    
    try:
        headers = {
            "Authorization": f"Bearer {WEAVIATE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # GraphQL query for semantic search
        query = {
            "query": """
            {
                Get {
                    FinancialDocument(
                        nearText: {
                            concepts: ["revenue growth"]
                        }
                        limit: 3
                    ) {
                        title
                        document_type
                        summary
                        _additional {
                            score
                        }
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
        documents = result.get("data", {}).get("Get", {}).get("FinancialDocument", [])
        
        if documents:
            print("✅ Search test successful!")
            for doc in documents:
                print(f"  📄 {doc.get('title', 'Untitled')}")
                print(f"     Type: {doc.get('document_type', 'Unknown')}")
                print(f"     Score: {doc.get('_additional', {}).get('score', 'N/A')}")
        else:
            print("ℹ️  No documents found in search")
            
    except Exception as e:
        print(f"❌ Search test failed: {e}")

def main():
    print("🚀 Testing Weaviate connection and setup...")
    print(f"Connecting to: {WEAVIATE_URL}")
    
    # Test connection
    if not test_connection():
        return
    
    # Create schema
    if not create_financial_schema():
        return
    
    # Add sample document
    doc_id = add_sample_document()
    if doc_id:
        print("⏳ Waiting for document to be indexed...")
        import time
        time.sleep(3)  # Wait for indexing
        
        # Test search
        test_search()
    
    print("\n🎉 Weaviate setup complete!")
    print("Next steps:")
    print("1. Update CrewAI to use HTTP-based Weaviate tools")
    print("2. Deploy your updated CrewAI")
    print("3. Load your actual financial documents")

if __name__ == "__main__":
    main()