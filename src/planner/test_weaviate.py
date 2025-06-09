#!/usr/bin/env python3
"""
Test script to verify Weaviate connection and create initial schema
Updated for Weaviate client v4
"""
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure
import json

# Weaviate configuration
WEAVIATE_URL = "http://ec2-18-232-170-162.compute-1.amazonaws.com:8080"
WEAVIATE_API_KEY = "your-secret-api-key"

def test_connection():
    """Test basic connection to Weaviate"""
    try:
        client = weaviate.connect_to_custom(
            http_host="ec2-18-232-170-162.compute-1.amazonaws.com",
            http_port=8080,
            http_secure=False,
            grpc_host="ec2-18-232-170-162.compute-1.amazonaws.com",
            grpc_port=50051,
            grpc_secure=False,
            auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY)
        )
        
        # Test connection
        meta = client.get_meta()
        print("✅ Connected to Weaviate successfully!")
        print(f"Version: {meta.version}")
        
        # Check modules
        print(f"Modules loaded: {list(meta.modules.keys())}")
        
        return client
        
    except Exception as e:
        print(f"❌ Failed to connect to Weaviate: {e}")
        return None

def create_financial_schema(client):
    """Create schema for financial documents"""
    
    try:
        # Check if collection exists
        if client.collections.exists("FinancialDocument"):
            print("ℹ️  FinancialDocument collection already exists")
            return True
        
        # Create FinancialDocument collection
        client.collections.create(
            name="FinancialDocument",
            description="Financial documents, reports, and policies",
            vectorizer_config=Configure.Vectorizer.text2vec_aws(
                model="amazon.titan-embed-text-v1",
                region="us-east-1"
            ),
            properties=[
                wvc.config.Property(
                    name="title",
                    data_type=wvc.config.DataType.TEXT,
                    description="Document title"
                ),
                wvc.config.Property(
                    name="content",
                    data_type=wvc.config.DataType.TEXT,
                    description="Full document content"
                ),
                wvc.config.Property(
                    name="document_type",
                    data_type=wvc.config.DataType.TEXT,
                    description="Type of document (report, policy, etc.)"
                ),
                wvc.config.Property(
                    name="date",
                    data_type=wvc.config.DataType.DATE,
                    description="Document date"
                ),
                wvc.config.Property(
                    name="summary",
                    data_type=wvc.config.DataType.TEXT,
                    description="Document summary"
                )
            ]
        )
        
        print("✅ Created FinancialDocument collection")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create collection: {e}")
        return False

def add_sample_document(client):
    """Add a sample financial document"""
    
    sample_doc = {
        "title": "Q4 2024 Revenue Report",
        "content": "Our Q4 2024 revenue reached $2.5M, representing a 15% increase from Q3. Key drivers included increased customer acquisition in the moving services sector and expansion of our premium service offerings. Operating expenses remained controlled at $1.8M, resulting in a healthy profit margin of 28%.",
        "document_type": "quarterly_report",
        "date": "2024-12-31T00:00:00Z",
        "summary": "Strong Q4 performance with revenue growth and controlled expenses"
    }
    
    try:
        # Get the collection
        collection = client.collections.get("FinancialDocument")
        
        # Add document to Weaviate
        result = collection.data.insert(sample_doc)
        
        print(f"✅ Added sample document with ID: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Failed to add sample document: {e}")
        return None

def test_search(client):
    """Test semantic search"""
    
    try:
        # Get the collection
        collection = client.collections.get("FinancialDocument")
        
        # Search for revenue-related documents
        result = collection.query.near_text(
            query="revenue growth",
            limit=3,
            return_metadata=wvc.query.MetadataQuery(score=True)
        )
        
        if result.objects:
            print("✅ Search test successful!")
            for obj in result.objects:
                print(f"  📄 {obj.properties.get('title', 'Untitled')}")
                print(f"     Type: {obj.properties.get('document_type', 'Unknown')}")
                print(f"     Score: {obj.metadata.score}")
        else:
            print("ℹ️  No documents found in search")
            
    except Exception as e:
        print(f"❌ Search test failed: {e}")

def main():
    print("🚀 Testing Weaviate connection and setup...")
    print(f"Connecting to: {WEAVIATE_URL}")
    
    # Test connection
    client = test_connection()
    if not client:
        return
    
    try:
        # Create schema
        schema_created = create_financial_schema(client)
        if not schema_created:
            return
        
        # Add sample document
        doc_id = add_sample_document(client)
        if doc_id:
            print("⏳ Waiting for document to be indexed...")
            import time
            time.sleep(3)  # Wait for indexing
            
            # Test search
            test_search(client)
        
        print("\n🎉 Weaviate setup complete!")
        print("Next steps:")
        print("1. Update weaviate_tools.py to use v4 client")
        print("2. Deploy your updated CrewAI with Weaviate tools")
        print("3. Load your actual financial documents")
        print("4. Test CrewAI agents with semantic search")
        
    finally:
        # Close connection
        client.close()

if __name__ == "__main__":
    main()