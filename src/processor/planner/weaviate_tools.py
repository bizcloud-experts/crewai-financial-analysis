import os
import json
from typing import List, Dict, Any, Optional
import weaviate
from weaviate.auth import AuthApiKey
from crewai_tools import BaseTool
from pydantic import BaseModel

class WeaviateConfig:
    """Configuration for Weaviate connection"""
    def __init__(self):
        self.url = os.getenv('WEAVIATE_URL', 'http://ec2-54-162-122-172.compute-1.amazonaws.com:8080')
        self.api_key = os.getenv('WEAVIATE_API_KEY', 'your-secret-api-key')
        # No OpenAI key needed - using AWS Bedrock

class WeaviateSearchTool(BaseTool):
    name: str = "Weaviate Document Search"
    description: str = """
    Search through financial documents and reports stored in Weaviate vector database.
    Use this tool to find relevant documents, policies, or reports that can provide context
    for financial analysis questions.
    
    Input should be a search query describing what financial information you're looking for.
    Examples:
    - "revenue recognition policies"
    - "quarterly financial reports Q4 2024"
    - "expense categorization guidelines"
    - "budget planning documents"
    """
    
    def _run(self, query: str) -> str:
        """Search Weaviate for relevant financial documents"""
        try:
            config = WeaviateConfig()
            
            # Connect to Weaviate
            client = weaviate.Client(
                url=config.url,
                auth_client_secret=AuthApiKey(api_key=config.api_key) if config.api_key else None
                # AWS Bedrock auth handled by IAM role
            )
            
            # Perform semantic search
            result = (
                client.query
                .get("FinancialDocument", ["title", "content", "document_type", "date", "summary"])
                .with_near_text({"concepts": [query]})
                .with_limit(5)
                .with_additional(["score", "id"])
                .do()
            )
            
            if not result.get('data', {}).get('Get', {}).get('FinancialDocument'):
                return f"No documents found for query: {query}"
            
            # Format results
            documents = result['data']['Get']['FinancialDocument']
            formatted_results = []
            
            for doc in documents:
                formatted_doc = f"""
Document: {doc.get('title', 'Untitled')}
Type: {doc.get('document_type', 'Unknown')}
Date: {doc.get('date', 'Unknown')}
Summary: {doc.get('summary', 'No summary available')}
Relevance Score: {doc.get('_additional', {}).get('score', 'N/A')}

Content Preview:
{doc.get('content', 'No content available')[:500]}...
"""
                formatted_results.append(formatted_doc)
            
            return "\n" + "="*50 + "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching Weaviate: {str(e)}"

class WeaviateMetricsTool(BaseTool):
    name: str = "Weaviate Financial Metrics Search"
    description: str = """
    Search for specific financial metrics, KPIs, and numerical data stored in Weaviate.
    Use this tool when you need to find specific financial numbers, ratios, or performance metrics.
    
    Input should describe the specific metric or financial data you're looking for.
    Examples:
    - "monthly recurring revenue MRR"
    - "customer acquisition cost CAC"
    - "gross profit margins by quarter"
    - "operating expenses breakdown"
    """
    
    def _run(self, metric_query: str) -> str:
        """Search for financial metrics in Weaviate"""
        try:
            config = WeaviateConfig()
            
            client = weaviate.Client(
                url=config.url,
                auth_client_secret=AuthApiKey(api_key=config.api_key) if config.api_key else None
                # AWS Bedrock auth handled by IAM role
            )
            
            # Search for financial metrics
            result = (
                client.query
                .get("FinancialMetric", ["metric_name", "value", "period", "category", "description", "source"])
                .with_near_text({"concepts": [metric_query]})
                .with_limit(10)
                .with_additional(["score"])
                .do()
            )
            
            if not result.get('data', {}).get('Get', {}).get('FinancialMetric'):
                return f"No financial metrics found for: {metric_query}"
            
            # Format metrics results
            metrics = result['data']['Get']['FinancialMetric']
            formatted_metrics = []
            
            for metric in metrics:
                formatted_metric = f"""
Metric: {metric.get('metric_name', 'Unknown')}
Value: {metric.get('value', 'N/A')}
Period: {metric.get('period', 'Unknown')}
Category: {metric.get('category', 'Unknown')}
Description: {metric.get('description', 'No description')}
Source: {metric.get('source', 'Unknown')}
Relevance: {metric.get('_additional', {}).get('score', 'N/A')}
"""
                formatted_metrics.append(formatted_metric)
            
            return "\nFINANCIAL METRICS FOUND:\n" + "\n" + "-"*40 + "\n".join(formatted_metrics)
            
        except Exception as e:
            return f"Error searching financial metrics: {str(e)}"

class WeaviateSchemaManager:
    """Manages Weaviate schema for financial data"""
    
    @staticmethod
    def create_financial_schema(client: weaviate.Client):
        """Create schema for financial documents and metrics"""
        
        # Financial Documents Schema
        financial_document_schema = {
            "class": "FinancialDocument",
            "description": "Financial documents, reports, and policies",
            "vectorizer": "text2vec-openai",
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
                    "description": "Type of document (report, policy, etc.)"
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
                },
                {
                    "name": "department",
                    "dataType": ["text"],
                    "description": "Originating department"
                }
            ]
        }
        
        # Financial Metrics Schema
        financial_metric_schema = {
            "class": "FinancialMetric",
            "description": "Financial metrics, KPIs, and numerical data",
            "vectorizer": "text2vec-openai",
            "properties": [
                {
                    "name": "metric_name",
                    "dataType": ["text"],
                    "description": "Name of the financial metric"
                },
                {
                    "name": "value",
                    "dataType": ["number"],
                    "description": "Metric value"
                },
                {
                    "name": "period",
                    "dataType": ["text"],
                    "description": "Time period (Q1 2024, etc.)"
                },
                {
                    "name": "category",
                    "dataType": ["text"],
                    "description": "Metric category (revenue, costs, etc.)"
                },
                {
                    "name": "description",
                    "dataType": ["text"],
                    "description": "Metric description and context"
                },
                {
                    "name": "source",
                    "dataType": ["text"],
                    "description": "Data source"
                }
            ]
        }
        
        try:
            # Create schemas if they don't exist
            if not client.schema.exists("FinancialDocument"):
                client.schema.create_class(financial_document_schema)
                print("Created FinancialDocument schema")
            
            if not client.schema.exists("FinancialMetric"):
                client.schema.create_class(financial_metric_schema)
                print("Created FinancialMetric schema")
                
        except Exception as e:
            print(f"Error creating schema: {e}")