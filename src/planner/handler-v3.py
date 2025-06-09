"""
Question Classification Agent for Moving Company
Uses CrewAI framework with AWS Bedrock for question classification only
This is the Planner Agent Function from your template.yaml
"""

# Fix sqlite3 version issue for ChromaDB before any other imports
import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
    print("Successfully replaced sqlite3 with pysqlite3")
except ImportError as e:
    print(f"Warning: Could not import pysqlite3: {e}")

# Configure writable directories for Lambda environment
import os
os.environ['HOME'] = '/tmp'
os.environ['TMPDIR'] = '/tmp'
os.environ['XDG_CACHE_HOME'] = '/tmp/.cache'
os.environ['XDG_DATA_HOME'] = '/tmp/.local/share'
os.environ['XDG_CONFIG_HOME'] = '/tmp/.config'
os.environ['CHROMA_DB_IMPL'] = 'duckdb+parquet'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Disable LiteLLM to prevent conflicts with direct Bedrock usage
os.environ['LITELLM_LOG'] = 'ERROR'
os.environ['LITELLM_DISABLE_TELEMETRY'] = 'True'

# Create writable directories if they don't exist
for dir_path in ['/tmp/.cache', '/tmp/.local/share', '/tmp/.config', '/tmp/chroma_db']:
    os.makedirs(dir_path, exist_ok=True)

import json
import uuid
from datetime import datetime
from typing import Dict, Any
import re

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai.llm import LLM


class MovingQuestionClassifier:
    """Moving company question classifier using CrewAI"""
    
    def __init__(self):
        # Initialize CrewAI's LLM wrapper for Bedrock
        try:
            # Use CrewAI's LLM class which properly handles Bedrock
            self.llm = LLM(
                model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                temperature=0.1,
                max_tokens=2000,
                top_p=0.9,
                aws_region_name=os.environ.get('AWS_REGION_TO_USE', 'us-east-1')
            )
            print("✅ CrewAI LLM initialized for Bedrock")
            
        except Exception as e:
            print(f"❌ CrewAI LLM initialization error: {str(e)}")
            # Fallback to basic configuration
            self.llm = LLM(
                model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
            )
    
    def create_question_classifier_agent(self):
        """Create an agent focused purely on question classification"""
        return Agent(
            role='Question Classification Expert',
            goal='Classify customer questions into categories and split multi-part questions. NEVER create execution plans.',
            backstory="""You are an expert at analyzing customer questions for a moving company. 
            Your ONLY job is to classify questions into these categories:
            - factual_direct: Direct factual questions requiring specific information
            - inferential: Questions requiring analysis and logical reasoning  
            - procedural: Questions about processes, how-to, and step-by-step guidance
            - diagnostic: Questions about identifying problems and troubleshooting
            - strategic_planning: Questions about planning and decision-making
            - predictive: Questions about forecasting and future outcomes
            
            You MUST split multi-part questions and classify each part separately.
            You NEVER create execution plans, steps, or detailed analysis - ONLY classification.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )
    
    def create_classification_task(self, question: str):
        """Create a task for classifying questions only"""
        return Task(
            description=f"""
            TASK: Classify this question and split if it contains multiple parts. Return ONLY JSON classification.

            Question: "{question}"

            CLASSIFICATION CATEGORIES:
            - factual_direct: Direct factual questions requiring specific information (e.g., "What are your rates?", "How many moves last month?")
            - inferential: Questions requiring analysis and logical reasoning (e.g., "Did it improve?", "Why did costs increase?", "Which option is better?")
            - procedural: Questions about processes, how-to, and step-by-step guidance (e.g., "How do I book a move?", "What's the moving process?")
            - diagnostic: Questions about identifying problems and troubleshooting (e.g., "Why is my booking not confirmed?", "What went wrong?")
            - strategic_planning: Questions about planning, strategy, and decision-making (e.g., "When should I schedule my move?", "What's the best plan?")
            - predictive: Questions about forecasting and future outcomes (e.g., "How long will my move take?", "Will prices increase?")

            CRITICAL INSTRUCTIONS:
            1. If the question has multiple parts separated by "?" or "and", split them into separate questions
            2. Classify each part independently
            3. Return ONLY the JSON response - no explanations, no execution plans, no steps
            4. Questions asking for data/numbers are factual_direct
            5. Questions asking for comparisons/analysis are inferential

            EXAMPLES:
            Question: "How many moves last month? Did it improve or decline?"
            Response: {{"questions": [{{"question": "How many moves last month?", "category": "factual_direct"}}, {{"question": "Did it improve or decline?", "category": "inferential"}}]}}

            Question: "What are your rates?"
            Response: {{"questions": [{{"question": "What are your rates?", "category": "factual_direct"}}]}}

            OUTPUT FORMAT (JSON ONLY - NO OTHER TEXT):
            {{
                "questions": [
                    {{
                        "question": "question text",
                        "category": "category_name"
                    }}
                ]
            }}
            """,
            agent=self.create_question_classifier_agent(),
            expected_output="JSON object with question classification only - no execution plans or steps"
        )
    
    def classify_question(self, question: str) -> Dict[str, Any]:
        """Classify a customer question and split if multi-part"""
        try:
            # Create classification task (no execution plan)
            task = self.create_classification_task(question)
            
            # Create crew with single agent for classification only
            crew = Crew(
                agents=[self.create_question_classifier_agent()],
                tasks=[task],
                process=Process.sequential,
                verbose=False  # Disable verbose to reduce noise
            )
            
            # Execute classification
            result = crew.kickoff()
            
            # Parse JSON response
            try:
                import json
                if hasattr(result, 'raw'):
                    response_text = result.raw
                else:
                    response_text = str(result)
                
                # Clean response text to extract JSON
                response_text = response_text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3]
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3]
                
                classification_result = json.loads(response_text)
                
                return {
                    'success': True,
                    'questions': classification_result.get('questions', []),
                    'timestamp': datetime.now().isoformat()
                }
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Raw response: {response_text}")
                
                # Fallback classification
                return {
                    'success': False,
                    'questions': [{
                        'question': question,
                        'category': 'factual_direct'
                    }],
                    'error': f'JSON parsing failed: {str(e)}',
                    'raw_response': response_text
                }
                
        except Exception as e:
            print(f"Classification error: {str(e)}")
            # Fallback classification
            return {
                'success': False,
                'questions': [{
                    'question': question,
                    'category': 'factual_direct'
                }],
                'error': f'Classification failed: {str(e)}'
            }


def lambda_handler(event, context):
    """
    AWS Lambda handler for the Planner Agent Function
    This is called by Step Functions from your template.yaml
    """
    try:
        print(f"Planner Agent received event: {json.dumps(event)}")
        
        # Initialize classifier
        classifier = MovingQuestionClassifier()
        
        # Extract question from Step Functions payload - handle different formats
        question = event.get('question', '')
        conversation_id = event.get('conversation_id', str(uuid.uuid4()))
        
        # Debug logging
        print(f"Extracted question: '{question}'")
        print(f"Conversation ID: '{conversation_id}'")
        
        if not question:
            print("ERROR: No question found in event")
            return {
                'statusCode': 400,
                'error': 'No question provided',
                'conversation_id': conversation_id,
                'question': '',
                'classification': {},
                'timestamp': datetime.now().isoformat(),
                'success': False
            }
        
        print(f"Classifying question: {question}")
        
        # Classify the question only (no execution plans)
        result = classifier.classify_question(question)
        
        print(f"Classification result: {json.dumps(result)}")
        
        # Return result in format expected by Step Functions
        response = {
            'statusCode': 200,
            'conversation_id': conversation_id,
            'question': question,
            'classification': result,
            'timestamp': datetime.now().isoformat(),
            'success': result.get('success', True)
        }
        
        print(f"Returning response: {json.dumps(response)}")
        return response
        
    except Exception as e:
        print(f"Planner Agent error: {str(e)}")
        error_response = {
            'statusCode': 500,
            'error': str(e),
            'conversation_id': event.get('conversation_id', str(uuid.uuid4())),
            'question': event.get('question', ''),
            'classification': {},
            'success': False,
            'timestamp': datetime.now().isoformat()
        }
        print(f"Error response: {json.dumps(error_response)}")
        return error_response