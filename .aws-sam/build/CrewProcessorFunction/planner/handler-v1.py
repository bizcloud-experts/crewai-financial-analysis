"""
Planner Agent for Moving Company Crew AI
Classifies questions and creates execution plans
"""

import json
import boto3
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any
import re

# AWS clients
bedrock = boto3.client('bedrock-runtime')

class PlannerAgent:
    """
    Planner Agent responsible for:
    1. Classifying incoming questions into types
    2. Creating detailed execution plans
    3. Decomposing complex questions into actionable steps
    """
    
    def __init__(self):
        self.question_types = {
            'factual_direct': 'Direct factual questions requiring specific information',
            'inferential': 'Questions requiring analysis and logical reasoning',
            'procedural': 'Questions about processes, how-to, and step-by-step guidance',
            'diagnostic': 'Questions about identifying problems and troubleshooting',
            'strategic_planning': 'Questions about planning, strategy, and decision-making',
            'predictive': 'Questions about forecasting and future outcomes'
        }
        
        # Moving company specific context
        self.moving_context = {
            'services': ['residential_moving', 'commercial_moving', 'storage', 'packing', 'long_distance', 'local_moving'],
            'common_issues': ['damage_claims', 'delays', 'pricing_disputes', 'scheduling_conflicts'],
            'processes': ['estimate_process', 'booking_process', 'moving_day_process', 'storage_process'],
            'resources': ['trucks', 'crew_members', 'packing_materials', 'storage_facilities']
        }

    def classify_question(self, question: str, context: Dict = None) -> Dict[str, Any]:
        """
        Classify the question type using Claude via Bedrock
        """
        
        classification_prompt = f"""
        You are a question classifier for a moving and storage company's AI system.
        
        IMPORTANT: Today's date is {datetime.utcnow().strftime('%B %d, %Y')} (Month Day, Year).
        Use this date to properly classify questions about past, present, and future events.
        
        Classify the following question into ONE of these categories:
        
        1. **factual_direct**: Direct questions about facts, rates, services, locations, hours, etc.
           Examples: "What are your storage rates?", "Do you serve Dallas?", "What's included in full-service packing?"
           ALSO INCLUDES: Questions about PAST events or data (like "How many moves were scheduled in March 2025" if March 2025 is in the past)
        
        2. **inferential**: Questions requiring analysis of information to draw conclusions
           Examples: "Which moving option would be best for my situation?", "Why might my estimate be higher than expected?"
        
        3. **procedural**: Questions about how to do something or step-by-step processes
           Examples: "How do I prepare for moving day?", "What's the process for filing a damage claim?"
        
        4. **diagnostic**: Questions about identifying problems or troubleshooting issues
           Examples: "Why was my delivery delayed?", "What went wrong with my estimate?", "How do I resolve a billing dispute?"
        
        5. **strategic_planning**: Questions about planning, decision-making, or strategic choices
           Examples: "Should I move in winter or summer?", "How should I plan a corporate relocation?", "What's the best moving strategy for my timeline?"
        
        6. **predictive**: Questions about future outcomes, forecasts, or predictions
           Examples: "How long will my move take?", "What will moving costs be next year?", "When is the best time to book?"
           NOTE: Only use this for questions about FUTURE events after today's date ({datetime.utcnow().strftime('%B %d, %Y')})
        
        Question to classify: "{question}"
        
        Context (if provided): {json.dumps(context) if context else 'None'}
        
        TIME-BASED CLASSIFICATION RULES:
        - If the question asks about events/data from BEFORE today ({datetime.utcnow().strftime('%B %d, %Y')}), classify as "factual_direct"
        - If the question asks about events/data for TODAY, classify as "factual_direct"  
        - If the question asks about events/data for AFTER today, then consider "predictive"
        - For questions about "when should I" or "what will happen", these are typically "strategic_planning" or "predictive"
        
        Respond with ONLY a JSON object in this exact format:
        {{
            "question_type": "category_name",
            "confidence": 0.95,
            "reasoning": "Brief explanation of classification including time consideration",
            "moving_domain": "specific area like residential_moving, storage, etc.",
            "complexity": "low|medium|high",
            "keywords": ["key", "terms", "identified"],
            "time_reference": "past|present|future|none"
        }}
        """
        
        try:
            response = bedrock.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                contentType='application/json',
                accept='application/json',
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": classification_prompt
                        }
                    ]
                })
            )
            
            response_body = json.loads(response['body'].read())
            classification_text = response_body['content'][0]['text']
            
            # Extract JSON from the response
            json_match = re.search(r'\{.*\}', classification_text, re.DOTALL)
            if json_match:
                classification = json.loads(json_match.group())
                return classification
            else:
                # Fallback classification
                return {
                    "question_type": "factual_direct",
                    "confidence": 0.5,
                    "reasoning": "Failed to parse classification, defaulting to factual",
                    "moving_domain": "general",
                    "complexity": "medium",
                    "keywords": [],
                    "time_reference": "none"
                }
                
        except Exception as e:
            print(f"Error in classification: {str(e)}")
            return {
                "question_type": "factual_direct",
                "confidence": 0.3,
                "reasoning": f"Classification error: {str(e)}",
                "moving_domain": "general",
                "complexity": "medium",
                "keywords": [],
                "time_reference": "none"
            }

    def create_execution_plan(self, question: str, classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a detailed execution plan based on the classification
        """
        
        plan_prompt = f"""
        You are creating an execution plan for a moving company AI agent to answer this question.
        
        Question: "{question}"
        Classification: {json.dumps(classification)}
        
        Create a detailed step-by-step execution plan. Consider:
        - What specific information needs to be gathered
        - What business rules need to be applied
        - What calculations might be needed
        - What customer data might be required
        - What external systems might need to be consulted
        - What follow-up actions might be needed
        
        Moving company context:
        - Services: {', '.join(self.moving_context['services'])}
        - Common processes: {', '.join(self.moving_context['processes'])}
        - Resources: {', '.join(self.moving_context['resources'])}
        
        Respond with a JSON object containing:
        {{
            "execution_steps": [
                {{
                    "step_number": 1,
                    "action": "specific action to take",
                    "description": "detailed description",
                    "data_needed": ["list", "of", "required", "data"],
                    "expected_output": "what this step should produce",
                    "dependencies": ["previous steps this depends on"]
                }}
            ],
            "required_data": ["overall data requirements"],
            "expected_duration": "estimated time in seconds",
            "complexity_factors": ["factors that increase complexity"],
            "success_criteria": ["how to measure success"],
            "potential_issues": ["possible problems and mitigation"]
        }}
        """
        
        try:
            response = bedrock.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                contentType='application/json',
                accept='application/json',
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2000,
                    "messages": [
                        {
                            "role": "user",
                            "content": plan_prompt
                        }
                    ]
                })
            )
            
            response_body = json.loads(response['body'].read())
            plan_text = response_body['content'][0]['text']
            
            # Extract JSON from the response
            json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
            if json_match:
                execution_plan = json.loads(json_match.group())
                return execution_plan
            else:
                # Fallback plan
                return self._create_fallback_plan(classification['question_type'])
                
        except Exception as e:
            print(f"Error creating execution plan: {str(e)}")
            return self._create_fallback_plan(classification['question_type'])

    def _create_fallback_plan(self, question_type: str) -> Dict[str, Any]:
        """Create a basic fallback execution plan"""
        
        base_steps = [
            {
                "step_number": 1,
                "action": "gather_context",
                "description": "Collect relevant context and customer information",
                "data_needed": ["customer_id", "service_type", "location"],
                "expected_output": "structured context data",
                "dependencies": []
            },
            {
                "step_number": 2,
                "action": "process_question",
                "description": f"Process {question_type} question with available data",
                "data_needed": ["context_data", "business_rules"],
                "expected_output": "processed response",
                "dependencies": ["gather_context"]
            },
            {
                "step_number": 3,
                "action": "format_response",
                "description": "Format final response for customer",
                "data_needed": ["processed_data"],
                "expected_output": "customer-ready response",
                "dependencies": ["process_question"]
            }
        ]
        
        return {
            "execution_steps": base_steps,
            "required_data": ["customer_context", "business_rules"],
            "expected_duration": "30",
            "complexity_factors": ["data_availability", "business_rule_complexity"],
            "success_criteria": ["accurate_response", "customer_satisfaction"],
            "potential_issues": ["missing_data", "unclear_requirements"]
        }

def lambda_handler(event, context):
    """
    Lambda handler for the Planner Agent
    """
    
    try:
        # Extract input data
        question = event.get('question', '')
        conversation_id = event.get('conversation_id', str(uuid.uuid4()))
        user_context = event.get('context', {})
        
        if not question:
            return {
                'error': 'Question is required',
                'conversation_id': conversation_id,
                'status': 'failed'
            }
        
        # Initialize planner agent
        planner = PlannerAgent()
        
        # Classify the question
        classification = planner.classify_question(question, user_context)
        
        # Create execution plan
        execution_plan = planner.create_execution_plan(question, classification)
        
        # Prepare response
        response = {
            'conversation_id': conversation_id,
            'question': question,
            'question_type': classification['question_type'],
            'classification': classification,
            'execution_plan': execution_plan,
            'next_agent': classification['question_type'],
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'planning_complete'
        }
        
        return response
        
    except Exception as e:
        print(f"Error in planner agent: {str(e)}")
        return {
            'error': f'Planning failed: {str(e)}',
            'conversation_id': event.get('conversation_id', 'unknown'),
            'status': 'planning_failed'
        }