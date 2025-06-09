"""
Planner Agent for Moving Company Crew AI
Uses CrewAI framework with AWS Bedrock for question classification and planning
"""

# Fix sqlite3 version issue for ChromaDB before any other imports
import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

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

class MovingCompanyKnowledgeTool(BaseTool):
    """Tool that provides moving company context and knowledge"""
    
    name: str = "moving_company_knowledge"
    description: str = "Get comprehensive moving company services, policies, pricing, and business context information"
    
    def _run(self, query: str) -> str:
        """Return moving company knowledge base based on query"""
        knowledge_base = {
            'services': {
                'residential_moving': {
                    'description': 'Full-service residential moving for homes and apartments',
                    'includes': ['packing', 'loading', 'transport', 'unloading', 'basic_assembly'],
                    'pricing': '$120/hour for 2 movers + truck',
                    'minimum_hours': 2
                },
                'commercial_moving': {
                    'description': 'Office and business relocations',
                    'includes': ['equipment_handling', 'IT_disconnect_reconnect', 'furniture_assembly'],
                    'pricing': '$150/hour for 3 movers + truck',
                    'minimum_hours': 4
                },
                'storage': {
                    'description': 'Climate-controlled storage facilities',
                    'unit_sizes': ['5x5', '5x10', '10x10', '10x15', '10x20'],
                    'rates': {
                        '5x5': '$75/month',
                        '5x10': '$95/month',
                        '10x10': '$125/month',
                        '10x15': '$175/month',
                        '10x20': '$225/month'
                    },
                    'security': '24/7 surveillance, keypad access'
                },
                'packing': {
                    'description': 'Professional packing services',
                    'full_packing': '$45/hour per packer',
                    'partial_packing': '$50/hour per packer',
                    'materials_included': True
                },
                'long_distance': {
                    'description': 'Interstate and cross-country moves',
                    'pricing': 'Based on weight and distance',
                    'estimate_range': '$2,500 - $8,000 for typical household',
                    'timeline': '3-14 business days'
                }
            },
            'coverage_areas': [
                'Dallas-Fort Worth Metroplex',
                'Houston Metro', 
                'Austin-Round Rock',
                'San Antonio',
                'El Paso',
                'Corpus Christi'
            ],
            'business_hours': {
                'monday_friday': '7:00 AM - 7:00 PM',
                'saturday': '8:00 AM - 5:00 PM',
                'sunday': '9:00 AM - 4:00 PM',
                'holidays': 'Limited availability, call for details'
            },
            'policies': {
                'cancellation': '24-hour notice required for free cancellation',
                'rescheduling': 'Free rescheduling with 48-hour notice',
                'payment': 'Cash, check, credit card accepted',
                'deposit': '25% deposit required for bookings over $500',
                'insurance': 'Basic coverage included, full-value protection available'
            },
            'common_processes': {
                'estimate_process': [
                    'Schedule in-home or virtual estimate',
                    'Assess items and requirements',
                    'Provide written estimate within 24 hours',
                    'Schedule move date upon acceptance'
                ],
                'booking_process': [
                    'Accept estimate and terms',
                    'Pay required deposit',
                    'Receive confirmation and crew details',
                    'Pre-move contact 24 hours before'
                ]
            }
        }
        
        # Return relevant subset based on query
        if not query or query.lower() == 'all':
            return json.dumps(knowledge_base, indent=2)
        
        # Simple keyword matching for relevant sections
        query_lower = query.lower()
        relevant_data = {}
        
        if any(word in query_lower for word in ['service', 'price', 'rate', 'cost']):
            relevant_data['services'] = knowledge_base['services']
        
        if any(word in query_lower for word in ['area', 'location', 'coverage', 'serve']):
            relevant_data['coverage_areas'] = knowledge_base['coverage_areas']
        
        if any(word in query_lower for word in ['hour', 'time', 'schedule', 'when']):
            relevant_data['business_hours'] = knowledge_base['business_hours']
        
        if any(word in query_lower for word in ['policy', 'cancel', 'payment', 'deposit']):
            relevant_data['policies'] = knowledge_base['policies']
        
        if any(word in query_lower for word in ['process', 'how', 'step', 'procedure']):
            relevant_data['common_processes'] = knowledge_base['common_processes']
        
        return json.dumps(relevant_data if relevant_data else knowledge_base, indent=2)

class PlannerCrew:
    """
    CrewAI implementation for question planning and classification using AWS Bedrock
    """
    
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
        
        # Initialize tools
        self.knowledge_tool = MovingCompanyKnowledgeTool()
        
        # Create agents
        self.classifier_agent = self._create_classifier_agent()
        self.planner_agent = self._create_planner_agent()
        
    def _create_classifier_agent(self) -> Agent:
        """Create the question classifier agent"""
        return Agent(
            role="Expert Question Classifier",
            goal="Accurately classify customer questions into the most appropriate category to ensure optimal handling and customer satisfaction",
            backstory=f"""You are a seasoned customer service analyst with over 10 years of experience in the moving and storage industry.
            You have handled thousands of customer inquiries and have developed an exceptional ability to understand customer intent.
            
            IMPORTANT CONTEXT:
            - Today's date is {datetime.utcnow().strftime('%B %d, %Y')}
            - Use this date to properly distinguish between past, present, and future events
            - You understand the nuances between different types of customer questions
            
            Your expertise includes:
            - Recognizing direct factual questions (including historical data requests)
            - Identifying questions that require analytical thinking
            - Understanding process and procedure inquiries
            - Diagnosing problem-solving situations
            - Recognizing strategic planning needs
            - Distinguishing predictive/forecasting requests
            
            You always consider the temporal aspect of questions to classify them correctly.
            """,
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.knowledge_tool]
        )
    
    def _create_planner_agent(self) -> Agent:
        """Create the execution planner agent"""
        return Agent(
            role="Strategic Execution Planner",
            goal="Create comprehensive, actionable execution plans that lead to exceptional customer service outcomes",
            backstory="""You are a strategic operations expert specializing in customer service workflow optimization.
            With extensive experience in process design and customer journey mapping, you excel at breaking down
            complex customer service scenarios into clear, executable steps.
            
            Your expertise includes:
            - Identifying required data sources and business rules
            - Anticipating potential complications and edge cases
            - Designing efficient workflows that minimize customer wait time
            - Creating contingency plans for various scenarios
            - Ensuring compliance with company policies and procedures
            
            You always prioritize customer satisfaction while maintaining operational efficiency.
            Your plans are detailed enough for execution but flexible enough to handle variations.
            """,
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.knowledge_tool]
        )
    
    def process_question(self, question: str, context: Dict = None) -> Dict[str, Any]:
        """
        Process a question using the CrewAI framework with Bedrock
        """
        
        # Create classification task
        classification_task = Task(
            description=f"""
            Analyze and classify this customer question: "{question}"
            
            Additional context provided: {json.dumps(context) if context else 'None provided'}
            
            CLASSIFICATION CATEGORIES:
            1. **factual_direct** - Direct questions about facts, rates, services, policies, hours, locations
               - INCLUDES: Questions about past events or historical data
               - Examples: "What are your storage rates?", "How many moves did you complete in March 2025?"
            
            2. **inferential** - Questions requiring analysis, comparison, or logical reasoning
               - Examples: "Which moving option would be best for my situation?", "Why might my estimate be higher?"
            
            3. **procedural** - Questions about processes, how-to guidance, step-by-step instructions
               - Examples: "How do I prepare for moving day?", "What's the process for filing a claim?"
            
            4. **diagnostic** - Questions about identifying problems, troubleshooting, or issue resolution
               - Examples: "Why was my delivery delayed?", "How do I resolve a billing dispute?"
            
            5. **strategic_planning** - Questions about planning, decision-making, or strategic choices
               - Examples: "Should I move in winter or summer?", "How should I plan a corporate relocation?"
            
            6. **predictive** - Questions about future outcomes, forecasts, or predictions
               - Examples: "How long will my move take?", "What will moving costs be next year?"
               - NOTE: Only for questions about FUTURE events after today's date
            
            CRITICAL RULE: Questions about events BEFORE today ({datetime.utcnow().strftime('%B %d, %Y')}) 
            should be classified as 'factual_direct', not 'predictive'.
            
            Use the moving company knowledge tool to understand our services and context.
            
            REQUIRED OUTPUT FORMAT (JSON):
            {{
                "question_type": "category_name",
                "confidence": 0.95,
                "reasoning": "Detailed explanation including time consideration if relevant",
                "moving_domain": "specific area like residential_moving, storage, commercial_moving, etc.",
                "complexity": "low|medium|high",
                "keywords": ["key", "terms", "identified", "in", "question"],
                "time_reference": "past|present|future|none",
                "requires_data_lookup": true/false,
                "customer_intent": "brief description of what customer really wants"
            }}
            """,
            agent=self.classifier_agent,
            expected_output="JSON object with detailed classification including time analysis and customer intent"
        )
        
        # Create planning task
        planning_task = Task(
            description=f"""
            Create a comprehensive execution plan for handling this question: "{question}"
            
            Use the classification results from the previous task to inform your planning.
            Consider the moving company's services, policies, and operational capabilities.
            
            PLANNING CONSIDERATIONS:
            - What specific data needs to be gathered from which sources
            - What business rules and policies need to be applied
            - What calculations or analysis might be required
            - What external systems or databases might need to be consulted
            - What potential issues could arise and how to mitigate them
            - What follow-up actions might be needed
            - How to ensure customer satisfaction throughout the process
            
            Use the moving company knowledge tool to understand available resources and constraints.
            
            REQUIRED OUTPUT FORMAT (JSON):
            {{
                "execution_steps": [
                    {{
                        "step_number": 1,
                        "action": "specific_action_to_take",
                        "description": "detailed description of what needs to be done",
                        "data_needed": ["list", "of", "required", "data", "sources"],
                        "expected_output": "what this step should produce",
                        "dependencies": ["previous", "steps", "this", "depends", "on"],
                        "estimated_time": "time in seconds",
                        "potential_issues": ["possible", "problems"],
                        "mitigation_strategies": ["how", "to", "handle", "issues"]
                    }}
                ],
                "required_data": ["overall", "data", "requirements"],
                "expected_duration": "total estimated time in seconds",
                "complexity_factors": ["factors", "that", "increase", "complexity"],
                "success_criteria": ["how", "to", "measure", "success"],
                "potential_issues": ["possible", "problems", "and", "mitigation"],
                "customer_touchpoints": ["points", "where", "customer", "interaction", "needed"],
                "escalation_triggers": ["conditions", "that", "require", "human", "intervention"]
            }}
            """,
            agent=self.planner_agent,
            expected_output="JSON object with comprehensive execution plan including contingencies",
            context=[classification_task]
        )
        
        # Create and execute crew
        crew = Crew(
            agents=[self.classifier_agent, self.planner_agent],
            tasks=[classification_task, planning_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            # Execute the crew
            print(f"🚀 Starting CrewAI processing for question: {question[:50]}...")
            result = crew.kickoff()
            
            # Extract results from task outputs
            classification_output = str(classification_task.output) if classification_task.output else ""
            planning_output = str(planning_task.output) if planning_task.output else ""
            
            # Extract JSON from outputs
            classification = self._extract_json_from_output(classification_output)
            execution_plan = self._extract_json_from_output(planning_output)
            
            # Validate classification
            if not classification.get('question_type'):
                classification = self._create_fallback_classification()
            
            # Validate execution plan
            if not execution_plan.get('execution_steps'):
                execution_plan = self._create_fallback_plan(classification.get('question_type', 'factual_direct'))
            
            print(f"✅ CrewAI processing completed successfully")
            
            return {
                'classification': classification,
                'execution_plan': execution_plan,
                'crew_result': str(result)[:500] + "..." if len(str(result)) > 500 else str(result),
                'tasks_completed': len([t for t in crew.tasks if hasattr(t, 'output') and t.output]),
                'processing_time': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error in CrewAI execution: {str(e)}")
            # Comprehensive fallback response
            return {
                'classification': self._create_fallback_classification(str(e)),
                'execution_plan': self._create_fallback_plan('factual_direct'),
                'crew_result': f"CrewAI execution encountered an error: {str(e)}",
                'tasks_completed': 0,
                'processing_time': datetime.utcnow().isoformat(),
                'error_details': str(e)
            }
    
    def _extract_json_from_output(self, output: str) -> Dict[str, Any]:
        """Extract JSON from CrewAI task output with multiple fallback strategies"""
        if not output:
            return {}
        
        try:
            # Strategy 1: Look for JSON blocks in markdown
            json_block_pattern = r'```json\s*(\{.*?\})\s*```'
            json_match = re.search(json_block_pattern, output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Strategy 2: Look for any JSON-like structure
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, output, re.DOTALL)
            
            for match in json_matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict) and len(parsed) > 1:  # Valid dict with content
                        return parsed
                except json.JSONDecodeError:
                    continue
            
            # Strategy 3: Try to parse the entire output as JSON
            return json.loads(output.strip())
            
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"⚠️ JSON extraction failed: {str(e)}")
            return {}
    
    def _create_fallback_classification(self, error_msg: str = "") -> Dict[str, Any]:
        """Create a fallback classification when extraction fails"""
        return {
            "question_type": "factual_direct",
            "confidence": 0.3,
            "reasoning": f"Fallback classification due to processing error: {error_msg}" if error_msg else "Fallback classification - unable to parse CrewAI output",
            "moving_domain": "general",
            "complexity": "medium",
            "keywords": [],
            "time_reference": "none",
            "requires_data_lookup": True,
            "customer_intent": "Customer needs assistance with moving-related inquiry"
        }
    
    def _create_fallback_plan(self, question_type: str) -> Dict[str, Any]:
        """Create a fallback execution plan when extraction fails"""
        return {
            "execution_steps": [
                {
                    "step_number": 1,
                    "action": "gather_context",
                    "description": "Collect relevant context and customer information",
                    "data_needed": ["customer_context", "question_details", "service_requirements"],
                    "expected_output": "structured context data",
                    "dependencies": [],
                    "estimated_time": "15",
                    "potential_issues": ["incomplete_information"],
                    "mitigation_strategies": ["request_additional_details"]
                },
                {
                    "step_number": 2,
                    "action": "apply_business_rules",
                    "description": f"Apply relevant business rules for {question_type} questions",
                    "data_needed": ["context_data", "business_policies", "service_catalog"],
                    "expected_output": "processed response with applied rules",
                    "dependencies": ["gather_context"],
                    "estimated_time": "20",
                    "potential_issues": ["policy_conflicts"],
                    "mitigation_strategies": ["escalate_to_supervisor"]
                },
                {
                    "step_number": 3,
                    "action": "format_response",
                    "description": "Format final response for customer delivery",
                    "data_needed": ["processed_data"],
                    "expected_output": "customer-ready response",
                    "dependencies": ["apply_business_rules"],
                    "estimated_time": "10",
                    "potential_issues": ["formatting_errors"],
                    "mitigation_strategies": ["manual_review"]
                }
            ],
            "required_data": ["customer_context", "business_rules", "service_catalog"],
            "expected_duration": "45",
            "complexity_factors": ["data_availability", "policy_complexity"],
            "success_criteria": ["accurate_response", "customer_satisfaction", "policy_compliance"],
            "potential_issues": ["missing_data", "unclear_requirements", "system_errors"],
            "customer_touchpoints": ["initial_inquiry", "clarification_requests", "final_response"],
            "escalation_triggers": ["complex_policy_questions", "pricing_disputes", "service_complaints"]
        }

def lambda_handler(event, context):
    """
    Lambda handler for the CrewAI Planner using Bedrock
    """
    
    try:
        # Extract input data
        question = event.get('question', '').strip()
        conversation_id = event.get('conversation_id', str(uuid.uuid4()))
        user_context = event.get('context', {})
        
        if not question:
            return {
                'error': 'Question is required',
                'conversation_id': conversation_id,
                'status': 'failed',
                'framework': 'crewai_bedrock'
            }
        
        print(f"🎯 Processing question: {question}")
        print(f"📝 Conversation ID: {conversation_id}")
        
        # Initialize CrewAI
        planner_crew = PlannerCrew()
        
        # Process question using CrewAI
        crew_result = planner_crew.process_question(question, user_context)
        
        # Prepare comprehensive response
        response = {
            'conversation_id': conversation_id,
            'question': question,
            'question_type': crew_result['classification'].get('question_type', 'factual_direct'),
            'classification': crew_result['classification'],
            'execution_plan': crew_result['execution_plan'],
            'crew_execution_details': {
                'tasks_completed': crew_result['tasks_completed'],
                'processing_time': crew_result['processing_time'],
                'crew_summary': crew_result['crew_result']
            },
            'next_agent': crew_result['classification'].get('question_type', 'factual_direct'),
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'planning_complete',
            'framework': 'crewai_bedrock',
            'confidence': crew_result['classification'].get('confidence', 0.5)
        }
        
        print(f"✅ Successfully processed question with {crew_result['tasks_completed']} tasks completed")
        
        return response
        
    except Exception as e:
        error_msg = f'CrewAI planning failed: {str(e)}'
        print(f"❌ {error_msg}")
        
        return {
            'error': error_msg,
            'conversation_id': event.get('conversation_id', 'unknown'),
            'status': 'planning_failed',
            'framework': 'crewai_bedrock',
            'timestamp': datetime.utcnow().isoformat(),
            'error_details': {
                'error_type': type(e).__name__,
                'error_message': str(e)
            }
        }