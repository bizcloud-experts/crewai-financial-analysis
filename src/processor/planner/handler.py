import json
import uuid
import os
import sys
from datetime import datetime
from typing import Dict, Any, List
import traceback

# SQLite fix for Lambda - must be before CrewAI imports
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
    print("Successfully replaced sqlite3 with pysqlite3")
except ImportError as e:
    print(f"Warning: Could not import pysqlite3: {e}")

# Configure CrewAI for Lambda environment
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CREWAI_STORAGE_DIR"] = "/tmp"
os.environ["HOME"] = "/tmp"
os.environ["TMPDIR"] = "/tmp"
os.environ["TEMP"] = "/tmp"
os.environ["TMP"] = "/tmp"

# Disable file output for tasks to prevent filesystem errors
os.environ["CREWAI_DISABLE_FILE_OUTPUT"] = "True"

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the financial crew
try:
    from crew import MovingFinancialCrew
except ImportError as e:
    print(f"Could not import crew: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Files in directory: {os.listdir(current_dir) if os.path.exists(current_dir) else 'Directory not found'}")
    raise

# Individual agent handlers for Step Functions

def task_planner_handler(event, context):
    """
    Step Function handler for financial task planning
    Input: {"question": "financial question", "context": {...}}
    Output: {"execution_plan": [...], "required_tables": [...], "next_steps": [...]}
    """
    try:
        question = event.get('question', '')
        analysis_context = event.get('context', {})
        
        print(f"Task Planner - Processing question: {question}")
        
        # Create just the task planner
        crew = MovingFinancialCrew()
        planning_task = crew.create_planning_task(question, analysis_context)
        
        # Execute the planning task
        result = planning_task.execute()
        
        return {
            'statusCode': 200,
            'body': {
                'execution_plan': str(result),
                'question': question,
                'context': analysis_context,
                'timestamp': datetime.utcnow().isoformat(),
                'step': 'task_planning',
                'success': True
            }
        }
        
    except Exception as e:
        print(f"Task planner error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return create_step_function_error('task_planning', str(e), event)

def metadata_handler(event, context):
    """
    Step Function handler for metadata retrieval
    Input: {"execution_plan": "...", "question": "..."}
    Output: {"schemas": {...}, "relationships": {...}, "query_guidance": "..."}
    """
    try:
        execution_plan = event.get('execution_plan', '')
        question = event.get('question', '')
        
        print(f"Metadata Agent - Processing question: {question}")
        
        crew = MovingFinancialCrew()
        metadata_task = crew.create_metadata_task(question, execution_plan)
        
        # Execute the metadata task
        result = metadata_task.execute()
        
        return {
            'statusCode': 200,
            'body': {
                'metadata_analysis': str(result),
                'schemas_retrieved': True,
                'question': question,
                'timestamp': datetime.utcnow().isoformat(),
                'step': 'metadata_retrieval',
                'success': True
            }
        }
        
    except Exception as e:
        print(f"Metadata agent error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return create_step_function_error('metadata_retrieval', str(e), event)

def query_execution_handler(event, context):
    """
    Step Function handler for query execution
    Input: {"metadata_analysis": "...", "execution_plan": "...", "question": "..."}
    Output: {"query_results": {...}, "summary_metrics": {...}}
    """
    try:
        metadata_analysis = event.get('metadata_analysis', '')
        execution_plan = event.get('execution_plan', '')
        question = event.get('question', '')
        
        print(f"Query Builder - Processing question: {question}")
        
        crew = MovingFinancialCrew()
        query_task = crew.create_query_task(question, metadata_analysis, execution_plan)
        
        # Execute the query task
        result = query_task.execute()
        
        return {
            'statusCode': 200,
            'body': {
                'query_results': str(result),
                'data_retrieved': True,
                'question': question,
                'timestamp': datetime.utcnow().isoformat(),
                'step': 'query_execution',
                'success': True
            }
        }
        
    except Exception as e:
        print(f"Query execution error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return create_step_function_error('query_execution', str(e), event)

def financial_analysis_handler(event, context):
    """
    Complete financial analysis using the full CrewAI crew
    Use this for simpler questions or when you want the full crew in one step
    """
    try:
        question = event.get('question', '')
        analysis_context = event.get('context', {})
        
        print(f"Complete Analysis - Processing question: {question}")
        
        crew = MovingFinancialCrew()
        result = crew.analyze_financial_question(question, analysis_context)
        
        return {
            'statusCode': 200,
            'body': {
                'analysis_result': result,
                'question': question,
                'context': analysis_context,
                'timestamp': datetime.utcnow().isoformat(),
                'step': 'complete_analysis',
                'success': True
            }
        }
        
    except Exception as e:
        print(f"Complete analysis error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return create_step_function_error('complete_analysis', str(e), event)

def reporting_handler(event, context):
    """
    Step Function handler for generating final reports
    Input: {"query_results": "...", "question": "...", "format": "summary|detailed|executive"}
    Output: {"report": "...", "key_insights": [...], "recommendations": [...]}
    """
    try:
        query_results = event.get('query_results', '')
        question = event.get('question', '')
        report_format = event.get('format', 'summary')
        
        print(f"Reporting - Formatting results for question: {question}")
        
        # Format the report
        formatted_report = format_financial_report(query_results, question, report_format)
        
        return {
            'statusCode': 200,
            'body': {
                'final_report': formatted_report,
                'report_format': report_format,
                'question': question,
                'timestamp': datetime.utcnow().isoformat(),
                'step': 'reporting',
                'success': True
            }
        }
        
    except Exception as e:
        print(f"Reporting error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return create_step_function_error('reporting', str(e), event)

# Main Lambda handler - routes to appropriate sub-handler
def lambda_handler(event, context):
    """
    Main handler - routes to appropriate sub-handler or runs complete analysis
    """
    try:
        print(f"Lambda handler called with event: {json.dumps(event, default=str)}")
        
        # Create /tmp directory structure if needed
        os.makedirs("/tmp/crewai", exist_ok=True)
        os.makedirs("/tmp/outputs", exist_ok=True)
        
        # Debug: Check what files are available
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Current directory: {current_dir}")
        try:
            files = os.listdir(current_dir)
            print(f"Files in current directory: {files}")
        except Exception as e:
            print(f"Could not list directory: {e}")
        
        # Check if this is a Step Function call with a specific step
        step = event.get('step')
        
        if step == 'task_planning':
            return task_planner_handler(event, context)
        elif step == 'metadata_retrieval':
            return metadata_handler(event, context)
        elif step == 'query_execution':
            return query_execution_handler(event, context)
        elif step == 'reporting':
            return reporting_handler(event, context)
        elif step == 'complete_analysis':
            return financial_analysis_handler(event, context)
        else:
            # Default to complete analysis for direct API calls
            return financial_analysis_handler(event, context)
            
    except Exception as e:
        print(f"Main handler error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
                'traceback': traceback.format_exc()
            })
        }

def create_step_function_error(step: str, error_message: str, event: Dict) -> Dict:
    """Create standardized error response for Step Functions"""
    return {
        'statusCode': 500,
        'body': {
            'error': error_message,
            'step': step,
            'input_event': event,
            'timestamp': datetime.utcnow().isoformat(),
            'success': False
        }
    }

def format_financial_report(query_results: str, question: str, format_type: str) -> str:
    """Format the final financial report based on requested format"""
    
    if format_type == 'executive':
        return f"""
EXECUTIVE FINANCIAL SUMMARY

Question: {question}

KEY FINDINGS:
{query_results}

STRATEGIC RECOMMENDATIONS:
[Based on the analysis above, key recommendations would be generated here]

NEXT ACTIONS:
[Specific next steps for leadership]
"""
    elif format_type == 'detailed':
        return f"""
DETAILED FINANCIAL ANALYSIS

Analysis Question: {question}
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

COMPLETE ANALYSIS:
{query_results}

DATA SOURCES AND METHODOLOGY:
[Details about data sources and calculation methods]

APPENDICES:
[Supporting data and detailed breakdowns]
"""
    else:  # summary
        return f"""
Financial Analysis Summary

Question: {question}
Analysis Results: {query_results}
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""

# Health check handler
def health_handler(event, context):
    """Health check endpoint"""
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'status': 'healthy',
            'service': 'moving-financial-analysis',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0'
        })
    }