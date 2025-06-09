"""
API Gateway Handler for Moving Company Crew AI
Handles /query and /status endpoints from your template.yaml
"""
import json
import boto3
import os
from datetime import datetime
from typing import Dict, Any
import uuid

# Initialize AWS clients
stepfunctions = boto3.client('stepfunctions')

def lambda_handler(event, context):
    """Main API handler for both /query and /status endpoints"""
    try:
        print(f"API received event: {json.dumps(event)}")
        
        # Get the HTTP method and path
        http_method = event.get('httpMethod', '')
        path = event.get('path', '')
        path_parameters = event.get('pathParameters') or {}
        
        # Route based on path and method
        if path == '/query' and http_method == 'POST':
            return handle_query(event, context)
        elif path.startswith('/status/') and http_method == 'GET':
            execution_id = path_parameters.get('execution_id')
            return handle_status(execution_id, event, context)
        else:
            return {
                'statusCode': 404,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'Endpoint not found',
                    'path': path,
                    'method': http_method
                })
            }
            
    except Exception as e:
        print(f"API handler error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e)
            })
        }

def handle_query(event, context):
    """Handle POST /query - Start new execution"""
    try:
        # Parse request body
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'No request body provided'
                })
            }
        
        question = body.get('question', '')
        if not question:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'No question provided'
                })
            }
        
        # Generate conversation ID
        conversation_id = str(uuid.uuid4())
        
        # Prepare input for Step Functions
        input_data = {
            'question': question,
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # Start Step Functions execution
        state_machine_arn = os.environ['STATE_MACHINE_ARN']
        execution_name = f"crew-ai-{conversation_id}-{int(datetime.now().timestamp())}"
        
        response = stepfunctions.start_execution(
            stateMachineArn=state_machine_arn,
            name=execution_name,
            input=json.dumps(input_data)
        )
        
        execution_arn = response['executionArn']
        execution_id = execution_arn.split(':')[-1]
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'execution_id': execution_id,
                'conversation_id': conversation_id,
                'status': 'started',
                'execution_arn': execution_arn,
                'question': question,
                'start_date': datetime.now().isoformat(),
                'timestamp': datetime.now().isoformat()
            })
        }
        
    except Exception as e:
        print(f"Error in handle_query: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f'Failed to start execution: {str(e)}'
            })
        }

def handle_status(execution_id, event, context):
    """Handle GET /status/{execution_id} - Check execution status"""
    try:
        if not execution_id:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'No execution_id provided'
                })
            }
        
        # Get execution ARN
        state_machine_arn = os.environ['STATE_MACHINE_ARN']
        account_id = state_machine_arn.split(':')[4]
        region = state_machine_arn.split(':')[3]
        state_machine_name = state_machine_arn.split(':')[-1]
        
        execution_arn = f"arn:aws:states:{region}:{account_id}:execution:{state_machine_name}:{execution_id}"
        
        # Get execution status
        response = stepfunctions.describe_execution(executionArn=execution_arn)
        
        result = {
            'execution_id': execution_id,
            'status': response['status'].lower(),
            'start_date': response['startDate'].isoformat(),
            'timestamp': datetime.now().isoformat()
        }
        
        # If execution completed, get the result
        if response['status'] in ['SUCCEEDED', 'FAILED']:
            if 'output' in response:
                try:
                    output = json.loads(response['output'])
                    
                    # Structure the response to match the format you showed
                    result['result'] = {
                        'ExecutedVersion': '$LATEST',
                        'Payload': {
                            'conversation_id': output.get('conversation_id'),
                            'question': output.get('question'),
                            'classification': output.get('classification', {}),
                            'timestamp': output.get('timestamp'),
                            'status': 'completed' if response['status'] == 'SUCCEEDED' else 'failed',
                            'success': output.get('success', True)
                        },
                        'StatusCode': 200
                    }
                    
                except json.JSONDecodeError:
                    result['result'] = response['output']
            
            if response['status'] == 'FAILED':
                result['error'] = response.get('error', 'Execution failed')
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result)
        }
        
    except Exception as e:
        print(f"Error in handle_status: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f'Failed to check execution status: {str(e)}'
            })
        }