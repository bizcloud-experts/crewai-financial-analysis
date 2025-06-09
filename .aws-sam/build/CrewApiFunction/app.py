import json
import uuid
import boto3
import os
from datetime import datetime, timedelta

lambda_client = boto3.client('lambda')
dynamodb = boto3.resource('dynamodb')

def lambda_handler(event, context):
    try:
        print(f"Event received: {json.dumps(event)}")
        
        # Get table name from environment
        jobs_table = dynamodb.Table(os.environ['JOBS_TABLE'])
        
        # Handle CORS preflight
        if event.get('httpMethod') == 'OPTIONS':
            return cors_response(200, {})
        
        path = event.get('path', '')
        method = event.get('httpMethod', '')
        
        print(f"Path: {path}, Method: {method}")
        
        if path == '/query' and method == 'POST':
            return start_job(event, jobs_table)
        elif path.startswith('/status/') and method == 'GET':
            # Extract job_id from path - handle multiple ways
            job_id = None
            
            # Method 1: pathParameters
            if event.get('pathParameters') and event['pathParameters'].get('job_id'):
                job_id = event['pathParameters']['job_id']
                print(f"Got job_id from pathParameters: {job_id}")
            
            # Method 2: manual path parsing
            if not job_id:
                path_parts = path.strip('/').split('/')
                print(f"Path parts: {path_parts}")
                if len(path_parts) >= 2 and path_parts[0] == 'status':
                    job_id = path_parts[1]
                    print(f"Got job_id from path parsing: {job_id}")
            
            if not job_id:
                print("Could not extract job_id")
                return cors_response(400, {'error': 'Job ID not found in path', 'path': path})
                
            return check_status(job_id, jobs_table)
        else:
            return cors_response(404, {'error': f'Not found: {method} {path}'})
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return cors_response(500, {'error': str(e)})

def start_job(event, jobs_table):
    body = json.loads(event.get('body', '{}'))
    question = body.get('question', '')
    context_data = body.get('context', {})
    
    if not question:
        return cors_response(400, {'error': 'Question is required'})
    
    job_id = str(uuid.uuid4())
    ttl = int((datetime.now() + timedelta(hours=24)).timestamp())
    
    # Store job in DynamoDB
    jobs_table.put_item(Item={
        'job_id': job_id,
        'status': 'processing',
        'question': question,
        'context': context_data,
        'created_at': datetime.now().isoformat(),
        'ttl': ttl
    })
    
    # Start background processor
    try:
        lambda_client.invoke(
            FunctionName=f"{os.environ['ENVIRONMENT']}-async-crew-processor",
            InvocationType='Event',
            Payload=json.dumps({
                'job_id': job_id,
                'question': question,
                'context': context_data
            })
        )
        print(f"Successfully invoked processor for job {job_id}")
    except Exception as e:
        print(f"Failed to invoke processor: {str(e)}")
        # Update job status to failed
        jobs_table.update_item(
            Key={'job_id': job_id},
            UpdateExpression='SET #status = :status, #error = :error',
            ExpressionAttributeNames={'#status': 'status', '#error': 'error'},
            ExpressionAttributeValues={':status': 'failed', ':error': str(e)}
        )
    
    return cors_response(202, {
        'job_id': job_id,
        'status': 'processing',
        'message': 'Your request is being processed',
        'check_status_url': f"/status/{job_id}"
    })

def check_status(job_id, jobs_table):
    try:
        print(f"Checking status for job_id: {job_id}")
        response = jobs_table.get_item(Key={'job_id': job_id})
        
        if 'Item' not in response:
            print(f"Job {job_id} not found in database")
            return cors_response(404, {'error': 'Job not found', 'job_id': job_id})
        
        job = response['Item']
        print(f"Found job: {job}")
        
        result = {
            'job_id': job_id,
            'status': job['status'],
            'created_at': job.get('created_at', ''),
            'question': job.get('question', '')
        }
        
        if job['status'] == 'completed':
            result['result'] = job.get('result', {})
        elif job['status'] == 'failed':
            result['error'] = job.get('error', 'Unknown error')
        
        return cors_response(200, result)
        
    except Exception as e:
        print(f"Error checking status: {str(e)}")
        import traceback
        traceback.print_exc()
        return cors_response(500, {'error': str(e)})

def cors_response(status_code, body):
    return {
        'statusCode': status_code,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
            'Content-Type': 'application/json'
        },
        'body': json.dumps(body)
    }
