import json
import os
import boto3
from datetime import datetime

dynamodb = boto3.resource('dynamodb')

def lambda_handler(event, context):
    job_id = event.get('job_id')
    question = event.get('question')
    
    jobs_table = dynamodb.Table(os.environ['JOBS_TABLE'])
    
    try:
        print(f"Processing job {job_id}: {question}")
        
        # Simulate processing (replace with actual CrewAI code)
        result = f"Processed: {question}"
        
        # Update job as completed
        jobs_table.update_item(
            Key={'job_id': job_id},
            UpdateExpression="SET #status = :status, result = :result, updated_at = :updated_at",
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={
                ':status': 'completed',
                ':result': {'answer': result},
                ':updated_at': datetime.now().isoformat()
            }
        )
        
        return {'statusCode': 200}
        
    except Exception as e:
        # Update job as failed
        jobs_table.update_item(
            Key={'job_id': job_id},
            UpdateExpression="SET #status = :status, error = :error",
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={
                ':status': 'failed',
                ':error': str(e)
            }
        )
        raise e
