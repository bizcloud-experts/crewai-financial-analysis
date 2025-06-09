# Delete the failed stack
aws cloudformation delete-stack --stack-name rahul-dev-weaviate
aws cloudformation wait stack-delete-complete --stack-name rahul-dev-weaviate

# Deploy with the simple template
aws cloudformation create-stack \
  --stack-name rahul-dev-weaviate \
  --template-body file://simple-weaviate.yaml \
  --parameters \
    ParameterKey=Environment,ParameterValue=rahul-dev \
  --capabilities CAPABILITY_IAM

# Wait for creation
aws cloudformation wait stack-create-complete --stack-name rahul-dev-weaviate

# Get endpoint
ENDPOINT=$(aws cloudformation describe-stacks \
  --stack-name rahul-dev-weaviate \
  --query 'Stacks[0].Outputs[?OutputKey==`WeaviateEndpoint`].OutputValue' \
  --output text)

echo "✅ Weaviate endpoint: $ENDPOINT"

# Test connection
curl -H 'Authorization: Bearer your-secret-api-key' $ENDPOINT/v1/meta