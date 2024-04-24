#!/bin/bash

AWS_SECRET_ID="dev/analitiq/env"
AWS_REGION="eu-central-1"
ENVFILE=".env"

aws secretsmanager get-secret-value --secret-id $AWS_SECRET_ID --region $AWS_REGION | jq -r '.SecretString'  > $ENVFILE
