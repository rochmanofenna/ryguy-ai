#!/bin/bash

# Generative Physics - Example script for generating physics educational videos

# Default values
API_URL="${API_URL:-http://localhost:8080}"
MODEL="${MODEL:-gpt-4o}"
DOMAIN="physics"

# Check if prompt is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 \"Your physics topic or concept\""
    echo "Example: $0 \"Explain Newton's second law with examples\""
    exit 1
fi

PROMPT="$1"

# Make API request to generate physics video
echo "Generating physics educational video..."
echo "Topic: $PROMPT"
echo "Using domain: $DOMAIN"

curl -X POST "${API_URL}/v1/video/rendering" \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"$PROMPT\",
    \"model\": \"$MODEL\",
    \"domain\": \"$DOMAIN\",
    \"aspect_ratio\": \"16:9\",
    \"stream\": true
  }"

echo -e "\n\nVideo generation complete!"