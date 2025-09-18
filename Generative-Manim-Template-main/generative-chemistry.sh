#!/bin/bash

# Generative Chemistry - Example script for generating chemistry educational videos

# Default values
API_URL="${API_URL:-http://localhost:8080}"
MODEL="${MODEL:-gpt-4o}"
DOMAIN="chemistry"

# Check if prompt is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 \"Your chemistry topic or concept\""
    echo "Example: $0 \"Show the mechanism of SN2 reaction\""
    exit 1
fi

PROMPT="$1"

# Make API request to generate chemistry video
echo "Generating chemistry educational video..."
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