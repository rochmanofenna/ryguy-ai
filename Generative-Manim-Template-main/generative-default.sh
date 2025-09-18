#!/bin/bash

# Generative Manim - Default script for generating general educational videos

# Default values
API_URL="${API_URL:-http://localhost:8080}"
MODEL="${MODEL:-gpt-4o}"
DOMAIN="${DOMAIN:-default}"

# Check if prompt is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 \"Your educational topic\""
    echo "Example: $0 \"Explain the water cycle\""
    echo ""
    echo "Optional: Set DOMAIN environment variable to use specific domain config"
    echo "Available domains: default, physics, chemistry"
    exit 1
fi

PROMPT="$1"

# Make API request to generate educational video
echo "Generating educational video..."
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