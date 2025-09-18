# Domain Configuration System

This document explains how to use and customize the domain configuration system in Generative Manim.

## Overview

The domain configuration system allows you to customize the educational content generation for different subjects and fields. Each domain has its own configuration file that defines:

- The system prompt used by the LLM
- Target templates for video generation
- Translation rules for non-English input
- Example topics

## Using Different Domains

### 1. Via Shell Scripts

We provide ready-to-use shell scripts for common domains:

```bash
# General educational content (default)
./generative-default.sh "Explain the Pythagorean theorem"

# Physics-specific content
./generative-physics.sh "Demonstrate conservation of momentum"

# Chemistry-specific content
./generative-chemistry.sh "Show the electron configuration of carbon"
```

### 2. Via API

When making API requests, include the `domain` parameter:

```json
{
  "prompt": "Your educational topic",
  "model": "gpt-4o",
  "domain": "physics",
  "aspect_ratio": "16:9"
}
```

### 3. Via Environment Variable

```bash
DOMAIN=chemistry ./generative-default.sh "Explain ionic bonding"
```

## Available Domains

- `default` - General educational content
- `physics` - Physics-specific animations and explanations
- `chemistry` - Chemistry visualizations and molecular animations

## Creating Custom Domains

To create a new domain configuration:

1. Create a new JSON file in `config/domains/` directory
2. Follow this structure:

```json
{
  "domain": "Your Domain Name",
  "system_prompt": "You are an expert in [domain] and Manim animation...",
  "target_template": "Create an educational video explaining [Topic] with...",
  "translation_rule": "If the user does not input in English...",
  "example_topics": [
    "Example topic 1",
    "Example topic 2"
  ]
}
```

3. Use your domain by referencing its filename (without .json):

```bash
curl -X POST "http://localhost:8080/v1/video/rendering" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Your topic",
    "domain": "your-domain-name"
  }'
```

## Example: Creating a Math Domain

1. Create `config/domains/math.json`:

```json
{
  "domain": "Mathematics Education",
  "system_prompt": "You are an expert mathematics educator and Manim animation specialist. You excel at creating clear visualizations of mathematical concepts, from basic arithmetic to advanced calculus.",
  "target_template": "Create an educational video explaining [Math Topic] with step-by-step visual proofs and examples",
  "translation_rule": "If the user does not input in English, translate to standard mathematical terminology.",
  "example_topics": [
    "Limits and Continuity",
    "Integration by Parts",
    "Matrix Multiplication",
    "Probability Distributions"
  ]
}
```

2. Create a shell script `generative-math.sh`:

```bash
#!/bin/bash
API_URL="${API_URL:-http://localhost:8080}"
MODEL="${MODEL:-gpt-4o}"
curl -X POST "${API_URL}/v1/video/rendering" \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"$1\",
    \"model\": \"$MODEL\",
    \"domain\": \"math\"
  }"
```

3. Use it:

```bash
./generative-math.sh "Visualize the chain rule in calculus"
```

## Docker Deployment

When using Docker, you can customize the image name and registry:

```bash
# Custom image name
DOCKER_IMAGE_NAME=my-org/educational-manim ./docker_deploy.sh

# Custom registry
./docker_deploy.sh -r my-registry.com:5000

# For service deployment
DOCKER_IMAGE_NAME=my-org/educational-manim \
SERVICE_NAME=my-manim-service \
./service.sh
```