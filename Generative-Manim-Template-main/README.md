# Generative Manim

AI-powered educational video generation using Manim animations. Create professional educational videos with mathematical formulas, graphs, and visualizations through simple text prompts.

## Features

- **AI-Powered**: GPT-4o, Claude, and other LLMs for content generation
- **Domain Flexible**: Configurable for any educational domain
- **Multi-Format**: 16:9, 9:16, and 1:1 aspect ratios
- **Real-time Progress**: Live updates during generation
- **Cloud Ready**: Docker support with configurable storage backends

## Quick Start

### Prerequisites
- Python 3.9+
- FFmpeg
- LaTeX
- Docker (optional)

### Installation

```bash
git clone https://github.com/yourusername/generative-manim.git
cd generative-manim
pip install -r requirements.txt
```

### Configuration

Create `.env` file:
```bash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
USE_LOCAL_STORAGE=true
```

### Run

```bash
python api/run.py
```

API available at `http://localhost:8080`

## Usage

### Basic Example
```bash
curl -X POST http://localhost:8080/v1/video/rendering \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain the Pythagorean theorem",
    "model": "gpt-4o"
  }'
```

### Domain-Specific Generation

Use provided scripts for different domains:

```bash
# Physics
./generative-physics.sh "Explain Newton's second law"

# Chemistry  
./generative-chemistry.sh "Show molecular orbital theory"

# General
./generative-default.sh "Visualize sorting algorithms"
```

## Custom Domains

Create your own domain in `config/domains/`:

```json
{
  "domain": "Your Domain",
  "system_prompt": "You are an expert in...",
  "target_template": "Create educational video about...",
  "translation_rule": "Translate to standard terminology"
}
```

See [DOMAINS.md](DOMAINS.md) for detailed documentation.

## Docker Deployment

```bash
# Build and deploy
./docker_deploy.sh

# Run service
./service.sh

# Custom configuration
DOCKER_IMAGE_NAME=my-org/manim ./docker_deploy.sh
```

## API Endpoints

- `POST /v1/video/rendering` - Generate video from prompt
- `POST /v1/chat/generation` - Interactive chat mode
- `POST /v1/code/generation` - Generate storyboard
- `POST /v1/video/exporting` - Export videos

## Project Structure

```
generative-manim/
├── api/                    # Flask application
├── config/domains/         # Domain configurations  
├── media/                  # Generated videos
├── docker_deploy.sh        # Docker deployment
├── service.sh             # Service management
└── generative-*.sh        # Domain-specific scripts
```

## License

MIT License - see [LICENSE](LICENSE) file for details.