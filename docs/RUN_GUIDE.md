# DIZI-AI Run Guide

## Dashboard

```powershell
cd C:\Users\user\OneDrive\Desktop\DIZI-AI
python backend/server.py
```

Open:

```text
http://localhost:5001
```

## Dashboard Controls

- Input box
- Mode selector: Chat, Code, Image, Pipeline
- Output panel
- Copy Output
- Reset / Clear Output
- Show Raw JSON
- Execution Stream
- Profiler
- Model Manager

## API

```http
POST /api/run-multi-agent
Content-Type: application/json
```

```json
{
  "prompt": "Your prompt here",
  "mode": "pipeline",
  "extra": null
}
```

Supported modes:

- `pipeline` runs Reader -> Summarizer -> Checker
- `chat` runs ConversationAgent
- `code` runs CodeAgent
- `image` runs ImageGenAgent

Response:

```json
{
  "output": "Text output",
  "image_url": null,
  "profile": {
    "total_time_seconds": 0,
    "task_count": 0,
    "total_tokens": 0,
    "tasks": []
  }
}
```

## Programmatic Usage

```python
from backend.orchestrator.orchestrator import Orchestrator

orchestrator = Orchestrator()
output, image_url, profile = orchestrator.run(
    "path/to/input.txt",
    mode="pipeline",
    use_fake_hardware=True,
)
```

## Notes

Legacy apps, experiments, virtual environments, generated logs, and old tests are kept in `archive/`.
