# DIZI-AI

Local Multi-Agent Orchestration for Technical Teams and Independent Developers

DIZI-AI is a local-first AI development console designed for engineers, researchers, and technical teams who require transparency, control, and extensibility. It provides a unified environment for running and orchestrating intelligent agents across multiple execution modes, with full visibility into system behavior.

DIZI-AI is positioned in the niche of local AI orchestration and agent-based workflow development. It serves developers who need a structured, inspectable, and modifiable system for building AI-driven processes without relying on cloud APIs or proprietary platforms.

## Core Capabilities

DIZI-AI includes four execution modes:

- Chat: conversational interaction with local models
- Code: code generation, explanation, and refactoring
- Image: local image generation workflows
- Pipeline: multi-agent execution using the Reader -> Summarizer -> Checker chain

All active runtime computation is local. No hosted AI service or API key is required.

## Local Image Generation

If you run an image model locally, DIZI-AI does not need any image API key.

Supported local-first setups include:

- Stable Diffusion through local Diffusers weights
- Stable Diffusion XL local checkpoints
- Flux local checkpoints
- GGUF-based local image models when integrated behind a local runner
- Automatic1111 running on localhost
- ComfyUI-style local backends when connected through a local-only adapter

In this setup:

- The model runs on the user's CPU or GPU
- The pipeline stays on the local machine
- No external image service is contacted
- No API key is needed

This is the preferred method for privacy, security, and enterprise use.

The current ImageAgent tries local generation in this order:

1. `DIZI_LOCAL_IMAGE_MODEL`: a local Diffusers model directory loaded with `local_files_only=True`
2. `DIZI_AUTOMATIC1111_URL`: an optional loopback-only URL such as `http://127.0.0.1:7860`
3. A deterministic local SVG preview written to `frontend/static/generated`

Remote Automatic1111 URLs are rejected. If no local image model is configured, Image mode still returns a local preview file so the dashboard remains functional.

## Purpose and Design Philosophy

Most AI tools abstract away internal logic, making it difficult for developers to understand or modify system behavior. DIZI-AI takes the opposite approach.

It provides:

- Transparent agent execution
- Real-time logs and traceability
- Profiling and hardware simulation
- Local model management
- A modular, extensible agent architecture
- A clean, developer-focused dashboard

The system is designed for users who need clarity, reproducibility, and full control over their AI workflows.

## Why Choose DIZI-AI

### Local-First Architecture

All inference and orchestration occur on the user's machine. This ensures privacy, security, and independence from cloud providers.

### Structured Multi-Agent Orchestration

DIZI-AI is not just a wrapper or chatbot. It is an orchestration engine capable of running multi-step pipelines, agent chains, and custom workflows.

### Developer-Centric Interface

The dashboard provides detailed visibility into:

- Execution flow
- Agent-level logs
- Profiling metrics
- Hardware simulation
- Model loading and management
- Multi-mode outputs
- Raw JSON responses for debugging

This enables precise debugging and optimization.

### Extensible Architecture

Every component can be modified or replaced. Developers can add:

- Custom agents
- New pipelines
- Additional tools
- Alternative local models
- Backend integrations

DIZI-AI is built as a platform, not a closed product.

## Open, Transparent, and Unrestricted

There are no locked features, paywalls, or proprietary barriers. The system is fully inspectable and modifiable.

## Target Users

DIZI-AI is designed for:

- Independent developers
- Research engineers
- Local LLM practitioners
- Technical teams exploring agent-based workflows
- Students and researchers studying AI systems
- Organizations requiring on-premise AI execution

The system is suitable for both experimentation and structured development.

## Why the Ecosystem Cannot Be Replicated

While open-source code can be copied, an ecosystem cannot.

The DIZI-AI ecosystem consists of:

- Its community
- Its documentation
- Its contributors
- Its roadmap
- Its update cadence
- Its design philosophy
- Its architecture
- Its long-term vision

A fork can duplicate the codebase, but it cannot reproduce:

- Leadership
- Development speed
- Future releases
- User adoption
- Support channels
- Platform identity

Forks follow. The original leads.

## Integration With Hugging Face

Hugging Face provides model weights. DIZI-AI provides the system required to use those models effectively.

DIZI-AI adds:

- Orchestration
- Pipelines
- Profiling
- Logging
- Multi-mode execution
- Agent-level reasoning
- A unified dashboard

The two systems complement each other.

## Path to Enterprise Adoption

DIZI-AI is designed with a long-term enterprise roadmap:

- Phase 1: Local Developer Tool - core orchestration engine, dashboard, agents, and profiling
- Phase 2: Team Features - shared pipelines, shared models, multi-user support, and access control
- Phase 3: Enterprise Edition - SSO integration, audit logs, compliance features, and on-prem deployment
- Phase 4: Cloud Platform - hosted orchestration, team collaboration, and scalable pipelines
- Phase 5: Hardware Integration - dedicated on-prem inference appliances for secure corporate environments

This progression mirrors the adoption path of successful open-source platforms.

## Project Structure

```text
backend/   Flask API, orchestrator, profiler, hardware simulation
frontend/  Dashboard UI
agents/    Agent implementations for Chat, Code, Image, and Pipeline
models/    Model manager and local model assets
docs/      Architecture notes and developer guides
archive/   Older files kept out of the active root
```

## Running Locally

```powershell
python backend/server.py
```

Open in a browser:

```text
http://localhost:5001
```

## API Endpoint

```http
POST /api/run-multi-agent
Content-Type: application/json
```

Example request:

```json
{
  "prompt": "Summarize this text",
  "mode": "pipeline",
  "extra": null
}
```

Supported modes:

- `pipeline`
- `chat`
- `code`
- `image`

Example response:

```json
{
  "output": "Result text",
  "image_url": null,
  "profile": {}
}
```

## Local Model Setup

Text and code generation use local GGUF models through `llama-cpp-python` when model files are present under `models/`. The system remains usable without those files by using local deterministic fallbacks.

For local Diffusers image generation:

```powershell
$env:DIZI_LOCAL_IMAGE_MODEL="C:\path\to\local\image-model"
```

For local Automatic1111:

```powershell
$env:DIZI_AUTOMATIC1111_URL="http://127.0.0.1:7860"
```

No hosted service is required for either path.


*As of right now, DIZI‑AI does not include automated model downloads. Users must obtain and install their own models, and full model‑management automation will be introduced in the next development phase.*

## Project Status

DIZI-AI is in active early development. Expect rapid iteration, architectural improvements, and expanded agent capabilities.

Contributions and feedback are welcome.

