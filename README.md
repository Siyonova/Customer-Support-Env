---
title: Mosscode
emoji: 😻
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
short_description: OpenEnv WhatsApp Commerce RL Environment
---

# Mosscode

This repository contains an OpenEnv-compatible WhatsApp Commerce environment and inference runner designed for GitHub and Hugging Face Spaces deployment. It implements a simulated WhatsApp customer support RL environment and a FastAPI-style inference entrypoint for model interaction.

## What’s included

- `env.py` — OpenEnv environment implementation for WhatsApp commerce, including state, reset, step, and reward handling.
- `inference.py` — Main inference runner with structured `[START]`, `[STEP]`, and `[END]` logs, model client setup, and tool integration logic.
- `openenv.yaml` — Task definitions and evaluation config for OpenEnv compliance.
- `Dockerfile` — Container build instructions for Hugging Face Spaces / Docker deployment.
- `requirements.txt` and `pyproject.toml` — Python dependencies and project metadata.

## GitHub repository

This code is stored in the GitHub repo:

`https://github.com/Siyonova/Customer-Support-Env`

Use this repository for source control, collaboration, and evaluation submission.

## Hugging Face Spaces deployment

The project was prepared for deployment to Hugging Face Spaces with a Docker-based Space configuration.

### Deployment notes

- The Dockerfile builds a minimal runtime image containing the environment and inference code.
- The Space is configured to run the `inference.py` application and expose it via the Hugging Face runtime.
- The README includes Hugging Face metadata frontmatter so it can be displayed properly in the Space UI.

## Setup and run locally

1. Clone the repository:

```bash
git clone https://github.com/Siyonova/Customer-Support-Env.git
cd Customer-Support-Env
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Set required environment variables:

```bash
export HF_TOKEN="<your-hf-token>"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
```

4. Run the inference entrypoint:

```bash
python inference.py
```

## Environment variables

- `HF_TOKEN` — Hugging Face or OpenAI-compatible API token.
- `API_BASE_URL` — Model API endpoint base URL.
- `MODEL_NAME` — Model name used for inference.

## Notes

- This repository was synchronized from a local `Mosscode` workspace and pushed into `Customer-Support-Env` for GitHub submission.
- It is intended to support both GitHub source distribution and Hugging Face Spaces deployment.
- The code is ready for evaluation and submission once environment variables are configured.

## References

- Hugging Face Spaces config: https://huggingface.co/docs/hub/spaces-config-reference
