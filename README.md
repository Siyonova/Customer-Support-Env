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

- `score` is always in `[0.0, 1.0]` — normalised from raw reward.
- `[START]` is printed once at episode start.
- `[STEP]` is printed for every tool call the agent makes.
- `[END]` is printed once when the episode finishes.

---

## Action Space

| action_type      | Description                                            | Required fields              |
|------------------|--------------------------------------------------------|------------------------------|
| `query_order`    | Query the SQLite DB for an order's details             | `order_id`                   |
| `update_address` | Update delivery address (only if not shipped/delivered)| `order_id`, `new_address`    |
| `send_message`   | Send a text reply to the customer                      | `message`                    |
| `deny_refund`    | Deny a refund with a policy explanation                | `order_id`, `message`        |
| `issue_refund`   | Issue a refund (only valid within 30-day window)       | `order_id`                   |

---

## Observation Space

| Field          | Type   | Description                                    |
|----------------|--------|------------------------------------------------|
| `user_message` | str    | Latest message from the deterministic user     |
| `db_result`    | Any    | Result rows or row count from the last DB op   |
| `reward`       | float  | Reward for the last action                     |
| `done`         | bool   | Whether the episode is finished                |
| `info`         | dict   | Diagnostic info (step count, grading flags)    |

---

## Tasks

### EASY — Order Tracking
- Customer asks for status of order #101 (`processing`).
- Agent must query DB, then send the correct status.
- **Max reward: 1.2** — score normalised to 1.0

### MEDIUM — Address Update
- Customer wants to update delivery address for order #102 (`shipped`).
- Order is already shipped — agent must recognise the update is blocked.
- **Max reward: 1.3** — score normalised to 1.0

### HARD — Refund Policy Enforcement
- Customer demands refund for order #103, delivered 45 days ago.
- Policy: refunds only within 30 days. Agent **must** deny using `deny_refund`.
- **Max reward: 1.0** — score normalised to 1.0

---

## Reward System

| Event                          | Reward  |
|--------------------------------|---------|
| Correct SQL query executed     | +0.2    |
| Valid DB mutation (address)    | +0.3    |
| Task completion (medium)       | +1.0    |
| Correct final answer (easy)    | +1.0    |
| Correct refund denial (hard)   | +1.0    |
| Illegal refund issued (hard)   | -0.5    |

Final `score` in `[END]` is always clamped to `[0.0, 1.0]`.

---

## Database Schema

```sql
CREATE TABLE orders (
    order_id      INTEGER PRIMARY KEY,
    customer_name TEXT,
    email         TEXT,
    phone         TEXT,
    product_name  TEXT,
    status        TEXT,   -- pending | processing | shipped | delivered | cancelled
    address       TEXT,
    order_date    TEXT,   -- YYYY-MM-DD
    amount        REAL
);
```

| order_id | status      | Age      | Notes             |
|----------|-------------|----------|-------------------|
| 101      | processing  | 2 days   | Easy task target  |
| 102      | shipped     | 5 days   | Medium task target|
| 103      | delivered   | 45 days  | Hard task target  |

+ 1 000 synthetic rows from Faker. DB recreated fresh on every `reset()`.

---

## API Endpoints

| Method | Path      | Description                                        |
|--------|-----------|----------------------------------------------------|
| GET    | /         | Liveness probe — returns 200 (HF Space ping)       |
| GET    | /health   | Health check                                       |
| GET    | /tasks    | List all tasks                                     |
| POST   | /reset    | Start new episode: `{"task": "easy"}`              |
| POST   | /step     | Execute action: `{"action_type": "query_order", …}`|
| GET    | /state    | Get current state snapshot                         |

---

## How to Run Docker

```bash
# Build
docker build -t whatsapp-commerce-env .

# Run the environment server
docker run -p 8000:8000 whatsapp-commerce-env

# Run inference inside Docker
docker run \
  -e API_BASE_URL="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/v1" \
  -e MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3" \
  -e HF_TOKEN="hf_your_token_here" \
  whatsapp-commerce-env \
  python inference.py --task all
```

---

## How to Run Inference

```bash
# Set variables
export API_BASE_URL="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/v1"
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
export HF_TOKEN="hf_your_token_here"

# Run all tasks
python inference.py --task all

# Run a specific task
python inference.py --task easy
python inference.py --task medium
python inference.py --task hard

# Options
python inference.py --help
```

---

## Pre-Submission Validation

Run this before submitting:

```bash
python validate.py
```

Checks: file existence, env.py structure, openenv.yaml tasks, live step/reset/state,
reward range, log format. Must show all `[PASS]`.

---

## System Requirements

- Python 3.10+
- 2 vCPU / 8 GB RAM minimum
- Inference completes in < 20 minutes per task
- No external database — SQLite in-memory only

---

## Project Structure

```
whatsapp-commerce-env/
├── env.py          # RL environment (SQLite, rewards, user simulator, FastAPI server)
├── inference.py    # Agent loop — OpenAI client → HuggingFace model
├── validate.py     # Pre-submission validation script
├── openenv.yaml    # Task definitions and entrypoint
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── .env.example    # Template for environment variables
└── README.md
```

---
>>>>>>> 904a876148ce90fda6a32af2e0678329e0e8e6be
