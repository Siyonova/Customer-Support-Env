# WhatsApp Commerce Customer — OpenEnv RL Environment

A production-ready reinforcement learning environment simulating a WhatsApp customer
support system. An AI agent interacts with a deterministic user and a backend SQLite
database to handle order tracking, address updates, and refund policy enforcement.

---

## Problem Description

Modern e-commerce platforms handle millions of customer support interactions daily.
Training AI agents to navigate these conversations correctly — querying databases,
respecting business rules, and communicating clearly — is a challenge well-suited to RL.

This environment places an agent in the role of a WhatsApp support agent. The agent
receives customer messages, decides which tools to call (SQL queries, address updates,
refund decisions), and earns rewards based on whether its actions are correct and
policy-compliant.

---

## Why Reinforcement Learning?

- **Sparse, delayed rewards** — the correct outcome (denying a refund) may only be clear
  after several turns of conversation.
- **Policy grounding** — the agent learns business rules (30-day refund window, no address
  changes on shipped orders) through trial and error rather than memorisation.
- **Tool-use sequencing** — the agent learns the correct order of actions (query first,
  then respond) through reward shaping.
- **Generalisation** — 1 000 unique synthetic orders prevent overfitting.

---

## Setup (from terminal, inside the repo)

```bash
# 1. Clone and enter the project directory
git clone https://github.com/your-username/your-repo.git
cd your-repo/whatsapp-commerce-env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and fill in environment variables
cp .env.example .env
# Edit .env with your API_BASE_URL, MODEL_NAME, and HF_TOKEN

# 4. Run pre-submission validation
python validate.py

# 5. Start the environment server
uvicorn env:app --host 0.0.0.0 --port 8000 --reload

# 6. In another terminal, run inference
source .env    # or: export API_BASE_URL=... MODEL_NAME=... HF_TOKEN=...
python inference.py --task all
```

---

## HuggingFace Connection

This project uses HuggingFace models via an **OpenAI-compatible API endpoint** — so you
use the standard `openai` Python library, just pointed at HuggingFace's server.

### Set these 3 variables:

```bash
export API_BASE_URL="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/v1"
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
export HF_TOKEN="hf_your_token_here"
```

Get your token at: https://huggingface.co/settings/tokens

---

## Structured Log Format — [START] / [STEP] / [END]

The evaluator parses these exact lines from stdout. Do not modify the format.

```
[START] {"task": "easy", "model": "mistralai/Mistral-7B-Instruct-v0.3", "timestamp": "..."}

[STEP]  {"step": 1, "action": "query_order", "observation": "[{...}]", "reward": 0.2, "done": false, "info": {...}}
[STEP]  {"step": 2, "action": "send_message", "observation": "Ok, thank you.", "reward": 1.0, "done": true, "info": {...}}

[END]   {"task": "easy", "total_reward": 1.2, "score": 1.0, "steps": 2, "success": true, "timestamp": "...", "grading": {...}}
```

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

## License

MIT
