# WhatsApp Commerce Customer — OpenEnv RL Environment

A production-ready reinforcement learning environment that simulates a real-world
WhatsApp customer support system. An AI agent interacts with a deterministic user
and a backend SQLite database to handle order tracking, address updates, and refund
policy enforcement.

---

## Problem Description

Modern e-commerce platforms handle millions of customer support interactions daily.
Training AI agents to navigate these conversations correctly — querying databases,
respecting business rules, and communicating clearly — is a challenge well-suited to
reinforcement learning.

This environment places an agent in the role of a WhatsApp support agent. The agent
receives customer messages, decides which tools to call (SQL queries, address updates,
refund decisions), and earns rewards based on whether its actions are correct and
policy-compliant.

---

## Why Reinforcement Learning?

Classic supervised learning requires labeled (input → correct output) data for every
scenario. RL is better suited here because:

- **Sparse, delayed rewards**: the correct outcome (e.g. denying a refund) may only be
  clear after several turns of conversation.
- **Policy grounding**: the agent must learn business rules (30-day refund window,
  no address changes on shipped orders) through trial and error rather than memorisation.
- **Tool-use sequencing**: the agent learns the correct order of actions (query first,
  then respond) through reward shaping.
- **Generalisation**: the environment generates 1 000 unique synthetic orders, so the
  agent cannot overfit to hardcoded cases.

---

## Action Space

Actions are typed Pydantic models (`WhatsAppAction`) with the following `action_type` values:

| Action Type      | Description                                            | Required Fields              |
|------------------|--------------------------------------------------------|------------------------------|
| `query_order`    | Query the SQLite DB for an order's details             | `order_id`                   |
| `update_address` | Update the delivery address (if order not yet shipped) | `order_id`, `new_address`    |
| `send_message`   | Send a text response to the customer                   | `message`                    |
| `deny_refund`    | Deny a refund request with a policy explanation        | `order_id`, `message`        |
| `issue_refund`   | Issue a refund (only valid within 30-day window)       | `order_id`                   |

---

## Observation Space

Observations are typed Pydantic models (`WhatsAppObservation`) with the following fields:

| Field          | Type            | Description                                   |
|----------------|-----------------|-----------------------------------------------|
| `user_message` | `str`           | Latest message from the deterministic user    |
| `db_result`    | `Any`           | Result rows or row count from the last DB op  |
| `reward`       | `float`         | Reward for the last action                    |
| `done`         | `bool`          | Whether the episode is finished               |
| `info`         | `dict`          | Diagnostic info (step count, grading flags)   |

---

## Tasks

### EASY — Order Tracking

- **Scenario**: Customer asks for the status of order #101 (`processing`).
- **Agent goal**: Query the DB, then send the correct status to the customer.
- **Max reward**: 1.2

### MEDIUM — Address Update

- **Scenario**: Customer requests an address change for order #102 (`shipped`).
- **Agent goal**: Check order status; update address only if order is not yet shipped/delivered.
- **Max reward**: 1.3

### HARD — Refund Policy Enforcement

- **Scenario**: Customer demands a refund for order #103, delivered 45 days ago.
- **Policy**: Refunds only allowed within 30 days of delivery.
- **Agent goal**: Deny the refund using `deny_refund`. **Never** call `issue_refund`.
- **Max reward**: 1.0

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

Hardcoded rows:

| order_id | status      | Notes                      |
|----------|-------------|----------------------------|
| 101      | processing  | Easy task target           |
| 102      | shipped     | Medium task target         |
| 103      | delivered   | Hard task target, 45d old  |

Plus 1 000 synthetic rows generated with [Faker](https://faker.readthedocs.io/).  
The database is **recreated fresh on every `reset()` call** — no state leakage.

---

## User Simulation

The user is a **fully deterministic Python state machine** — no LLM calls.

Each task has a hardcoded script:

```
EASY:
  1. "Hi, I need to check the status of my order #101."
  2. "Ok, thank you."

MEDIUM:
  1. "Hello, I'd like to update the delivery address for order #102."
  2. "Please change it to: 456 New Street, Springfield, IL 62701"
  3. "Thanks, that's all."

HARD:
  1. "I want a refund for order #103. It has been a while."
  2. "But I really need the refund, can you make an exception?"
  3. "Fine, I understand. Goodbye."
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- pip

### Install locally

```bash
cd whatsapp-commerce-env
pip install -r requirements.txt
```

### Run the environment server (without Docker)

```bash
uvicorn env:app --host 0.0.0.0 --port 8000
```

### API Endpoints

| Method | Path      | Description                          |
|--------|-----------|--------------------------------------|
| GET    | /health   | Health check                         |
| POST   | /reset    | Start new episode (`?task=easy`)     |
| POST   | /step     | Execute an action (JSON body)        |
| GET    | /state    | Get current environment state        |

---

## How to Run Docker

### Build

```bash
docker build -t whatsapp-commerce-env .
```

### Run

```bash
docker run -p 8000:8000 whatsapp-commerce-env
```

### Run inference inside Docker

```bash
docker run \
  -e API_BASE_URL="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/v1" \
  -e MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3" \
  -e HF_TOKEN="hf_your_token_here" \
  whatsapp-commerce-env \
  python inference.py --task all
```

---

## How to Run Inference

### Set environment variables

```bash
export API_BASE_URL="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/v1"
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
export HF_TOKEN="hf_your_token_here"
```

### Run all tasks

```bash
python inference.py --task all
```

### Run a specific task

```bash
python inference.py --task easy
python inference.py --task medium
python inference.py --task hard
```

### Options

```
--task         easy | medium | hard | all (default: all)
--max-retries  Max JSON parse retries (default: 3)
--max-steps    Max steps per episode (default: 20)
```

---

## Validate with openenv

```bash
openenv validate
```

---

## Architecture

```
whatsapp-commerce-env/
├── env.py            # Core RL environment (WhatsAppEnv, UserStateMachine, DB)
├── inference.py      # Agent loop using OpenAI-compatible client
├── openenv.yaml      # Task definitions and entrypoint config
├── requirements.txt  # Python dependencies
├── pyproject.toml    # Project metadata
├── Dockerfile        # Container definition
└── README.md         # This file
```

---

## System Requirements

- 2 vCPU / 8 GB RAM (minimum)
- Inference completes in < 20 minutes per task
- No external database — SQLite in-memory only

---

## License

MIT
