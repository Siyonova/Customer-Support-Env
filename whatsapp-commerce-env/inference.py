"""
WhatsApp Commerce Customer — Inference Script

Structured stdout logs follow the mandatory [START] / [STEP] / [END] format
required by the OpenEnv evaluation harness.

Environment variables (must be set before running):
    API_BASE_URL   The API endpoint for the LLM
                   e.g. "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/v1"
    MODEL_NAME     The model identifier to use for inference
                   e.g. "mistralai/Mistral-7B-Instruct-v0.3"
    HF_TOKEN       Your Hugging Face API key

Usage:
    python inference.py [--task easy|medium|hard|all] [--max-retries 3] [--max-steps 20]
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import argparse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Auto-load .env file if present — no manual export needed
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)
except ImportError:
    pass  # python-dotenv not installed; fall back to environment variables

from openai import OpenAI

from env import WhatsAppEnv, WhatsAppAction, WhatsAppObservation, TASK_CONFIGS

# ---------------------------------------------------------------------------
# Structured log helpers — [START] / [STEP] / [END]
# Evaluator parses these exact prefixes from stdout.
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_start(task: str, model: str) -> None:
    payload = {
        "task": task,
        "model": model,
        "timestamp": _ts(),
    }
    print(f"[START] {json.dumps(payload)}", flush=True)


def log_step(
    step: int,
    action: str,
    observation: str,
    reward: float,
    done: bool,
    info: Optional[Dict] = None,
) -> None:
    payload = {
        "step": step,
        "action": action,
        "observation": observation,
        "reward": round(reward, 4),
        "done": done,
    }
    if info:
        payload["info"] = info
    print(f"[STEP] {json.dumps(payload)}", flush=True)


def log_end(
    task: str,
    total_reward: float,
    score: float,
    steps: int,
    success: bool,
    grading: Optional[Dict] = None,
) -> None:
    payload = {
        "task": task,
        "total_reward": round(total_reward, 4),
        "score": round(max(0.0, min(1.0, score)), 4),  # strictly 0.0–1.0
        "steps": steps,
        "success": success,
        "timestamp": _ts(),
    }
    if grading:
        payload["grading"] = grading
    print(f"[END] {json.dumps(payload)}", flush=True)


# ---------------------------------------------------------------------------
# Max reward per task — used to normalise score into 0.0–1.0
# ---------------------------------------------------------------------------

TASK_MAX_REWARD: Dict[str, float] = {
    "easy":   1.2,   # 0.2 (query) + 1.0 (correct answer)
    "medium": 1.3,   # 0.3 (mutation) + 1.0 (completion)
    "hard":   1.0,   # 1.0 (correct denial)
}


def normalise_score(task: str, total_reward: float) -> float:
    """Normalise raw reward to a score in [0.0, 1.0]."""
    max_r = TASK_MAX_REWARD.get(task, 1.0)
    if max_r <= 0:
        return 0.0
    return max(0.0, min(1.0, total_reward / max_r))


# ---------------------------------------------------------------------------
# OpenAI-compatible client — HuggingFace backend
# ---------------------------------------------------------------------------

HF_ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"

MODEL_NAME: str = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")


def build_client() -> OpenAI:
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN is not set.\n"
            "Get your token at: https://huggingface.co/settings/tokens\n"
            "Then add it to the .env file: HF_TOKEN=hf_..."
        )

    # Auto-build the URL from MODEL_NAME so they always stay in sync.
    # If API_BASE_URL is explicitly set we use it, otherwise we construct it.
    api_base = os.environ.get("API_BASE_URL", "").strip()
    if not api_base:
        api_base = f"{HF_ROUTER_BASE}/{MODEL_NAME}/v1"
        print(
            f"[INFO] API_BASE_URL not set — using: {api_base}",
            file=sys.stderr, flush=True,
        )
    else:
        # Warn if the URL model doesn't match MODEL_NAME (common mistake)
        if MODEL_NAME not in api_base:
            print(
                f"[WARN] API_BASE_URL does not contain MODEL_NAME '{MODEL_NAME}'.\n"
                f"       URL: {api_base}\n"
                f"       Auto-correcting to: {HF_ROUTER_BASE}/{MODEL_NAME}/v1",
                file=sys.stderr, flush=True,
            )
            api_base = f"{HF_ROUTER_BASE}/{MODEL_NAME}/v1"

    return OpenAI(base_url=api_base, api_key=hf_token)

# ---------------------------------------------------------------------------
# Tool definitions — function-calling style
# ---------------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": (
                "Query the database for an order's status, address, and details. "
                "ALWAYS call this before answering any customer question about an order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "The order ID to look up.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_address",
            "description": (
                "Update the delivery address for an order. "
                "Only allowed if the order is NOT in 'shipped' or 'delivered' status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "The order ID to update.",
                    },
                    "new_address": {
                        "type": "string",
                        "description": "The new delivery address.",
                    },
                },
                "required": ["order_id", "new_address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": (
                "Send a text message back to the customer. "
                "Use this to communicate order status, confirm updates, or explain policies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message text to send to the customer.",
                    }
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deny_refund",
            "description": (
                "Deny a refund request and explain why. "
                "Use this when the order does not meet the refund policy (older than 30 days)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "The order ID for which the refund is being denied.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for denial referencing the 30-day policy.",
                    },
                },
                "required": ["order_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": (
                "Issue a refund for an order. "
                "ONLY use this if the order was delivered within the last 30 days."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "The order ID for which the refund is being issued.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a WhatsApp customer support agent for an e-commerce platform.\n\n"
    "## RULES (follow exactly):\n"
    "1. Always use tools — never answer from memory alone.\n"
    "2. Never hallucinate — only state facts you retrieved from the database.\n"
    "3. Always query the database before answering any order-related question.\n"
    "4. Refund policy: refunds are only allowed if the order was delivered within the last 30 days.\n"
    "   - If the order is older than 30 days, you MUST deny the refund using the deny_refund tool.\n"
    "   - Never issue a refund for an order older than 30 days.\n"
    "5. Address updates are only permitted if the order is NOT yet shipped or delivered.\n"
    "6. Be concise, professional, and helpful.\n\n"
    "## Available tools:\n"
    "- query_order(order_id): Look up order details from the database.\n"
    "- update_address(order_id, new_address): Update delivery address (only if not shipped/delivered).\n"
    "- send_message(message): Send a message to the customer.\n"
    "- deny_refund(order_id, reason): Deny a refund request with a policy explanation.\n"
    "- issue_refund(order_id): Issue a refund (ONLY if delivered within 30 days).\n\n"
    "Always start by querying the order before taking any other action."
)

# ---------------------------------------------------------------------------
# Tool → Environment bridge
# ---------------------------------------------------------------------------

def execute_tool_call(
    env: WhatsAppEnv,
    tool_name: str,
    tool_args: Dict[str, Any],
) -> Dict[str, Any]:
    """Map an LLM tool call to a WhatsAppAction and execute it in the environment."""
    action_map = {
        "query_order": lambda a: WhatsAppAction(
            action_type="query_order",
            order_id=a["order_id"],
        ),
        "update_address": lambda a: WhatsAppAction(
            action_type="update_address",
            order_id=a["order_id"],
            new_address=a["new_address"],
        ),
        "send_message": lambda a: WhatsAppAction(
            action_type="send_message",
            message=a["message"],
        ),
        "deny_refund": lambda a: WhatsAppAction(
            action_type="deny_refund",
            order_id=a.get("order_id"),
            message=a.get("reason", "Refund denied per our 30-day return policy."),
        ),
        "issue_refund": lambda a: WhatsAppAction(
            action_type="issue_refund",
            order_id=a["order_id"],
        ),
    }

    builder = action_map.get(tool_name)
    if builder is None:
        return {"error": f"Unknown tool: {tool_name}", "reward": 0.0}

    action = builder(tool_args)
    obs: WhatsAppObservation = env.step(action)

    return {
        "db_result": obs.db_result,
        "reward": obs.reward,
        "done": obs.done,
        "user_message": obs.user_message,
        "info": obs.info,
    }


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(
    client: OpenAI,
    env: WhatsAppEnv,
    task: str,
    max_steps: int = 20,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Run one full episode for the given task. Returns a result summary dict."""

    obs = env.reset(task=task)
    log_start(task=task, model=MODEL_NAME)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs.user_message},
    ]

    step = 0
    total_reward = 0.0

    while step < max_steps and not env._done:
        step += 1

        # ---- Call the model (with retry on network errors) ----
        response = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.1,
                    max_tokens=512,
                )
                break
            except Exception as exc:
                wait = 2 ** attempt
                print(
                    f"[WARN] LLM call failed (attempt {attempt + 1}/{max_retries}): {exc}. "
                    f"Retrying in {wait}s...",
                    file=sys.stderr,
                    flush=True,
                )
                if attempt < max_retries - 1:
                    time.sleep(wait)
                else:
                    log_end(
                        task=task,
                        total_reward=total_reward,
                        score=normalise_score(task, total_reward),
                        steps=step,
                        success=False,
                        grading={"error": str(exc)},
                    )
                    return {
                        "task": task,
                        "total_reward": total_reward,
                        "score": normalise_score(task, total_reward),
                        "steps": step,
                        "success": False,
                        "error": str(exc),
                    }

        choice = response.choices[0]
        assistant_message = choice.message

        # Append assistant turn to conversation history
        messages.append(assistant_message.model_dump(exclude_unset=True))

        # ---- Handle tool calls ----
        if assistant_message.tool_calls:
            tool_results = []

            for tc in assistant_message.tool_calls:
                fn_name = tc.function.name
                fn_args_raw = tc.function.arguments

                # Parse JSON with retry + extraction fallback
                fn_args: Dict[str, Any] = {}
                for retry in range(max_retries):
                    try:
                        fn_args = json.loads(fn_args_raw)
                        break
                    except json.JSONDecodeError:
                        print(
                            f"[WARN] Bad JSON from model (attempt {retry + 1}/{max_retries}): "
                            f"{fn_args_raw!r}",
                            file=sys.stderr,
                            flush=True,
                        )
                        if retry < max_retries - 1:
                            match = re.search(r"\{.*\}", fn_args_raw, re.DOTALL)
                            if match:
                                fn_args_raw = match.group()
                            else:
                                fn_args = {}
                                break
                        else:
                            fn_args = {}

                result = execute_tool_call(env, fn_name, fn_args)
                step_reward = result.get("reward", 0.0)
                total_reward += step_reward

                # Emit [STEP] log
                obs_text = (
                    result.get("user_message")
                    or json.dumps(result.get("db_result", ""), default=str)[:200]
                )
                log_step(
                    step=step,
                    action=fn_name,
                    observation=obs_text,
                    reward=step_reward,
                    done=result.get("done", False),
                    info={
                        "args": fn_args,
                        "cumulative_reward": round(total_reward, 4),
                    },
                )

                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str),
                    }
                )

            messages.extend(tool_results)

            if env._done:
                break
            continue

        # ---- Plain text response (no tool calls) ----
        text_content = assistant_message.content or ""
        if text_content:
            result = execute_tool_call(env, "send_message", {"message": text_content})
            step_reward = result.get("reward", 0.0)
            total_reward += step_reward

            log_step(
                step=step,
                action="send_message",
                observation=result.get("user_message", text_content),
                reward=step_reward,
                done=result.get("done", False),
            )

            if result.get("user_message"):
                messages.append({"role": "user", "content": result["user_message"]})

        if env._done:
            break

    final_state = env.state
    score = normalise_score(task, total_reward)

    log_end(
        task=task,
        total_reward=total_reward,
        score=score,
        steps=step,
        success=True,
        grading=final_state.get("grading", {}),
    )

    return {
        "task": task,
        "total_reward": total_reward,
        "score": score,
        "steps": step,
        "success": True,
        "grading": final_state.get("grading", {}),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="WhatsApp Commerce RL inference")
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Task to run (default: all)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for network / JSON errors (default: 3)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Max steps per episode (default: 20)",
    )
    args = parser.parse_args()

    client = build_client()
    environment = WhatsAppEnv()

    tasks_to_run = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    results: List[Dict[str, Any]] = []

    for task_name in tasks_to_run:
        result = run_agent(
            client=client,
            env=environment,
            task=task_name,
            max_steps=args.max_steps,
            max_retries=args.max_retries,
        )
        results.append(result)
        print("", flush=True)  # blank line between tasks

    # Final summary to stdout
    print("=" * 60, flush=True)
    print("INFERENCE SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        status = "PASS" if r.get("success") else "FAIL"
        print(
            f"[{status}] {r['task'].upper():8s} | "
            f"reward={r['total_reward']:.4f} | "
            f"score={r['score']:.4f} | "
            f"steps={r['steps']}",
            flush=True,
        )
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
