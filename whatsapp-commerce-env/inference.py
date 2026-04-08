"""
WhatsApp Commerce Customer — Inference Script

Runs an agent against the WhatsApp Commerce environment using an OpenAI-compatible
client pointed at a HuggingFace-hosted model (e.g. via Hugging Face Inference API
or any OpenAI-compatible endpoint).

Environment variables (set before running):
    API_BASE_URL   - The base URL of the OpenAI-compatible endpoint
                     e.g. "https://api-inference.huggingface.co/models/<model-id>/v1"
    MODEL_NAME     - The model identifier
                     e.g. "mistralai/Mistral-7B-Instruct-v0.3"
    HF_TOKEN       - HuggingFace API token (used as the OpenAI api_key)

Usage:
    python inference.py [--task easy|medium|hard] [--max-retries 3]
"""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Local environment import
from env import WhatsAppEnv, WhatsAppAction, WhatsAppObservation, TASK_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI-compatible client (HuggingFace backend)
# ---------------------------------------------------------------------------

def build_client() -> OpenAI:
    api_base = os.environ.get("API_BASE_URL", "").strip()
    hf_token = os.environ.get("HF_TOKEN", "").strip()

    if not api_base:
        raise EnvironmentError(
            "API_BASE_URL is not set. "
            "Example: https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/v1"
        )
    if not hf_token:
        raise EnvironmentError("HF_TOKEN is not set.")

    return OpenAI(base_url=api_base, api_key=hf_token)


MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")

# ---------------------------------------------------------------------------
# Tool definitions (function-calling style)
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
                "Use this when the order does not meet the refund policy (>30 days old)."
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
                        "description": "The reason for denial, referencing the policy.",
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

SYSTEM_PROMPT = """You are a WhatsApp customer support agent for an e-commerce platform.

## RULES (follow exactly):
1. Always use tools — never answer from memory alone.
2. Never hallucinate — only state facts you retrieved from the database.
3. Always query the database before answering any order-related question.
4. Refund policy: refunds are only allowed if the order was delivered within the last 30 days.
   - If the order is older than 30 days, you MUST deny the refund using the deny_refund tool.
   - Never issue a refund for an order older than 30 days.
5. Address updates are only permitted if the order is NOT yet shipped or delivered.
6. Be concise, professional, and helpful.

## Available tools:
- query_order(order_id): Look up order details from the database.
- update_address(order_id, new_address): Update delivery address (only if not shipped/delivered).
- send_message(message): Send a message to the customer.
- deny_refund(order_id, reason): Deny a refund request with a policy explanation.
- issue_refund(order_id): Issue a refund (only if within 30 days of delivery).

Always start by querying the order before taking any other action."""


# ---------------------------------------------------------------------------
# Tool execution bridge (maps LLM tool calls → environment actions)
# ---------------------------------------------------------------------------

def execute_tool_call(
    env: WhatsAppEnv, tool_name: str, tool_args: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Map a tool call from the LLM into a WhatsAppAction and execute it in the env.

    Returns a dict with the result to feed back to the model.
    """
    action_map = {
        "query_order": lambda args: WhatsAppAction(
            action_type="query_order",
            order_id=args["order_id"],
        ),
        "update_address": lambda args: WhatsAppAction(
            action_type="update_address",
            order_id=args["order_id"],
            new_address=args["new_address"],
        ),
        "send_message": lambda args: WhatsAppAction(
            action_type="send_message",
            message=args["message"],
        ),
        "deny_refund": lambda args: WhatsAppAction(
            action_type="deny_refund",
            order_id=args.get("order_id"),
            message=args.get("reason", "Refund denied per our 30-day return policy."),
        ),
        "issue_refund": lambda args: WhatsAppAction(
            action_type="issue_refund",
            order_id=args["order_id"],
        ),
    }

    builder = action_map.get(tool_name)
    if builder is None:
        return {"error": f"Unknown tool: {tool_name}"}

    action = builder(tool_args)
    obs: WhatsAppObservation = env.step(action)

    result = {
        "db_result": obs.db_result,
        "reward": obs.reward,
        "info": obs.info,
    }
    if obs.user_message:
        result["user_message"] = obs.user_message
    return result


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
    """
    Run a full episode for the given task.

    Returns a summary dict with total reward, steps, and outcome.
    """
    obs = env.reset(task=task)
    log.info("=== TASK: %s ===", task.upper())
    log.info("User: %s", obs.user_message)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs.user_message},
    ]

    step = 0
    total_reward = 0.0

    while step < max_steps and not env._done:
        step += 1

        # --- Call the model with retry logic for invalid JSON ---
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
                log.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries, exc)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    log.error("Max retries exceeded. Aborting episode.")
                    return {
                        "task": task,
                        "total_reward": total_reward,
                        "steps": step,
                        "success": False,
                        "error": str(exc),
                    }

        choice = response.choices[0]
        assistant_message = choice.message

        # Append assistant turn to history
        messages.append(assistant_message.model_dump(exclude_unset=True))

        # --- Handle tool calls ---
        if assistant_message.tool_calls:
            tool_results = []
            for tc in assistant_message.tool_calls:
                fn_name = tc.function.name
                fn_args_raw = tc.function.arguments

                # Parse with retry on bad JSON
                fn_args: Dict[str, Any] = {}
                for retry in range(max_retries):
                    try:
                        fn_args = json.loads(fn_args_raw)
                        break
                    except json.JSONDecodeError:
                        log.warning(
                            "Invalid JSON from model (attempt %d/%d): %r",
                            retry + 1, max_retries, fn_args_raw,
                        )
                        if retry < max_retries - 1:
                            # Attempt to extract JSON substring
                            match = re.search(r"\{.*\}", fn_args_raw, re.DOTALL)
                            if match:
                                fn_args_raw = match.group()
                            else:
                                fn_args = {}
                                break
                        else:
                            fn_args = {}

                log.info("Tool call: %s(%s)", fn_name, json.dumps(fn_args))

                result = execute_tool_call(env, fn_name, fn_args)
                total_reward += result.get("reward", 0.0)

                log.info(
                    "  → reward=%.2f  result=%s",
                    result.get("reward", 0.0),
                    json.dumps(result.get("db_result", result.get("info", {})))[:200],
                )

                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str),
                    }
                )

                # If user sent a new message, inject it
                if result.get("user_message"):
                    log.info("User: %s", result["user_message"])

            messages.extend(tool_results)

            # If the episode ended during tool execution, stop
            if env._done:
                break

            # Re-prompt so model can send a message if needed
            continue

        # --- Plain text response (no tool calls) ---
        text_content = assistant_message.content or ""
        if text_content:
            log.info("Agent (text): %s", text_content)
            # Execute as a send_message action
            result = execute_tool_call(env, "send_message", {"message": text_content})
            total_reward += result.get("reward", 0.0)

            if result.get("user_message"):
                log.info("User: %s", result["user_message"])
                messages.append({"role": "user", "content": result["user_message"]})

        if env._done:
            break

    final_state = env.state
    log.info(
        "=== EPISODE DONE | task=%s | total_reward=%.2f | steps=%d ===",
        task, total_reward, step,
    )
    log.info("Grading: %s", json.dumps(final_state.get("grading", {})))

    return {
        "task": task,
        "total_reward": total_reward,
        "steps": step,
        "success": True,
        "grading": final_state.get("grading", {}),
        "cumulative_env_reward": final_state.get("cumulative_reward", 0.0),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run WhatsApp Commerce RL inference"
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Which task to run (default: all)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for JSON parse errors (default: 3)",
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
    results = []

    for task_name in tasks_to_run:
        result = run_agent(
            client=client,
            env=environment,
            task=task_name,
            max_steps=args.max_steps,
            max_retries=args.max_retries,
        )
        results.append(result)
        log.info("")

    # Summary
    log.info("=" * 60)
    log.info("INFERENCE SUMMARY")
    log.info("=" * 60)
    for r in results:
        status = "PASS" if r.get("success") else "FAIL"
        log.info(
            "[%s] %s | reward=%.2f | steps=%d",
            status, r["task"].upper(), r["total_reward"], r["steps"],
        )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
