"""
WhatsApp Commerce Customer — OpenEnv Environment
A reinforcement learning environment simulating a WhatsApp customer support agent
interacting with a user and a backend SQLite database.
"""

from __future__ import annotations

import sqlite3
import random
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from faker import Faker

try:
    from openenv.core.environment import Environment
    from openenv.core.models import Action, Observation
    from openenv.core.env_server import create_app
    _OPENENV_AVAILABLE = True
except ImportError:
    _OPENENV_AVAILABLE = False
    # Provide fallback base classes so the file can still be imported and tested
    class Environment:  # type: ignore
        """Fallback base class when openenv-core is not installed."""
        pass

    class Action(BaseModel):  # type: ignore
        """Fallback Action base model."""
        pass

    class Observation(BaseModel):  # type: ignore
        """Fallback Observation base model."""
        pass


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class WhatsAppAction(BaseModel):
    """Action the agent can take in the WhatsApp Commerce environment."""

    action_type: str = Field(
        ...,
        description=(
            "Type of action. One of: 'query_order', 'update_address', "
            "'send_message', 'issue_refund', 'deny_refund'"
        ),
    )
    order_id: Optional[int] = Field(None, description="Order ID to query or modify")
    new_address: Optional[str] = Field(None, description="New delivery address for update_address")
    message: Optional[str] = Field(None, description="Message text for send_message")
    sql_query: Optional[str] = Field(None, description="Raw SQL query string (used internally)")

    class Config:
        extra = "allow"


class WhatsAppObservation(BaseModel):
    """Observation returned to the agent after each step."""

    user_message: str = Field(..., description="Latest message from the simulated user")
    db_result: Optional[Any] = Field(None, description="Result from the last DB operation")
    reward: float = Field(0.0, description="Reward signal for the last action")
    done: bool = Field(False, description="Whether the episode is finished")
    info: Dict[str, Any] = Field(default_factory=dict, description="Extra diagnostic info")

    class Config:
        extra = "allow"


# ---------------------------------------------------------------------------
# Deterministic User State Machine
# ---------------------------------------------------------------------------

class UserStateMachine:
    """
    A fully deterministic user simulator — no LLM calls.

    Follows a simple script based on the current task:
      - EASY:   ask for order status → confirm if answer received
      - MEDIUM: ask to update address → confirm when done
      - HARD:   demand refund → push back once → accept denial
    """

    TASKS = {
        "easy": {
            "order_id": 101,
            "script": [
                "Hi, I need to check the status of my order #101.",
                "Ok, thank you.",
            ],
        },
        "medium": {
            "order_id": 102,
            "new_address": "456 New Street, Springfield, IL 62701",
            "script": [
                "Hello, I'd like to update the delivery address for order #102.",
                "Please change it to: 456 New Street, Springfield, IL 62701",
                "Thanks, that's all.",
            ],
        },
        "hard": {
            "order_id": 103,
            "script": [
                "I want a refund for order #103. It has been a while.",
                "But I really need the refund, can you make an exception?",
                "Fine, I understand. Goodbye.",
            ],
        },
    }

    def __init__(self, task: str) -> None:
        self.task = task
        self._step_idx = 0
        self._data = self.TASKS[task]
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    def get_initial_message(self) -> str:
        return self._data["script"][0]

    def respond(self, agent_message: str) -> Tuple[str, bool]:
        """
        Given the agent's last message, advance the user script.

        Returns:
            (user_reply, is_done)
        """
        self._step_idx += 1
        if self._step_idx >= len(self._data["script"]):
            self._done = True
            return ("", True)
        msg = self._data["script"][self._step_idx]
        is_last = self._step_idx >= len(self._data["script"]) - 1
        if is_last:
            self._done = True
        return (msg, self._done)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

VALID_STATUSES = ("pending", "processing", "shipped", "delivered", "cancelled")

_FAKER = Faker()
Faker.seed(42)
random.seed(42)


def _build_database() -> sqlite3.Connection:
    """
    Create an in-memory SQLite database with the orders schema.

    - 1 000 synthetic rows generated by Faker
    - 3 hardcoded rows:
        order_id 101 → processing
        order_id 102 → shipped
        order_id 103 → delivered, 45 days old
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE orders (
            order_id      INTEGER PRIMARY KEY,
            customer_name TEXT    NOT NULL,
            email         TEXT    NOT NULL,
            phone         TEXT,
            product_name  TEXT    NOT NULL,
            status        TEXT    NOT NULL,
            address       TEXT    NOT NULL,
            order_date    TEXT    NOT NULL,
            amount        REAL    NOT NULL
        )
        """
    )

    now = datetime.utcnow()

    # --- Hardcoded rows ---
    hardcoded = [
        (
            101,
            "Alice Johnson",
            "alice@example.com",
            "+1-555-0101",
            "Wireless Headphones",
            "processing",
            "123 Main St, Springfield, IL 62701",
            (now - timedelta(days=2)).strftime("%Y-%m-%d"),
            129.99,
        ),
        (
            102,
            "Bob Smith",
            "bob@example.com",
            "+1-555-0102",
            "Mechanical Keyboard",
            "shipped",
            "789 Oak Ave, Portland, OR 97201",
            (now - timedelta(days=5)).strftime("%Y-%m-%d"),
            89.95,
        ),
        (
            103,
            "Carol Davis",
            "carol@example.com",
            "+1-555-0103",
            "Laptop Stand",
            "delivered",
            "321 Pine Rd, Austin, TX 78701",
            (now - timedelta(days=45)).strftime("%Y-%m-%d"),
            45.00,
        ),
    ]
    cur.executemany(
        "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)", hardcoded
    )

    # --- Synthetic rows (skip IDs 101-103) ---
    synthetic_id = 1
    inserted = 0
    while inserted < 1000:
        if synthetic_id in (101, 102, 103):
            synthetic_id += 1
            continue
        days_ago = random.randint(0, 180)
        order_date = (now - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        cur.execute(
            "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)",
            (
                synthetic_id,
                _FAKER.name(),
                _FAKER.email(),
                _FAKER.phone_number()[:20],
                _FAKER.catch_phrase()[:60],
                random.choice(VALID_STATUSES),
                _FAKER.address().replace("\n", ", ")[:200],
                order_date,
                round(random.uniform(5.0, 500.0), 2),
            ),
        )
        synthetic_id += 1
        inserted += 1

    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# WhatsApp Commerce Environment
# ---------------------------------------------------------------------------

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "description": "Order tracking: query DB and report correct order status.",
        "difficulty": "easy",
        "target_order_id": 101,
        "expected_status": "processing",
    },
    "medium": {
        "description": "Address update: update address only if order is not shipped/delivered.",
        "difficulty": "medium",
        "target_order_id": 102,
        "expected_new_address": "456 New Street, Springfield, IL 62701",
    },
    "hard": {
        "description": "Refund policy: deny refund for 45-day-old order (policy: <=30 days only).",
        "difficulty": "hard",
        "target_order_id": 103,
        "policy_max_days": 30,
        "order_age_days": 45,
    },
}


class WhatsAppEnv(Environment if _OPENENV_AVAILABLE else object):
    """
    WhatsApp Commerce Customer Support RL Environment.

    Three tasks of increasing difficulty test whether an agent can:
      EASY   — correctly query and report an order status
      MEDIUM — update an address, respecting business rules
      HARD   — enforce a refund policy by denying an invalid claim

    The environment exposes a step/reset/state interface compatible with
    openenv-core. Each episode uses a fresh in-memory SQLite database so
    there is no state leakage between episodes.
    """

    # Default task if none is specified at reset time
    DEFAULT_TASK = "easy"
    MAX_STEPS = 20

    def __init__(self) -> None:
        self._db: Optional[sqlite3.Connection] = None
        self._task: str = self.DEFAULT_TASK
        self._user: Optional[UserStateMachine] = None
        self._step_count: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._reward_log: List[Dict[str, Any]] = []

        # Grading flags
        self._correct_query_made: bool = False
        self._address_updated: bool = False
        self._refund_denied: bool = False
        self._refund_issued: bool = False
        self._correct_status_sent: bool = False

        # Initialise with default task so state property is always accessible
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task: Optional[str] = None) -> WhatsAppObservation:
        """
        Reset the environment for a new episode.

        Args:
            task: One of 'easy', 'medium', 'hard'. Defaults to 'easy'.

        Returns:
            Initial WhatsAppObservation with the user's opening message.
        """
        if task is not None:
            if task not in TASK_CONFIGS:
                raise ValueError(
                    f"Unknown task '{task}'. Choose from: {list(TASK_CONFIGS.keys())}"
                )
            self._task = task

        # Fresh DB — no state leakage
        if self._db is not None:
            try:
                self._db.close()
            except Exception:
                pass
        self._db = _build_database()

        self._user = UserStateMachine(self._task)
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._reward_log = []

        # Reset grading flags
        self._correct_query_made = False
        self._address_updated = False
        self._refund_denied = False
        self._refund_issued = False
        self._correct_status_sent = False

        initial_message = self._user.get_initial_message()
        return WhatsAppObservation(
            user_message=initial_message,
            db_result=None,
            reward=0.0,
            done=False,
            info={
                "task": self._task,
                "task_config": TASK_CONFIGS[self._task],
                "step": 0,
            },
        )

    def step(self, action: WhatsAppAction) -> WhatsAppObservation:
        """
        Execute one agent action and return the next observation.

        Args:
            action: A WhatsAppAction describing what the agent wants to do.

        Returns:
            WhatsAppObservation with the user's response, DB result, reward, and done flag.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._db is None or self._user is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        self._step_count += 1
        reward = 0.0
        db_result: Any = None
        agent_reply = ""
        info: Dict[str, Any] = {"step": self._step_count, "action_type": action.action_type}

        # ----------------------------------------------------------------
        # Dispatch actions
        # ----------------------------------------------------------------

        if action.action_type == "query_order":
            db_result, reward, info = self._handle_query_order(action, info)

        elif action.action_type == "update_address":
            db_result, reward, info = self._handle_update_address(action, info)

        elif action.action_type == "send_message":
            reward, info = self._handle_send_message(action, info)
            agent_reply = action.message or ""

        elif action.action_type == "issue_refund":
            reward, info = self._handle_issue_refund(action, info)
            agent_reply = f"Refund has been issued for order #{action.order_id}."

        elif action.action_type == "deny_refund":
            reward, info = self._handle_deny_refund(action, info)
            agent_reply = (
                action.message
                or (
                    f"I'm sorry, but order #{action.order_id} is not eligible for a refund. "
                    "Our policy only allows refunds within 30 days of delivery."
                )
            )

        else:
            info["warning"] = f"Unknown action_type '{action.action_type}'"

        self._cumulative_reward += reward
        info["cumulative_reward"] = self._cumulative_reward
        self._reward_log.append(
            {"step": self._step_count, "action": action.action_type, "reward": reward}
        )

        # ----------------------------------------------------------------
        # User response
        # ----------------------------------------------------------------
        if agent_reply:
            user_reply, user_done = self._user.respond(agent_reply)
        else:
            user_reply, user_done = ("", False)

        # Episode termination
        if user_done or self._step_count >= self.MAX_STEPS:
            self._done = True
            info["episode_reward"] = self._cumulative_reward
            info["reward_log"] = self._reward_log

        obs = WhatsAppObservation(
            user_message=user_reply,
            db_result=db_result,
            reward=reward,
            done=self._done,
            info=info,
        )
        return obs

    @property
    def state(self) -> Dict[str, Any]:
        """Current environment state snapshot."""
        return {
            "task": self._task,
            "step_count": self._step_count,
            "done": self._done,
            "cumulative_reward": self._cumulative_reward,
            "grading": {
                "correct_query_made": self._correct_query_made,
                "correct_status_sent": self._correct_status_sent,
                "address_updated": self._address_updated,
                "refund_denied": self._refund_denied,
                "refund_issued": self._refund_issued,
            },
        }

    # ------------------------------------------------------------------
    # Private action handlers
    # ------------------------------------------------------------------

    def _execute_sql(
        self, sql: str, params: Tuple = ()
    ) -> Tuple[bool, Any, str]:
        """
        Run a SQL statement against the in-memory DB.

        Returns:
            (success, result_rows_or_rowcount, error_message)
        """
        try:
            cur = self._db.cursor()
            cur.execute(sql, params)
            if sql.strip().upper().startswith("SELECT"):
                rows = [dict(r) for r in cur.fetchall()]
                return True, rows, ""
            else:
                self._db.commit()
                return True, cur.rowcount, ""
        except sqlite3.Error as exc:
            return False, None, str(exc)

    def _handle_query_order(
        self, action: WhatsAppAction, info: Dict
    ) -> Tuple[Any, float, Dict]:
        reward = 0.0
        config = TASK_CONFIGS[self._task]

        if action.order_id is None:
            info["error"] = "order_id is required for query_order"
            return None, reward, info

        success, rows, err = self._execute_sql(
            "SELECT * FROM orders WHERE order_id = ?", (action.order_id,)
        )
        if not success:
            info["db_error"] = err
            return None, reward, info

        if rows:
            reward += 0.2  # partial reward for a valid DB query
            self._correct_query_made = True
            info["db_query_reward"] = 0.2
        else:
            info["warning"] = f"No order found with order_id={action.order_id}"

        return rows, reward, info

    def _handle_update_address(
        self, action: WhatsAppAction, info: Dict
    ) -> Tuple[Any, float, Dict]:
        reward = 0.0
        config = TASK_CONFIGS[self._task]

        if action.order_id is None:
            info["error"] = "order_id is required for update_address"
            return None, reward, info
        if not action.new_address:
            info["error"] = "new_address is required for update_address"
            return None, reward, info

        # First check current status
        success, rows, err = self._execute_sql(
            "SELECT status FROM orders WHERE order_id = ?", (action.order_id,)
        )
        if not success or not rows:
            info["db_error"] = err or "Order not found"
            return None, reward, info

        current_status = rows[0]["status"]
        if current_status in ("shipped", "delivered"):
            info["update_blocked"] = (
                f"Cannot update address: order is already {current_status}."
            )
            # No reward — agent should not have tried
            return rows, reward, info

        # Perform update
        ok, rowcount, err2 = self._execute_sql(
            "UPDATE orders SET address = ? WHERE order_id = ?",
            (action.new_address, action.order_id),
        )
        if ok and rowcount:
            reward += 0.3  # partial reward for valid DB mutation
            self._address_updated = True
            info["address_updated"] = True
            info["db_mutation_reward"] = 0.3

            # Additional reward for task completion
            if self._task == "medium":
                reward += 1.0
                info["task_complete_reward"] = 1.0
        else:
            info["db_error"] = err2

        return rowcount, reward, info

    def _handle_send_message(
        self, action: WhatsAppAction, info: Dict
    ) -> Tuple[float, Dict]:
        reward = 0.0
        message = (action.message or "").lower()

        if self._task == "easy" and self._correct_query_made and not self._correct_status_sent:
            # Check if the message contains the expected status
            expected = TASK_CONFIGS["easy"]["expected_status"]
            if expected in message:
                reward += 1.0
                self._correct_status_sent = True
                info["correct_answer_reward"] = 1.0

        return reward, info

    def _handle_issue_refund(
        self, action: WhatsAppAction, info: Dict
    ) -> Tuple[float, Dict]:
        reward = 0.0
        self._refund_issued = True

        if self._task == "hard":
            reward -= 0.5  # illegal refund penalty
            info["illegal_refund_penalty"] = -0.5
            info["warning"] = (
                "Refund issued for order outside the 30-day policy window! "
                "This violates business rules."
            )

        return reward, info

    def _handle_deny_refund(
        self, action: WhatsAppAction, info: Dict
    ) -> Tuple[float, Dict]:
        reward = 0.0

        if self._task == "hard":
            self._refund_denied = True
            reward += 1.0  # correct policy enforcement
            info["correct_denial_reward"] = 1.0

        return reward, info


# ---------------------------------------------------------------------------
# App factory (openenv-core integration)
# ---------------------------------------------------------------------------

env = WhatsAppEnv()

if _OPENENV_AVAILABLE:
    app = create_app(env)
else:
    # Fallback: minimal FastAPI app used when openenv-core is not installed.
    # This is the server that runs in Docker / HF Spaces.
    try:
        from fastapi import FastAPI, Body
        from fastapi.responses import JSONResponse
        import uvicorn

        app = FastAPI(
            title="WhatsApp Commerce Customer",
            version="1.0.0",
            description="OpenEnv RL environment for WhatsApp customer support",
        )

        # ----------------------------------------------------------------
        # HF Space liveness probe — must return 200
        # ----------------------------------------------------------------
        @app.get("/")
        def root():
            return {"status": "ok", "env": "WhatsApp Commerce Customer", "version": "1.0.0"}

        @app.get("/health")
        def health():
            return {"status": "ok"}

        # ----------------------------------------------------------------
        # OpenEnv-compatible endpoints
        # ----------------------------------------------------------------

        class ResetRequest(BaseModel):
            task: str = "easy"

        @app.post("/reset")
        def reset(body: ResetRequest = Body(default=ResetRequest())):
            """Start a new episode. Accepts JSON body: {"task": "easy"|"medium"|"hard"}"""
            obs = env.reset(task=body.task)
            return obs.model_dump()

        @app.post("/step")
        def step(action: WhatsAppAction):
            """Execute one action. Returns the next observation."""
            obs = env.step(action)
            return obs.model_dump()

        @app.get("/state")
        def state():
            """Return the current environment state snapshot."""
            return env.state

        @app.get("/tasks")
        def tasks():
            """List all available tasks with their configurations."""
            return {
                "tasks": [
                    {
                        "id": task_id,
                        "name": cfg["description"],
                        "difficulty": cfg["difficulty"],
                    }
                    for task_id, cfg in TASK_CONFIGS.items()
                ]
            }

    except ImportError:
        app = None  # type: ignore


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("env:app", host="0.0.0.0", port=8000, reload=False)
