"""
Pre-submission validation script.

Checks:
  1. env.py imports and class structure
  2. openenv.yaml is valid and has 3 tasks
  3. inference.py exists and has required patterns
  4. Environment server starts and responds correctly
  5. step() / reset() / state work
  6. Rewards are in 0.0–1.0 range per task
  7. [START] / [STEP] / [END] log format check

Run:
    python validate.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from typing import Any, Dict, List

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    status = PASS if ok else FAIL
    line = f"{status} {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    if not ok:
        failures += 1


# ---------------------------------------------------------------------------
# 1. File existence
# ---------------------------------------------------------------------------
print("\n── File checks ─────────────────────────────────────────")
import os

for fname in ["env.py", "inference.py", "openenv.yaml", "requirements.txt",
              "pyproject.toml", "Dockerfile"]:
    check(f"File exists: {fname}", os.path.exists(fname))


# ---------------------------------------------------------------------------
# 2. env.py structural checks
# ---------------------------------------------------------------------------
print("\n── env.py checks ───────────────────────────────────────")
with open("env.py") as f:
    env_src = f.read()

check("env.py: imports openenv.core.environment", "from openenv.core.environment import Environment" in env_src)
check("env.py: imports openenv.core.models", "from openenv.core.models import Action, Observation" in env_src)
check("env.py: imports openenv.core.env_server", "from openenv.core.env_server import create_app" in env_src)
check("env.py: class WhatsAppEnv", "class WhatsAppEnv(" in env_src)
check("env.py: def reset(", "def reset(" in env_src)
check("env.py: def step(", "def step(" in env_src)
check("env.py: @property state", "@property" in env_src and "def state(" in env_src)
check("env.py: WhatsAppAction model", "class WhatsAppAction(" in env_src)
check("env.py: WhatsAppObservation model", "class WhatsAppObservation(" in env_src)
check("env.py: SQLite :memory:", 'sqlite3.connect(":memory:"' in env_src)
check("env.py: 1000 synthetic rows", "1000" in env_src)
check("env.py: order_id 101", "101" in env_src)
check("env.py: order_id 102", "102" in env_src)
check("env.py: order_id 103", "103" in env_src)
check("env.py: env = WhatsAppEnv()", "env = WhatsAppEnv()" in env_src)
check("env.py: app = create_app(env)", "app = create_app(env)" in env_src)
check("env.py: GET / endpoint", 'def root(' in env_src or '"/health"' in env_src)
check("env.py: POST /reset", '"/reset"' in env_src or "def reset(" in env_src)


# ---------------------------------------------------------------------------
# 3. inference.py checks
# ---------------------------------------------------------------------------
print("\n── inference.py checks ─────────────────────────────────")
with open("inference.py") as f:
    inf_src = f.read()

check("inference.py: from openai import OpenAI", "from openai import OpenAI" in inf_src)
check("inference.py: API_BASE_URL env var", "API_BASE_URL" in inf_src)
check("inference.py: MODEL_NAME env var", "MODEL_NAME" in inf_src)
check("inference.py: HF_TOKEN env var", "HF_TOKEN" in inf_src)
check("inference.py: base_url= (OpenAI client)", "base_url=" in inf_src)
check("inference.py: api_key= (OpenAI client)", "api_key=" in inf_src)
check("inference.py: tool_calls", "tool_calls" in inf_src)
check("inference.py: json.loads retry logic", inf_src.count("json.loads") >= 1 and "max_retries" in inf_src)
check("inference.py: [START] log", "[START]" in inf_src)
check("inference.py: [STEP] log", "[STEP]" in inf_src)
check("inference.py: [END] log", "[END]" in inf_src)
check("inference.py: Always use tools", "Always use tools" in inf_src)
check("inference.py: Never hallucinate", "Never hallucinate" in inf_src)
check("inference.py: Refund policy 30 days", "30 days" in inf_src or "30-day" in inf_src)
check("inference.py: Always query DB", "Always query" in inf_src or "query the database" in inf_src.lower())
check("inference.py: score normalised 0.0-1.0", "min(1.0" in inf_src or "normalise_score" in inf_src)


# ---------------------------------------------------------------------------
# 4. openenv.yaml checks
# ---------------------------------------------------------------------------
print("\n── openenv.yaml checks ─────────────────────────────────")
try:
    import yaml  # type: ignore
    with open("openenv.yaml") as f:
        cfg = yaml.safe_load(f)
    check("openenv.yaml: parseable YAML", True)
    check("openenv.yaml: entrypoint = env:app", cfg.get("entrypoint") == "env:app")
    tasks = cfg.get("tasks", [])
    check("openenv.yaml: 3 tasks defined", len(tasks) >= 3, f"found {len(tasks)}")
    task_ids = [t.get("id") for t in tasks]
    check("openenv.yaml: easy task", "easy" in task_ids)
    check("openenv.yaml: medium task", "medium" in task_ids)
    check("openenv.yaml: hard task", "hard" in task_ids)
    for t in tasks:
        tid = t.get("id", "?")
        check(f"openenv.yaml: task '{tid}' has reward_spec", "reward_spec" in t)
except ImportError:
    print(f"{WARN} PyYAML not installed — skipping YAML parse checks")
    check("openenv.yaml: contains 'entrypoint: env:app'", "entrypoint: env:app" in open("openenv.yaml").read())
    check("openenv.yaml: contains 3 task ids", all(t in open("openenv.yaml").read() for t in ["easy", "medium", "hard"]))


# ---------------------------------------------------------------------------
# 5. Live environment tests (no LLM — direct Python import)
# ---------------------------------------------------------------------------
print("\n── Live environment tests ──────────────────────────────")
try:
    from env import WhatsAppEnv, WhatsAppAction, TASK_CONFIGS

    for task_name in ["easy", "medium", "hard"]:
        try:
            env_inst = WhatsAppEnv()
            obs = env_inst.reset(task=task_name)
            check(f"reset('{task_name}') returns observation", hasattr(obs, "user_message"))
            check(f"reset('{task_name}') user_message not empty", bool(obs.user_message))

            st = env_inst.state
            check(f"state property works for '{task_name}'", isinstance(st, dict))
            check(f"state.task == '{task_name}'", st.get("task") == task_name)

            # Step with a valid action
            action = WhatsAppAction(action_type="query_order", order_id=TASK_CONFIGS[task_name]["target_order_id"])
            obs2 = env_inst.step(action)
            check(f"step() works for '{task_name}'", hasattr(obs2, "reward"))
            check(f"step() reward is float for '{task_name}'", isinstance(obs2.reward, float))
            check(f"step() reward >= 0 for '{task_name}'", obs2.reward >= 0.0)

        except Exception as exc:
            check(f"task '{task_name}' live test", False, str(exc))

    # Reward range check per task
    print("\n── Reward / score range checks ─────────────────────────")
    from inference import normalise_score, TASK_MAX_REWARD

    for task_name in ["easy", "medium", "hard"]:
        max_r = TASK_MAX_REWARD[task_name]
        score = normalise_score(task_name, max_r)
        check(f"normalise_score('{task_name}', max_reward) == 1.0", abs(score - 1.0) < 1e-9, f"got {score}")
        score_zero = normalise_score(task_name, 0.0)
        check(f"normalise_score('{task_name}', 0.0) == 0.0", score_zero == 0.0, f"got {score_zero}")
        score_neg = normalise_score(task_name, -1.0)
        check(f"normalise_score('{task_name}', -1.0) == 0.0 (clamped)", score_neg == 0.0, f"got {score_neg}")

except Exception as exc:
    check("env.py importable", False, str(exc))


# ---------------------------------------------------------------------------
# 6. [START]/[STEP]/[END] format sanity check (regex)
# ---------------------------------------------------------------------------
print("\n── Log format checks ───────────────────────────────────")
with open("inference.py") as f:
    inf_src = f.read()

start_pat = re.compile(r'\[START\].*json\.dumps', re.DOTALL)
step_pat  = re.compile(r'\[STEP\].*json\.dumps', re.DOTALL)
end_pat   = re.compile(r'\[END\].*json\.dumps', re.DOTALL)

check("log format: [START] emits JSON", bool(start_pat.search(inf_src)))
check("log format: [STEP] emits JSON", bool(step_pat.search(inf_src)))
check("log format: [END] emits JSON", bool(end_pat.search(inf_src)))
check("log format: [END] contains 'score'", "'score'" in inf_src or '"score"' in inf_src)
check("log format: [END] contains 'total_reward'", "total_reward" in inf_src)
check("log format: [END] contains 'steps'", "'steps'" in inf_src or '"steps"' in inf_src)
check("log format: [END] contains 'success'", "'success'" in inf_src or '"success"' in inf_src)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
if failures == 0:
    print("\033[92m✓ All checks passed — ready to submit!\033[0m")
else:
    print(f"\033[91m✗ {failures} check(s) failed — fix before submitting.\033[0m")
print("=" * 55)

sys.exit(0 if failures == 0 else 1)
