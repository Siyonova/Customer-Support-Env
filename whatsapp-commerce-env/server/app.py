"""
server/app.py — OpenEnv multi-mode deployment entry point.

This module is referenced by [project.scripts] in pyproject.toml:
    server = "server.app:main"

It imports the FastAPI app from env.py and exposes it here so OpenEnv's
validator can locate the server entry point at `server.app:app`.
"""

from __future__ import annotations

import os
import sys

# Load .env automatically if present
try:
    from dotenv import load_dotenv
    load_dotenv(
        dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"),
        override=False,
    )
except ImportError:
    pass

# Add the parent directory to sys.path so env.py is importable
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from env import app  # noqa: E402  — imported after path setup


def main() -> None:
    """Start the uvicorn server. Called via `server` console script."""
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
