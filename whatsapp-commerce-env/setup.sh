#!/usr/bin/env bash
# setup.sh — Run this once after cloning to prepare the environment.
# Installs uv, generates uv.lock, and installs all dependencies.

set -e

echo "==> Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Make sure uv is on PATH for the rest of this script
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

echo "==> Generating uv.lock..."
uv lock

echo "==> Installing dependencies..."
uv pip install -r requirements.txt

echo ""
echo "✓ Setup complete. You can now run:"
echo "  python inference.py --task all"
