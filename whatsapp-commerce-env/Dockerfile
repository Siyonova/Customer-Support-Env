FROM python:3.10-slim

LABEL maintainer="WhatsApp Commerce Team"
LABEL description="WhatsApp Commerce Customer — OpenEnv RL Environment"
LABEL version="1.0.0"

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY env.py .
COPY inference.py .
COPY openenv.yaml .
COPY pyproject.toml .

# Expose the environment server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the environment server
CMD ["uvicorn", "env:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
