FROM python:3.10-slim

# Avoid Python writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps needed for building and runtime (git for pip git+, build tools for some wheels,
# and libs for image handling/matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    gcc \
    g++ \
    openjdk-17-jre-headless \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps early to leverage Docker layer caching
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install jupyterlab

# Copy project files (datasets/checkpoints are usually mounted, see docker-compose)
COPY . .

EXPOSE 8888

# Start Jupyter Lab (no token, no password; rely on local port mapping)
RUN chmod +x scripts/start-jupyter.sh scripts/start-streamlit.sh || true
CMD ["/bin/bash", "scripts/start-jupyter.sh"]
