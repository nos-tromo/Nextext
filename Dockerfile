# -------- CPU stage (default) --------
FROM python:3.11-slim AS cpu

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git curl ca-certificates \
    build-essential cmake ffmpeg \
    libboost-all-dev libeigen3-dev \
 && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY . .

RUN rm -f uv.lock \
 && uv sync --no-cache --no-dev \
 && uv run load-spacy-models

EXPOSE 8501

CMD ["uv", "run", "nextext"]

# -------- GPU stage --------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS gpu

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev git curl ca-certificates \
    build-essential cmake ffmpeg \
    libboost-all-dev libeigen3-dev \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

ENV PATH="/root/.local/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY . .

RUN rm -f uv.lock \
 && uv sync --no-cache --no-dev \
 && uv run load-spacy-models

EXPOSE 8501

CMD ["uv", "run", "nextext"]
