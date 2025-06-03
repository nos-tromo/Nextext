FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev git curl ca-certificates \
    build-essential cmake ffmpeg \
    libboost-all-dev libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

ENV PATH="/root/.local/bin:$PATH"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY . .

RUN uv venv --python=/usr/bin/python3.11 \
    && uv pip install . \
    && uv sync --frozen --no-cache \
    && uv run python -m nextext.utils.spacy_model_loader

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py"]
