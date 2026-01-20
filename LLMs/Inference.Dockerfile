ARG DOCKER_INFERENCE=mistral

########################################
# ---------- Base (CPU / slim) ----------
########################################
FROM python:3.11-slim AS base

# --- Proxy (Build-Args) ---
ARG USE_PROXY=false
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# --- Logging (Build-Arg → ENV) ---
ARG LOG_DIR=/var/log/llm
ENV LOG_DIR=${LOG_DIR}

# --- Common env ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Logs + offload
RUN mkdir -p "$LOG_DIR" /app/offload && chmod -R 777 "$LOG_DIR" /app/offload
ENV OFFLOAD_FOLDER=/app/offload

# tini
ENTRYPOINT ["/usr/bin/tini","-g","--"]

# --- Optional CA certs if proxy active ---
RUN set -eux; \
    if [ "$USE_PROXY" = "true" ]; then \
        apt-get update; \
        apt-get install -y --no-install-recommends curl ca-certificates tini; \
        rm -rf /var/lib/apt/lists/*; \
        mkdir -p /usr/local/share/ca-certificates; \
        echo "ca-certificates installiert"; \
    else \
        echo "Proxy aus -> keine CA Installation"; \
    fi

# --- Copy certs if present ---
COPY certs/ /usr/local/share/ca-certificates/
RUN set -eu; \
    if find /usr/local/share/ca-certificates -type f -name '*.crt' -print -quit | grep -q .; then \
        update-ca-certificates; \
        echo "CA Store aktualisiert"; \
    else \
        echo "Keine .crt im Build Kontext"; \
    fi

# --- Pip config ---
RUN set -eux; \
    printf "[global]\ntrusted-host = pypi.org\n    files.pythonhosted.org\n" > /etc/pip.conf; \
    if [ "$USE_PROXY" = "true" ]; then \
        printf "proxy = %s\n" "$HTTP_PROXY" >> /etc/pip.conf; \
        echo "Pip Proxy gesetzt"; \
    else \
        echo "Pip ohne Proxy"; \
    fi

WORKDIR /app

########################################
# ---------- Base (CUDA) ----------------
########################################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base_cuda

ARG USE_PROXY=false
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ARG LOG_DIR=/var/log/llm
ENV LOG_DIR=${LOG_DIR}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential cmake ninja-build git\
    ca-certificates curl tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p "$LOG_DIR" /app/offload && chmod -R 777 "$LOG_DIR" /app/offload
ENV OFFLOAD_FOLDER=/app/offload

ENTRYPOINT ["/usr/bin/tini","-g","--"]

# --- CA certs ---
COPY certs/ /usr/local/share/ca-certificates/
RUN set -eu; \
    if find /usr/local/share/ca-certificates -type f -name '*.crt' -print -quit | grep -q .; then \
        update-ca-certificates; \
    fi

# --- Pip config ---
RUN set -eux; \
    printf "[global]\ntrusted-host = pypi.org\n    files.pythonhosted.org\n" > /etc/pip.conf; \
    if [ "$USE_PROXY" = "true" ]; then \
        printf "proxy = %s\n" "$HTTP_PROXY" >> /etc/pip.conf; \
    fi

########################################
# ---------- Target: Mistral ------------
########################################
FROM base AS mistral

RUN mkdir -p "$LOG_DIR/mistral-inference" && chmod -R 777 "$LOG_DIR/mistral-inference"

COPY LLMs/Mistral7B/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY utils/ /app/utils/
COPY LLMs/Mistral7B/app /app/app

ENV PORT=8100
EXPOSE 8100
CMD ["sh","-c","uvicorn app.server:app --host 0.0.0.0 --port ${PORT}"]

########################################
# ---------- Target: Meditron ----------
########################################
FROM base AS meditron

RUN mkdir -p "$LOG_DIR/meditron-inference" && chmod -R 777 "$LOG_DIR/meditron-inference"

COPY LLMs/Meditron7B/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY utils/ /app/utils/
COPY LLMs/Meditron7B/app /app/app

ENV PORT=8100
EXPOSE 8100
CMD ["sh","-c","uvicorn app.server:app --host 0.0.0.0 --port ${PORT}"]

########################################
# ---------- Target: Apertus8B ----------
########################################
FROM base AS apertus

RUN mkdir -p "$LOG_DIR/apertus-inference" && chmod -R 777 "$LOG_DIR/apertus-inference"

COPY LLMs/Apertus8B/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY utils/ /app/utils/
COPY LLMs/Apertus8B/app /app/app

ENV PORT=8100
EXPOSE 8100
CMD ["sh","-c","uvicorn app.server:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --timeout-keep-alive 90"]

########################################
# ---------- Target: Qwen3 --------------
########################################
FROM base AS qwen3

RUN mkdir -p "$LOG_DIR/qwen-inference" && chmod -R 777 "$LOG_DIR/qwen-inference"

COPY LLMs/Qwen3/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY utils/ /app/utils/
COPY LLMs/Qwen3/app /app/app

ENV PORT=8100
EXPOSE 8100
CMD ["sh","-c","uvicorn app.server:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --timeout-keep-alive 120 --log-level debug --access-log"]

########################################
# ---------- Target: Apertus70B ---------
########################################
FROM base_cuda AS apertus70b

RUN mkdir -p "$LOG_DIR/apertus70b-inference" && chmod -R 777 "$LOG_DIR/apertus70b-inference"

ENV CMAKE_ARGS="-DGGML_CUDA=on" \
    FORCE_CMAKE=1 \
    LLAMA_CPP_BUILD_TYPE=Release

COPY LLMs/Apertus70B/requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY utils/ /app/utils/
COPY LLMs/Apertus70B/app /app/app

ENV PORT=8100
EXPOSE 8100
CMD ["sh","-c","uvicorn app.server:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --timeout-keep-alive 120"]