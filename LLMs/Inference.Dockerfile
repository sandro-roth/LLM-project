# syntax=docker/dockerfile:1.6
ARG DOCKER_INFERENCE=mistral
# ---------- Base ----------
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

# Logs dir (service subdir created in targets)
RUN mkdir -p "$LOG_DIR" && chmod -R 777 "$LOG_DIR"

# --- Optional CA certs if proxy active ---
RUN set -eux; \
    if [ "$USE_PROXY" = "true" ]; then \
        apt-get update; \
        apt-get install -y --no-install-recommends ca-certificates; \
        rm -rf /var/lib/apt/lists/*; \
        mkdir -p /usr/local/share/ca-certificates; \
        echo "ca-certificates installiert"; \
    else \
        echo "Proxy aus -> keine CA Installation"; \
    fi

# --- Copy any provided certs, register if present ---
COPY certs/ /usr/local/share/ca-certificates/
RUN set -eu; \
    if find /usr/local/share/ca-certificates -type f -name '*.crt' -print -quit | grep -q .; then \
        update-ca-certificates; \
        echo "CA Store aktualisiert"; \
    else \
        echo "Keine .crt im Build Kontext -> nichts zu registrieren"; \
    fi

# --- Pip config (proxy only if active) ---
RUN set -eux; \
    printf "[global]\ntrusted-host = pypi.org\n    files.pythonhosted.org\n" > /etc/pip.conf; \
    if [ "$USE_PROXY" = "true" ]; then \
        printf "proxy = %s\n" "$HTTP_PROXY" >> /etc/pip.conf; \
        echo "Pip Proxy gesetzt"; \
    else \
        echo "Pip ohne Proxy"; \
    fi

WORKDIR /app

# ---------- Target: Mistral ----------
FROM base AS mistral
# service specific logs
RUN mkdir -p "$LOG_DIR/mistral-inference" && chmod -R 777 "$LOG_DIR/mistral-inference"

# requirements + install
COPY LLMs/Mistral7B/requirements.txt /app/requirements.txt
RUN set -eux; \
    if [ "$USE_PROXY" = "true" ]; then \
        export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" no_proxy="$NO_PROXY"; \
    fi; \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# app code
COPY utils/ /app/utils/
COPY LLMs/Mistral7B/app /app/app

# normalize to a single PORT (default 8100)
ENV PORT=8100
EXPOSE 8100
CMD ["sh","-c","uvicorn app.server:app --host 0.0.0.0 --port ${PORT}"]

# ---------- Target: Meditron ----------
FROM base AS meditron
# service specific logs
RUN mkdir -p "$LOG_DIR/meditron-inference" && chmod -R 777 "$LOG_DIR/meditron-inference"

# requirements + install
COPY LLMs/Meditron7B/requirements.txt /app/requirements.txt
RUN set -eux; \
    if [ "$USE_PROXY" = "true" ]; then \
        export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" no_proxy="$NO_PROXY"; \
    fi; \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# app code
COPY utils/ /app/utils/
COPY LLMs/Meditron7B/app/ /app/app/

# normalize to a single PORT (default 8100, was 8200 before)
ENV PORT=8100
EXPOSE 8100
CMD ["sh","-c","uvicorn app.server:app --host 0.0.0.0 --port ${PORT}"]


# ---------- Target: Apertus ----------
FROM base AS apertus
# service specific logs
RUN mkdir -p "$LOG_DIR/apertus-inference" && chmod -R 777 "$LOG_DIR/apertus-inference"

# requirements + install
COPY LLMs/Apertus8B/requirements.txt /app/requirements.txt
RUN set -eux; \
    if [ "$USE_PROXY" = "true" ]; then \
        export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" no_proxy="$NO_PROXY"; \
    fi; \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# app code
COPY utils/ /app/utils/
COPY LLMs/Apertus8B/app /app/app

# normalize to a single PORT (default 8100)
ENV PORT=8100
EXPOSE 8100
CMD ["sh","-c","uvicorn app.server:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --timeout-keep-alive 90"]

# ---------- Target: Qwen3 ----------
FROM base AS qwen3
# service specific logs
RUN mkdir -p "$LOG_DIR/qwen-inference" && chmod -R 777 "$LOG_DIR/qwen-inference"

# ---------- Final: pick target by name ----------
FROM ${DOCKER_INFERENCE} AS final