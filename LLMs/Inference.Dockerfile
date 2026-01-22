# LLMs/Inference.Dockerfile
ARG DOCKER_INFERENCE=mistral

########################################
# ---------- Base (CPU / slim) ----------
########################################
FROM python:3.11-slim AS base

ARG USE_PROXY=false
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ARG LOG_DIR=/var/log/llm
ENV LOG_DIR=${LOG_DIR}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN mkdir -p "$LOG_DIR" /app/offload && chmod -R 777 "$LOG_DIR" /app/offload
ENV OFFLOAD_FOLDER=/app/offload

ENTRYPOINT ["/usr/bin/tini","-g","--"]

RUN set -eux; \
    if [ "$USE_PROXY" = "true" ]; then \
        apt-get update; \
        apt-get install -y --no-install-recommends curl ca-certificates tini; \
        rm -rf /var/lib/apt/lists/*; \
        mkdir -p /usr/local/share/ca-certificates; \
    else \
        echo "Proxy aus -> keine CA Installation"; \
    fi

COPY certs/ /usr/local/share/ca-certificates/
RUN set -eu; \
    if find /usr/local/share/ca-certificates -type f -name '*.crt' -print -quit | grep -q .; then \
        update-ca-certificates; \
    fi

RUN set -eux; \
    printf "[global]\ntrusted-host = pypi.org\n    files.pythonhosted.org\n" > /etc/pip.conf; \
    if [ "$USE_PROXY" = "true" ]; then \
        printf "proxy = %s\n" "$HTTP_PROXY" >> /etc/pip.conf; \
    fi

WORKDIR /app

########################################
# ---------- Base (CUDA) ----------------
########################################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base_cuda

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
    build-essential cmake ninja-build git \
    ca-certificates curl tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p "$LOG_DIR" /app/offload && chmod -R 777 "$LOG_DIR" /app/offload
ENV OFFLOAD_FOLDER=/app/offload

ENTRYPOINT ["/usr/bin/tini","-g","--"]

COPY certs/ /usr/local/share/ca-certificates/
RUN set -eu; \
    if find /usr/local/share/ca-certificates -type f -name '*.crt' -print -quit | grep -q .; then \
        update-ca-certificates; \
    fi

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
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt
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
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt
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
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt
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
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt
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

# Curl aus, llava aus, Tools aus: weniger Abhängigkeiten und weniger Build-Fläche
ENV CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_USE_GRAPHS=on -DGGML_CUDA_ENABLE_VMM=off -DLLAMA_CURL=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAVA_BUILD=OFF" \
    FORCE_CMAKE=1 \
    LLAMA_CPP_BUILD_TYPE=Release \
    CC=gcc \
    CXX=g++

COPY LLMs/Apertus70B/requirements.txt /app/requirements.txt

RUN set -eux; \
    python3 -m pip install --no-cache-dir -U pip setuptools wheel; \
    pip install --no-cache-dir -r /app/requirements.txt; \
    pip uninstall -y llama-cpp-python || true; \
    \
    LCPP_DIR=/tmp/llama-cpp-python; \
    rm -rf "$LCPP_DIR"; \
    git clone https://github.com/abetlen/llama-cpp-python.git "$LCPP_DIR"; \
    cd "$LCPP_DIR"; \
    \
    # WICHTIG: vendor/llama.cpp auf remote aktualisieren, Workaround für fehlende Symbole wie kv_cache_view_init
    git submodule update --init --recursive --remote; \
    \
    # CUDA driver stub nur während Build
    test -f /usr/local/cuda/lib64/stubs/libcuda.so; \
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1; \
    export CUDAToolkit_ROOT=/usr/local/cuda; \
    export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}; \
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}; \
    \
    pip install --no-cache-dir .; \
    rm -f /usr/local/cuda/lib64/stubs/libcuda.so.1 || true; \
    \
    python3 -c "from llama_cpp import Llama; print('IMPORT_OK')"

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

COPY utils/ /app/utils/
COPY LLMs/Apertus70B/app /app/app

ENV PORT=8100
EXPOSE 8100
CMD ["sh","-c","uvicorn app.server:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --timeout-keep-alive 120"]
