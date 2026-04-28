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

# tini muss IMMER da sein, weil ENTRYPOINT es nutzt
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p "$LOG_DIR" /app/offload && chmod -R 777 "$LOG_DIR" /app/offload
ENV OFFLOAD_FOLDER=/app/offload

ENTRYPOINT ["/usr/bin/tini","-g","--"]

# Zertifikate optional
COPY certs/ /usr/local/share/ca-certificates/
RUN set -eu; \
    if find /usr/local/share/ca-certificates -type f -name '*.crt' -print -quit | grep -q .; then \
        update-ca-certificates; \
    fi

# pip Proxy optional
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

# devel ist sinnvoll, weil wir llama-cpp-python mit CUDA bauen wollen (nvcc + header + toolchain)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential cmake ninja-build \
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
# (Ignoriert für Stabilität, unverändert lassen)
########################################
FROM base_cuda AS apertus70b

RUN mkdir -p "$LOG_DIR/apertus70b-inference" && chmod -R 777 "$LOG_DIR/apertus70b-inference"

ENV CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_USE_GRAPHS=on -DGGML_CUDA_ENABLE_VMM=off -DLLAMA_CURL=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAVA_BUILD=OFF" \
    FORCE_CMAKE=1 \
    LLAMA_CPP_BUILD_TYPE=Release \
    CC=gcc \
    CXX=g++

COPY LLMs/Apertus70B/requirements.txt /app/requirements.txt

RUN set -eux; \
    python3 -m pip install --no-cache-dir -U pip setuptools wheel; \
    pip install --no-cache-dir -r /app/requirements.txt; \
    python3 -c "print('Apertus70B build block present; not maintained in this iteration')"

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

COPY utils/ /app/utils/
COPY LLMs/Apertus70B/app /app/app

ENV PORT=8100
EXPOSE 8100
CMD ["sh","-c","uvicorn app.server:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --timeout-keep-alive 120"]

########################################
# ------- Target: Nemotron49B ----------
########################################
FROM base_cuda AS nemotron49b

RUN mkdir -p "$LOG_DIR/nemotron-inference" && chmod -R 777 "$LOG_DIR/nemotron-inference"

# Stabil: kein Git, kein Submodule, nur gepinntes llama-cpp-python
# FORCE_CMAKE sorgt dafür, dass CUDA Build wirklich ausgeführt wird
ENV CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_USE_GRAPHS=on -DGGML_CUDA_ENABLE_VMM=off -DLLAMA_CURL=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF" \
    FORCE_CMAKE=1 \
    LLAMA_CPP_BUILD_TYPE=Release

COPY LLMs/Nemotron49B/requirements.txt /app/requirements.txt

RUN set -eux; \
    python3 -m pip install --no-cache-dir -U pip setuptools wheel; \
    pip install --no-cache-dir -r /app/requirements.txt; \
    \
    pip uninstall -y llama-cpp-python || true; \
    \
    # CUDA driver stub nur während Build (Linker)
    test -f /usr/local/cuda/lib64/stubs/libcuda.so; \
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1; \
    export CUDAToolkit_ROOT=/usr/local/cuda; \
    export LIBRARY_PATH=/usr/local/cuda/lib64/stubs; \
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs; \
    \
    pip install --no-cache-dir "llama-cpp-python==0.3.16"; \
    \
    rm -f /usr/local/cuda/lib64/stubs/libcuda.so.1 || true \

# Runtime CUDA libs
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

COPY utils/ /app/utils/
COPY LLMs/Nemotron49B/app /app/app

ENV PORT=8100
EXPOSE 8100
CMD ["sh","-c","uvicorn app.server:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --timeout-keep-alive 120 --log-level debug --access-log"]

########################################
# ------- Target: Generic Transformers -
########################################
FROM base_cuda AS transformers

RUN mkdir -p "$LOG_DIR/transformers-inference" && chmod -R 777 "$LOG_DIR/transformers-inference"

COPY LLMs/TransformersGeneric/requirements.txt /app/requirements.txt

RUN set -eux; \
    python3 -m pip install --no-cache-dir -U pip setuptools wheel; \
    pip install --no-cache-dir -r /app/requirements.txt

COPY utils/ /app/utils/
COPY LLMs/TransformersGeneric/app /app/app

ENV PORT=8100
ENV MODEL_ID=/app/models/current
ENV MODEL_NAME=TransformersModel
ENV TORCH_DTYPE=bfloat16
ENV TRUST_REMOTE_CODE=true

EXPOSE 8100

CMD ["sh","-c","uvicorn app.server:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --timeout-keep-alive 120 --log-level info --access-log"]