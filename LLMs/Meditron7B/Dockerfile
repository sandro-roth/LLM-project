# Lightweight, stable python image
FROM python:3.11-slim

# --- Proxy (Build-Args) ---
ARG USE_PROXY=false
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# --- Logging (Build-Arg → ENV) ---
ARG LOG_DIR=/var/log/llm
ENV LOG_DIR=${LOG_DIR}

# Logs-Ordner sicherstellen
RUN mkdir -p "$LOG_DIR/meditron-inference" && chmod -R 777 "$LOG_DIR"


# --- Nur CA-Zertifikate installieren, wenn Proxy aktiv ---
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


# --- Falls Zertifikate vorhanden: registrieren ---
COPY certs/ /usr/local/share/ca-certificates/
RUN set -eu; \
    if find /usr/local/share/ca-certificates -type f -name '*.crt' -print -quit | grep -q .; then \
        update-ca-certificates; \
        echo "CA Store aktualisiert"; \
    else \
        echo "Keine .crt im Build Kontext -> nichts zu registrieren"; \
    fi


# --- Pip-Config nur mit Proxy ---
RUN set -eux; \
    printf "[global]\ntrusted-host = pypi.org\n    files.pythonhosted.org\n" > /etc/pip.conf; \
    if [ "$USE_PROXY" = "true" ]; then \
        printf "proxy = %s\n" "$HTTP_PROXY" >> /etc/pip.conf; \
        echo "Pip Proxy gesetzt"; \
    else \
        echo "Pip ohne Proxy"; \
    fi


# --- Env config ---
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# --- Requirements installieren ---
COPY LLMs/Meditron7B/requirements.txt /app/requirements.txt
RUN set -eux; \
    if [ "$USE_PROXY" = "true" ]; then \
        export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" no_proxy="$NO_PROXY"; \
    fi; \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt


# Copy app folder into container
COPY utils/ /app/utils/
COPY LLMs/Meditron7B/app/ /app/app/

# Expose port for FastAPI
EXPOSE 8200

# Run FastAPI server
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8200"]