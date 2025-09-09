# Use lightweight, stable python image FROM python:3.11-slim
FROM python:3.11-slim

# --- Proxy (Build-Args) ---
ARG USE_PROXY=false
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY


# --- CA-Zertifikate nur, wenn Proxy aktiv ---
RUN set -eux; \
    if [ "$USE_PROXY" = "true" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends ca-certificates && \
        rm -rf /var/lib/apt/lists/* && \
        mkdir -p /usr/local/share/ca-certificates && \
        echo "Proxy aktiv -> Zertifikate installiert"; \
    else \
        echo "Proxy inaktiv -> keine Zertifikate installiert"; \
    fi

# --- Zertifikate registrieren nur wenn vorhanden ---
COPY certs/*.crt /usr/local/share/ca-certificates/
RUN if [ -d /usr/local/share/ca-certificates ] && [ "$(ls -A /usr/local/share/ca-certificates 2>/dev/null)" ]; then \
        update-ca-certificates; \
    fi


# --- Pip-Config (Proxy nur wenn aktiv) ---
RUN set -eux; \
    mkdir -p /etc/pip; \
    printf "[global]\ntrusted-host = pypi.org\n    files.pythonhosted.org\n" > /etc/pip/pip.conf; \
    if [ "$USE_PROXY" = "true" ]; then \
        printf "proxy = %s\n" "$HTTP_PROXY" >> /etc/pip/pip.conf; \
        echo "Proxy-Einstellungen für Pip gesetzt"; \
    else \
        echo "Kein Proxy für Pip"; \
    fi


# --- Python Env Settings ---
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# --- Requirements installieren ---
COPY requirements.txt /app/requirements.txt
RUN set -eux; \
    if [ "$USE_PROXY" = "true" ]; then \
        export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" no_proxy="$NO_PROXY"; \
    fi; \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt


# --- App kopieren ---
COPY app /app/app

# --- Port freigeben ---
EXPOSE 8100

# --- Start FastAPI Server ---
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8100"]
