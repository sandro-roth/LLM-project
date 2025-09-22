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
RUN mkdir -p $LOG_DIR && chmod 777 $LOG_DIR

# --- CA-Zertifikate nur, wenn Proxy aktiv ---
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

# --- Zertifikate registrieren nur wenn vorhanden ---
COPY certs/ /usr/local/share/ca-certificates/
RUN set -eu; \
    if find /usr/local/share/ca-certificates -type f -name '*.crt' -print -quit | grep -q .; then \
        update-ca-certificates; \
        echo "CA Store aktualisiert"; \
    else \
        echo "Keine .crt im Build Kontext -> nichts zu registrieren"; \
    fi


# --- manchmal hilfreich für Python/Requests ---
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt


# --- Pip-Config (Proxy nur wenn aktiv) ---
RUN set -eux; \
    printf "[global]\ntrusted-host = pypi.org\n    files.pythonhosted.org\n" > /etc/pip.conf; \
    if [ "$USE_PROXY" = "true" ]; then \
        printf "proxy = %s\n" "$HTTP_PROXY" >> /etc/pip.conf; \
        echo "Pip Proxy gesetzt"; \
    else \
        echo "Pip ohne Proxy"; \
    fi


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN set -eux; \
    if [ "$USE_PROXY" = "true" ]; then \
        export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" no_proxy="$NO_PROXY"; \
    fi; \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY app /app

RUN mkdir -p /app/logs

EXPOSE 8501

CMD ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
